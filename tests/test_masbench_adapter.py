"""Offline contract tests for the MASBench-to-Meta-Team adapter."""
from __future__ import annotations

from methods.Evoteam.bootstrap import source_root

import argparse
import asyncio
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
import yaml

from methods.Evoteam.masbench.adapter import MASBenchAdapter, _squad_f1


class MASBenchAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = MASBenchAdapter()
        try:
            cls.source = cls.adapter._source_root()
        except FileNotFoundError as exc:
            raise unittest.SkipTest(str(exc))
        if not cls.adapter.data_dir.exists():
            raise unittest.SkipTest("Set MASBENCH_SPLIT_DIR for integration tests")
        cls.train = cls.adapter.load_dataset(argparse.Namespace(
            split="train", data_dir=str(cls.adapter.data_dir), masbench_root=str(cls.source), max_items=None
        ))
        cls.test = cls.adapter.load_dataset(argparse.Namespace(
            split="test", data_dir=str(cls.adapter.data_dir), masbench_root=str(cls.source), max_items=None
        ))

    def env_for(self, item):
        class Session:
            pass
        session = Session()
        session.workspace = tempfile.mkdtemp(prefix="masbench-test-")
        return self.adapter.setup_environment(item, session)

    def test_split_sizes_and_disjoint_ids(self):
        self.assertEqual(len(self.train), 18)
        self.assertEqual(len(self.test), 90)
        self.assertFalse({self.adapter.get_item_id(x) for x in self.train} &
                         {self.adapter.get_item_id(x) for x in self.test})

    def test_dynamic_six_member_pool_is_default_and_has_official_prompt_files(self):
        pool = Path(__file__).resolve().parents[1] / "methods/Evoteam/masbench/pools/initial"
        self.assertEqual(self.adapter.default_team, "pool_MASBENCH_DYNAMIC_6")
        self.assertEqual(len(list(pool.glob("*/config.yaml"))), 7)
        for name in ("planner", "evidence_agent", "schedule_agent", "game_agent",
                     "verifier", "integrator", "answer_agent"):
            self.assertTrue((pool / name / "prompt.md").is_file(), name)
        self.assertEqual(
            yaml.safe_load((pool / "pool.yaml").read_text(encoding="utf-8"))["settings"]["chairman_forced_submission_seconds"],
            90,
        )

    def test_reasoning_f1_matches_official_ascii_punctuation_normalization(self):
        self.assertEqual(_squad_f1("café", "cafe"), 0.0)
        self.assertEqual(_squad_f1("A, the cat!", "cat"), 1.0)

    def test_runtime_capacity_covers_native_actor_count(self):
        item = {"_actors": ["A", "B", "C", "D", "E", "F"]}
        pool = SimpleNamespace(
            chairman_name="planner",
            agents={name: object() for name in
                    ("planner", "answer_agent", "worker-a", "worker-b", "worker-c", "worker-d", "worker-e")},
        )
        selection = SimpleNamespace(allowed_agent_ids=["worker-a"])
        added = self.adapter.runtime_team_capacity_members(item, pool, selection)
        self.assertEqual(added, ["worker-b", "worker-c", "worker-d", "worker-e"])

    def test_game_setup_message_requires_assignment_and_evolution_redacts_answer(self):
        class RPC:
            def call(self, _operation, **_kwargs):
                return True

        env = type("Env", (), {"data": {
            "bindings": {}, "game_rpc": RPC(), "runner": None,
        }})()
        item = {"category": "game"}
        with self.assertRaises(PermissionError):
            self.adapter.validate_native_message(item, env, "worker-a", "worker-b")
        env.data["bindings"].update({"worker-a": "A", "worker-b": "B"})
        self.adapter.validate_native_message(item, env, "worker-a", "worker-b")

        evaluation = self.adapter.evaluate(
            {"category": "reasoning", "row": {"answer": "confidential answer"}},
            "FINAL ANSWER: confidential answer",
            None,
        )
        reflected = self.adapter.eval_result_for_evolution_reflection(
            {"category": "scheduling"}, evaluation,
        )
        self.assertEqual(reflected.details, {"category": "scheduling"})
        self.assertNotIn("600", json.dumps(reflected.details))

    def test_reasoning_private_context_and_reference_isolation(self):
        item = next(x for x in self.train if x["category"] == "reasoning")
        env = self.env_for(item)
        with tempfile.TemporaryDirectory() as work:
            task = self.adapter.build_task(item, type("S", (), {"workspace": work})(), [])
        self.assertNotIn(item["row"]["answer"], task)
        self.assertNotIn(item["row"]["answer"], json.dumps(env.data["private"]))
        schemas = self.adapter.native_tool_schemas(item, env)
        self.assertIn("claim_actor", [x["function"]["name"] for x in schemas])

        async def check():
            with self.assertRaises(PermissionError):
                await self.adapter.execute_native_tool(item, env, "observe_private", {}, "coordinator")
            await self.adapter.execute_native_tool(item, env, "claim_actor", {"actor": "Alice"}, "worker-a")
            value = await self.adapter.execute_native_tool(item, env, "observe_private", {}, "worker-a")
            self.assertIn("Article:", value)
            with self.assertRaises(PermissionError):
                await self.adapter.execute_native_tool(item, env, "claim_actor", {"actor": "Alice"}, "worker-b")
            with self.assertRaises(PermissionError):
                await self.adapter.execute_native_tool(item, env, "claim_actor", {"actor": "Bob"}, "worker-a")

        asyncio.run(check())
        try:
            result = self.adapter.evaluate(item, "FINAL ANSWER: " + item["row"]["answer"], None)
            self.assertEqual(result.score, 1.0)
            self.assertNotIn("reference", json.dumps(result.details))
        finally:
            self.adapter.teardown_environment(env)

    def test_scheduling_native_exact_slot_and_private_calendar(self):
        item = next(x for x in self.train if x["category"] == "scheduling")
        env = self.env_for(item)
        try:
            plan = json.loads((Path(item["path"]) / "task_plans").read_text())["meeting"]["solutions"][0]
            start, end = plan["start_time"], plan["end_time"]
            answer = {"day": start // 144 + 1, "start_minute": (start % 144) * 10,
                      "end_minute": (end % 144) * 10}
            result = self.adapter.evaluate(item, "FINAL ANSWER: " + json.dumps(answer), None)
            self.assertTrue(result.success)
            answer["day"] = True
            self.assertFalse(self.adapter.evaluate(item, "FINAL ANSWER: " + json.dumps(answer), None).success)
            async def private():
                await self.adapter.execute_native_tool(item, env, "claim_actor", {"actor": "Member1"}, "worker")
                return await self.adapter.execute_native_tool(item, env, "observe_private", {}, "worker")
            self.assertTrue(json.loads(asyncio.run(private())))
        finally:
            self.adapter.teardown_environment(env)

    def test_game_simultaneous_round_and_native_score(self):
        item = next(x for x in self.train if x["category"] == "game")
        env = self.env_for(item)
        try:
            rpc = env.data["game_rpc"]
            observation = rpc.call("observe", actor=rpc.actors[0])
            self.assertIn("view", observation)
            self.assertNotIn("grid", observation)
            for actor in rpc.actors:
                rpc.call("action", actor=actor, action={"move": "STAY"})
            status = rpc.call("score")
            self.assertEqual(status["round"], 1)
            context = {key: value for key, value in env.data.items() if key != "item"}
            self.assertEqual(self.adapter.evaluate(item, "FINAL ANSWER: self-reported 1", None,
                                                   **context).score, status["score"])
        finally:
            self.adapter.teardown_environment(env)

    def test_game_action_schema_exposes_native_interaction_parameters(self):
        item = next(x for x in self.train if x["category"] == "game")
        env = self.env_for(item)
        try:
            action = next(tool["function"] for tool in self.adapter.native_tool_schemas(item, env)
                          if tool["function"]["name"] == "game_action")
            properties = action["parameters"]["properties"]
            self.assertEqual(properties["interaction_qty"]["minimum"], 1)
            self.assertEqual(properties["interaction_type"]["enum"], ["wood", "sand", "stone"])
        finally:
            self.adapter.teardown_environment(env)


if __name__ == "__main__":
    unittest.main()
