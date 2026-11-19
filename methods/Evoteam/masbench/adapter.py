"""Meta-Team adapter for the fixed MASBench train/test split."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import string
import sys
import threading
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1] / "runtime"
sys.path.insert(0, str(BASE_DIR))

from benchmarks.adapter import BenchmarkAdapter, EnvContext, EvalResult
from core.execution_policy import ExecutionPolicy
from core.output_contract import OutputContract

DEFAULT_SPLIT_DIR = Path(os.environ.get(
    "MASBENCH_SPLIT_DIR",
    str(BASE_DIR.parents[2] / "data" / "masbench_split18_test90"),
))
DEFAULT_SOURCE_CANDIDATES = (BASE_DIR.parents[2] / "external" / "MASBench-main",)
GAME_WORKER = BASE_DIR.parents[2] / "benchmarks" / "masbench" / "game" / "worker.py"




from benchmarks.masbench.reasoning.scoring import _squad_f1

class GameRPC:
    """Isolated worker around upstream MASBench game physics."""

    def __init__(self, source_root: Path, map_path: Path, output_dir: Path):
        import subprocess

        output_dir.mkdir(parents=True, exist_ok=True)
        skill_root_value = os.environ.get("MASBENCH_SUPPORT_ROOT")
        skill_root = Path(skill_root_value).expanduser() if skill_root_value else None
        dependency_dir = skill_root / ".masbench_deps" if skill_root else None
        if dependency_dir is None or not dependency_dir.exists():
            candidates = [BASE_DIR.parents[2] / "external" / ".masbench_deps"]
            dependency_dir = next((p for p in candidates if p.exists()), None)
        if dependency_dir is None:
            raise FileNotFoundError(
                "MASBench Game dependencies not found; set MASBENCH_SUPPORT_ROOT "
                "and install the packages in requirements/masbench-game.txt into that directory"
            )
        self._lock = threading.Lock()
        self._log = (output_dir / "worker.log").open("a", encoding="utf-8")
        self._process = subprocess.Popen(
            [sys.executable, str(GAME_WORKER), str(source_root), str(map_path),
             str(output_dir), str(dependency_dir)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
            text=True, bufsize=1,
        )
        try:
            self.actors = self._read().get("actors", [])
        except BaseException:
            self.close()
            raise

    def _read(self) -> dict:
        import select
        if not select.select([self._process.stdout], [], [], 90)[0]:
            raise TimeoutError("MASBench Game worker timed out")
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError("MASBench Game worker exited; inspect worker.log")
        response = json.loads(line)
        if not response.get("ok"):
            raise ValueError(response.get("error", "Game worker error"))
        return response

    def call(self, operation: str, **kwargs):
        with self._lock:
            self._process.stdin.write(json.dumps({"op": operation, **kwargs}) + "\n")
            self._process.stdin.flush()
            return self._read().get("result")

    def close(self) -> None:
        import subprocess
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        for stream in (self._process.stdin, self._process.stdout):
            if stream:
                stream.close()
        self._log.close()


class MASBenchAdapter(BenchmarkAdapter):
    benchmark_name = "masbench"
    default_team = "pool_MASBENCH_DYNAMIC_6"
    default_timeout = 1800.0
    default_evolve_timeout = 3600.0
    default_effective_timeout = 1800.0
    default_split = "test"
    split_choices = ["train", "test"]
    results_subdir = "masbench-results"
    enforce_native_tool_scope = True

    def output_contract(self) -> OutputContract:
        return OutputContract(
            name="masbench",
            answer_agent_enabled=True,
            enforce_answer_agent=True,
            answer_agent_prefix="FINAL ANSWER:",
            answer_agent_instruction=(
                "Return the complete benchmark answer after `FINAL ANSWER:`. "
                "For Scheduling, output exactly one JSON object with day, "
                "start_minute, and end_minute integer fields. For other tasks, "
                "return only the concise answer or completion status."
            ),
            chairman_instruction=(
                "Send the task, verified evidence, candidate response, and the "
                "required MASBench answer format to the AnswerAgent."
            ),
            submission_type="native_short_answer_or_json",
            chairman_submission_hint=(
                "Submit the complete answer through the AnswerAgent using the "
                "required FINAL ANSWER: prefix."
            ),
            emergency_mode="tagged_output",
            emergency_instruction=(
                "Return the best complete MASBench answer now, preserving the "
                "required JSON format for Scheduling."
            ),
            emergency_prefix="FINAL ANSWER:",
        )

    def __init__(self, data_dir: str | Path | None = None,
                 masbench_root: str | Path | None = None):
        self.data_dir = Path(data_dir or DEFAULT_SPLIT_DIR).expanduser()
        self.masbench_root = Path(masbench_root).expanduser() if masbench_root else None

    def add_extra_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--data-dir", default=str(self.data_dir),
                            help="Directory containing train/test manifests")
        parser.add_argument("--masbench-root", default=None,
                            help="Path to the original MASBench-main source tree")
        parser.add_argument("--max-items", type=int, default=None,
                            help="Limit loaded cases (useful for smoke runs)")

    def _source_root(self, override: str | Path | None = None) -> Path:
        candidates = ([Path(override).expanduser()] if override else [])
        if self.masbench_root:
            candidates.append(self.masbench_root)
        if os.environ.get("MASBENCH_ROOT"):
            candidates.append(Path(os.environ["MASBENCH_ROOT"]).expanduser())
        candidates.extend(DEFAULT_SOURCE_CANDIDATES)
        for path in candidates:
            if (path / "Reasoning" / "reasoning.json").is_file():
                return path.resolve()
        raise FileNotFoundError(
            "MASBench source tree not found. Pass --masbench-root or set MASBENCH_ROOT."
        )

    def load_dataset(self, args: argparse.Namespace) -> list[dict]:
        from benchmarks.masbench.data import load_dataset
        return load_dataset(args, Path(getattr(args, "data_dir", self.data_dir)).expanduser(),
                            self._source_root(getattr(args, "masbench_root", None)))

    def get_item_id(self, item: dict) -> str:
        return f"{item['category']}__{item['id']}"

    def execution_policy(self, item: dict) -> ExecutionPolicy:
        return ExecutionPolicy(
            name="masbench_native_environment", version="1",
            environment_mode="private_native_tools",
            allowed_tools=(),
            chairman_instructions=(
                "Coordinate the selected Meta-Team members. Assign native actor identities through claim_actor before using private tools.",
                "Private evidence, calendars, and local game observations are available only through role-bound tools; never inspect dataset or answer files.",
            ),
            special_rules=(
                "Only explicit messages share information between Meta-Team agents.",
                "The Chairman claims one native actor, starts enough workers for the remaining actors, and calls assign_actor(agent, actor) for each worker. This sends only the public task and identity assignment to that worker.",
                "Each execution agent owns at most one actor. Use observe_private for that actor's evidence/calendar/local view; no agent may read another actor's private state.",
                "Return the requested answer with the exact FINAL ANSWER: prefix.",
            ),
            infrastructure_failure_conditions=("MASBench source data is missing",),
        )

    def setup_environment(self, item: dict, session: Any) -> EnvContext:
        source_root = Path(item["source_root"])
        category = item["category"]
        data: dict[str, Any] = {"bindings": {}, "actor_owners": {},
                                "inboxes": {}, "messages": [], "lock": threading.RLock()}
        if category == "reasoning":
            row = item["row"]
            private = row.get("context", {})
            data["private"] = {
                "Alice": self._format_context(private.get("agent_a", [])),
                "Bob": self._format_context(private.get("agent_b", [])),
            }
        elif category == "scheduling":
            path = Path(item["path"])
            data["private"] = {
                f"Member{i}": json.loads((path / f"Member{i}_final_schedule.json").read_text())
                for i in range(1, 6)
            }
        else:
            output_dir = Path(session.workspace) / "masbench_game"
            rpc = GameRPC(source_root, Path(item["path"]), output_dir)
            data["game_rpc"] = rpc
            data["private"] = {actor: None for actor in rpc.actors}
        data["actors"] = list(data["private"])
        data["inboxes"] = {actor: [] for actor in data["actors"]}
        return EnvContext(data=data)

    @staticmethod
    def _format_context(articles: list) -> str:
        lines = []
        for article in articles:
            if not isinstance(article, list) or len(article) < 2:
                continue
            lines.append(f"Article: {article[0]}")
            lines.extend(f"Sentence: {text}" for text in article[1] if str(text).strip())
        return "\n".join(lines)

    def build_task(self, item: dict, session: Any, workspace_files: list[str]) -> str:
        category = item["category"]
        visible_actors = {"reasoning": ["Alice", "Bob"],
                          "scheduling": [f"Member{i}" for i in range(1, 6)]}.get(category)
        if category == "game":
            visible_actors = json.loads(Path(item["path"]).read_text())["map"]["agents"]
            visible_actors = [a.get("name", f"Agent{i + 1}")
                              for i, a in enumerate(visible_actors)]
        item["_actors"] = visible_actors
        task = item.get("row", {}).get("question", "") if category == "reasoning" else {
            "scheduling": "Find a shared continuous 20-minute meeting for all five members across five days.",
            "game": "Cooperate to collect and deliver the requested resources in the partially observed grid world within 50 simultaneous rounds.",
        }[category]
        answer_format = (
            "a concise answer" if category == "reasoning" else
            '{"day": 1, "start_minute": 600, "end_minute": 620}' if category == "scheduling" else
            "a concise completion status"
        )
        game_instructions = (
            " In Game, each actor must observe and submit exactly one action per round; "
            "the world advances only after all actors submit. Repeat until the game reports done or 50 rounds. "
            "Messages between execution agents obey the native communication range. "
            "Only the Chairman exchanges final formatting messages with the AnswerAgent."
            if category == "game" else ""
        )
        return (
            f"{task}\n\nMASBench actor identities: {json.dumps(visible_actors)}. "
            "A Meta-Team agent may claim at most one actor, and each actor has one owner. "
            "The Chairman must claim one actor and start enough workers for every other actor. "
            "After start_agent, use assign_actor(agent, actor) to bind the worker and send it the public task. "
            "Call claim_actor(actor) before observe_private or game_action. A coordinator without an actor "
            "cannot read private data or act. Use send_message/read_messages for explicit information sharing. "
            "Do not read benchmark files or expose reference answers. "
            f"Return FINAL ANSWER: followed by {answer_format}.{game_instructions}"
        )

    def build_team_selection_task(self, item: dict, full_task: str) -> str:
        actor_count = len(item.get("_actors", []))
        # The Chairman can own one native actor, so the minimum recruitable
        # worker count is one below the number of native actors.
        workers = max(0, actor_count - 1)
        return (
            f"{full_task}\n\nTeam sizing constraint: this case has {actor_count} "
            f"native actor identities. Recruit at least {workers} workers plus "
            "the Chairman so each actor can have a distinct Meta-Team owner. "
            "Do not expose private data during team selection."
        )

    def runtime_team_capacity_members(
        self, item: dict, pool: Any, selection: Any,
    ) -> list[str]:
        """Add only the runtime capacity needed to cover the dataset's actors."""
        required_workers = max(0, len(item.get("_actors", [])) - 1)
        candidates = [name for name in pool.agents
                      if name not in {pool.chairman_name, "answer_agent"}]
        if len(candidates) < required_workers:
            raise ValueError(
                f"MASBench needs {required_workers} workers plus Chairman; "
                f"this pool has only {len(candidates)} recruitable workers"
            )
        selected = set(selection.allowed_agent_ids) if selection is not None else set(candidates)
        missing = max(0, required_workers - len(selected))
        return [name for name in candidates if name not in selected][:missing]

    def configure_native_runner(self, item: dict, env_ctx: EnvContext, runner: Any) -> None:
        env_ctx.data["runner"] = runner

    def build_test_task(self, item: dict, session: Any, task: str) -> str:
        return task + "\nTreat all private environment data as role-scoped; only share it through messages."

    def evaluate(self, item: dict, predicted: str, session: Any, **ctx) -> EvalResult:
        answer = re.split(r"(?i)final\s*answer\s*:", predicted or "", maxsplit=1)
        answer = answer[1].strip() if len(answer) > 1 else (predicted or "").strip()
        category = item["category"]
        details: dict[str, Any] = {"category": category, "prediction": answer}
        if category == "reasoning":
            reference = str(item["row"].get("answer", ""))
            score = _squad_f1(reference, answer)
        elif category == "scheduling":
            from benchmarks.masbench.scheduling.scoring import score_schedule
            score, scheduling_details = score_schedule(answer, item["path"])
            details.update(scheduling_details)
        else:
            rpc = ctx.get("game_rpc")
            status = rpc.call("score") if rpc else {"score": 0.0}
            score = float(status["score"])
            details["environment"] = status
        return EvalResult(score >= 1.0, float(score), f"MASBench {category}", details=details)

    def build_record(self, item: dict, idx: int, predicted: str,
                     eval_result: EvalResult, session: Any, error: str) -> dict:
        return {"idx": idx, "task_id": self.get_item_id(item),
                "category": item["category"], "prediction": predicted,
                "success": eval_result.success, "score": eval_result.score,
                "evaluation": eval_result.summary, "details": eval_result.details,
                "run_error": error}

    def print_summary(self, records: list[dict]) -> None:
        scores = [float(r.get("score", 0)) for r in records]
        print(f"MASBench: {len(records)} cases, avg_score={sum(scores)/len(scores) if scores else 0:.4f}")
        for category in ("reasoning", "scheduling", "game"):
            values = [float(r.get("score", 0)) for r in records if r.get("category") == category]
            if values:
                print(f"  {category}: {sum(values)/len(values):.4f} ({len(values)} cases)")

    def native_tool_schemas(self, item: dict, env_ctx: EnvContext) -> list[dict]:
        actors = env_ctx.data["actors"]
        def schema(name, description, properties=None, required=None):
            return {"type": "function", "function": {
                "name": name, "description": description,
                "parameters": {"type": "object", "properties": properties or {},
                                "required": required or [], "additionalProperties": False},
            }}
        schemas = [
            schema("assign_actor", "Chairman only: bind a started worker to an actor and send it the public task. No private evidence is sent.",
                   {"agent": {"type": "string"}, "actor": {"type": "string", "enum": actors}},
                   ["agent", "actor"]),
            schema("claim_actor", "Claim one MASBench actor identity for this Meta-Team agent.",
                   {"actor": {"type": "string", "enum": actors}}, ["actor"]),
            schema("observe_private", "Read only the private evidence, calendar, or local game observation for your claimed actor."),
            schema("read_native_messages", "Read explicit messages delivered to your claimed actor."),
            schema("send_native_message", "Send an explicit message to another actor. Game messages must be in range.",
                   {"recipient": {"type": "string", "enum": actors}, "message": {"type": "string"}},
                   ["recipient", "message"]),
        ]
        if item["category"] == "game":
            schemas.append(schema("game_action", "Submit your actor's single action for this simultaneous round.",
                                  {
                                      "move": {"type": "string", "enum": ["UP", "DOWN", "LEFT", "RIGHT", "STAY", "PICK_UP", "PUT_DOWN"]},
                                      "interaction_qty": {"type": "integer", "minimum": 1},
                                      "interaction_type": {"type": "string", "enum": ["wood", "sand", "stone"]},
                                  }, ["move"]))
        return schemas

    def _caller_actor(self, env: dict, caller: str) -> str:
        actor = env["bindings"].get(caller)
        if actor is None:
            raise PermissionError("claim an actor identity before using private tools")
        return actor

    async def execute_native_tool(self, item: dict, env_ctx: EnvContext,
                                  name: str, args: dict, caller: str) -> str:
        env = env_ctx.data
        with env["lock"]:
            if name == "assign_actor":
                runner = env.get("runner")
                if runner is None or caller != runner.chairman_name:
                    raise PermissionError("only the Chairman can assign native actors")
                worker = str(args.get("agent", ""))
                if (worker not in runner.agents or worker == runner.chairman_name
                        or worker in runner._global_service_agent_ids):
                    raise ValueError("assign_actor requires a started execution worker")
                # Reuse the same one-to-one binding validation as self-claims.
                result = await self.execute_native_tool(
                    item, env_ctx, "claim_actor", {"actor": args.get("actor")}, worker,
                )
                public_task = self.build_task(item, None, [])
                runner.message_store.send(
                    sender=caller, receiver=worker,
                    content=f"Your assigned native actor is {args['actor']}.\n\nPublic task:\n{public_task}",
                )
                return result + " Public task delivered to worker."
            if name == "claim_actor":
                actor = str(args.get("actor", ""))
                if actor not in env["private"]:
                    raise ValueError("unknown actor")
                if caller in env["bindings"] and env["bindings"][caller] != actor:
                    raise PermissionError("a Meta-Team agent may claim only one actor")
                owner = env["actor_owners"].get(actor)
                if owner and owner != caller:
                    raise PermissionError("actor is already claimed")
                env["bindings"][caller] = actor
                env["actor_owners"][actor] = caller
                return f"Claimed actor {actor}."
            actor = self._caller_actor(env, caller)
            rpc = env.get("game_rpc")
            if name == "observe_private":
                value = await asyncio.to_thread(rpc.call, "observe", actor=actor) if rpc else env["private"][actor]
            elif name == "read_native_messages":
                value = list(env["inboxes"][actor])
                env["inboxes"][actor].clear()
            elif name == "send_native_message":
                recipient = str(args.get("recipient", ""))
                if recipient not in env["inboxes"]:
                    raise ValueError("unknown recipient")
                if rpc and not await asyncio.to_thread(rpc.call, "in_range", actor=actor, recipient=recipient):
                    raise PermissionError("recipient is outside native communication range")
                message = {"sender": actor, "recipient": recipient, "message": str(args.get("message", ""))}
                env["inboxes"][recipient].append(message)
                env["messages"].append(message)
                value = {"delivered": True}
            elif name == "game_action" and rpc:
                quantity = args.get("interaction_qty")
                if quantity is not None and (isinstance(quantity, bool) or int(quantity) < 1):
                    raise ValueError("interaction_qty must be a positive integer")
                interaction_type = args.get("interaction_type")
                if interaction_type is not None and args.get("move") != "PICK_UP":
                    raise ValueError("interaction_type is only valid with PICK_UP")
                value = await asyncio.to_thread(rpc.call, "action", actor=actor,
                                                action={key: args[key] for key in
                                                        ("move", "interaction_qty", "interaction_type")
                                                        if key in args})
            else:
                raise ValueError(f"native tool unavailable: {name}")
        return json.dumps(value, ensure_ascii=False, default=str)

    def validate_native_message(self, item: dict, env_ctx: EnvContext,
                                sender: str, receiver: str) -> None:
        env = env_ctx.data
        source, target = env["bindings"].get(sender), env["bindings"].get(receiver)
        rpc = env.get("game_rpc")
        if rpc:
            runner = env.get("runner")
            if runner and "answer_agent" in {sender, receiver}:
                if {sender, receiver} == {runner.chairman_name, "answer_agent"}:
                    return
                raise PermissionError("only the Chairman may use the final answer service")
            if not source or not target:
                raise PermissionError("Game messages require actor bindings; use assign_actor to initialize a started worker")
            if not rpc.call("in_range", actor=source, recipient=target):
                raise PermissionError(f"{target} is outside {source}'s communication range")

    def filter_reflection_handoff_trace(
        self, item: dict, trace: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        # Preserve participants and handoff IDs for workflow learning, but do
        # not replay task-specific/private message text into reusable memory.
        return [
            {key: value for key, value in row.items() if key != "content"}
            for row in trace
        ]

    def result_for_evolution_reflection(self, item: dict, result: Any) -> Any:
        # Predicted answers can coincide with the held-out reference. Keep
        # scores and participation evidence while withholding the final text.
        return replace(result, output="[MASBench final response withheld from reflection]")

    def eval_result_for_evolution_reflection(
        self, item: dict, eval_result: EvalResult,
    ) -> EvalResult:
        return replace(eval_result, details={"category": item["category"]})

    def teardown_environment(self, env_ctx: EnvContext) -> None:
        rpc = env_ctx.data.pop("game_rpc", None)
        if rpc:
            rpc.close()


if __name__ == "__main__":
    MASBenchAdapter().cli()
