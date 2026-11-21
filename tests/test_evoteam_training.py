"""Offline initial-pool lifecycle: real selection, runner, reflection and persistence.

Only model responses and the benchmark environment/scoring are synthetic.
These tests check reproducible wiring, not benchmark performance.
"""
import asyncio
import hashlib
import json
from pathlib import Path
import pytest

from methods.Evoteam.bootstrap import configure_source
configure_source()
from benchmarks.adapter import BenchmarkAdapter, EvalResult
from core import llm, run_manager
from litellm import ModelResponse

ROOT = Path(__file__).resolve().parents[1]


def snapshot(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


class LifecycleAdapter(BenchmarkAdapter):
    benchmark_name = "offline_lifecycle"
    results_subdir = "offline_lifecycle"

    def output_contract(self):
        from core.output_contract import OutputContract
        return OutputContract()

    def load_dataset(self, args):
        return [{"id": "case-0"}, {"id": "case-1"}]

    def get_item_id(self, item):
        return item["id"]

    def build_task(self, item, session, workspace_files):
        return "Coordinate a check and return verified."

    def evaluate(self, item, predicted, session, **ctx):
        return EvalResult(success=predicted == "verified", score=float(predicted == "verified"))

    def build_record(self, item, idx, predicted, eval_result, session, error):
        return {"task_id": item["id"], "score": eval_result.score, "error": error}

    def print_summary(self, records):
        pass


@pytest.mark.parametrize("benchmark", ["gaia", "macbench", "masbench"])
def test_initial_pool_evolves_and_next_case_reuses_version(tmp_path, monkeypatch, benchmark):
    # Import each real adapter in the same process to check shared/external namespace wiring.
    import importlib
    module = importlib.import_module(f"benchmarks.{benchmark}.adapters.evoteam.adapter")
    assert module.BenchmarkAdapter is BenchmarkAdapter
    from main import load_pool
    from core.runner import Runner
    source = ROOT / "benchmarks" / benchmark / "adapters" / "evoteam" / "pools/initial"
    before = snapshot(source)
    pool = load_pool(source)
    assert pool.chairman_name in pool.agents
    assert pool.registry.get("bash") is not None
    members = [n for n in pool.agents if n not in {pool.chairman_name, "answer_agent"}]
    member = members[0]
    calls = []
    started = 0
    submitted_answer = False

    async def completion(**kwargs):
        nonlocal started, submitted_answer
        await asyncio.sleep(0)
        names = {t["function"]["name"] for t in kwargs.get("tools") or []}
        calls.append(names)
        if "select_template_family" in names:
            catalog = json.loads(kwargs["messages"][1]["content"])["template_catalog"]
            if catalog:
                payload = {"family_action": "reuse", "family_id": catalog[0]["family_id"]}
            else:
                payload = {"family_action": "cold_start", "family_id": "evidence_check",
                           "family_description": "Complementary evidence checking and independent verification."}
            tool = "select_template_family"
        elif "compose_cold_start_team" in names:
            tool = "compose_cold_start_team"
            payload = {"provisional_agent_ids": [member], "reason": "Complementary evidence check."}
        elif "record_team_template_decision" in names:
            context = json.loads(kwargs["messages"][1]["content"])
            tool = "record_team_template_decision"
            if context["current_template"]:
                payload = {"action": "retain", "reason": "Repeated check supports this team.",
                           "target_template_id": context["current_template"]["template_id"]}
            else:
                payload = {"action": "create", "reason": "A completed check supports this team.",
                           "agent_ids": [member],
                           "family_description": "Complementary evidence checking and independent verification for tasks with explicit acceptance criteria."}
        elif "skip_l2_reflection" in names:
            tool, payload = "skip_l2_reflection", {"reason": "No new peer evidence."}
        elif "skip_l1_reflection" in names:
            tool, payload = "skip_l1_reflection", {"reason": "No reusable individual change."}
        elif "finalize_task" in names:
            # Each case starts one real worker, then submits via the normal training primitive.
            prior = [m for m in kwargs["messages"] if m.get("role") == "tool"]
            if not prior:
                started += 1
                submitted_answer = False
                tool, payload = "start_agent", {"name": member}
            elif benchmark != "macbench" and not any(
                c.get("function", {}).get("name") == "send_message"
                for m in kwargs["messages"] for c in m.get("tool_calls", [])
            ):
                tool, payload = "send_message", {"to": ["answer_agent"], "content": "Verify and return verified."}
            elif benchmark != "macbench" and not submitted_answer:
                tool, payload = "wait_for_replies", {"from_agents": ["answer_agent"], "reason": "Await verified answer."}
            else:
                tool, payload = "finalize_task", {"output": "verified"}
        elif "send_message" in names:
            tool, payload = "send_message", {"to": [pool.chairman_name], "content": "FINAL ANSWER: verified"}
            submitted_answer = True
        else:
            raise AssertionError(f"Unexpected model invocation: {sorted(names)}")
        return ModelResponse(model="openai/gpt-4o-mini", choices=[{"index": 0, "message": {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": f"call-{len(calls)}", "type": "function",
                            "function": {"name": tool, "arguments": json.dumps(payload)}}] }},
        ], usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30})

    monkeypatch.setattr(llm, "complete", completion)
    monkeypatch.setattr(run_manager, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(Runner, "_CHAIRMAN_EXIT_GRACE_SECONDS", 0)
    manager = run_manager.RunManager.create_run("offline", source, benchmark)
    adapter = LifecycleAdapter()
    native = getattr(module, {"gaia": "GAIAAdapter", "macbench": "MacBenchAdapter", "masbench": "MASBenchAdapter"}[benchmark])()
    adapter.output_contract = native.output_contract
    adapter.benchmark_name = benchmark
    args = adapter.build_parser().parse_args(["--evolve"])
    records = []
    async def run():
        for i in range(2):
            records.append(await adapter.run_single_case(
                {"id": f"case-{i}"}, i, args, manager, benchmark, True, 300, 300,
            ))
    asyncio.run(asyncio.wait_for(run(), timeout=30))
    assert started == 2
    assert all(not r["error"] and r["score"] == 1 for r in records), records
    assert records[0]["team_version_before"] == "v000"
    assert records[0]["team_version_after"] == "v001", records
    assert records[1]["team_version_before"] == "v001"
    assert records[0]["team_reflection"]["action"] == "create"
    assert records[1]["team_reflection"]["action"] == "retain"
    assert (manager.team_dir / "v001/templates.json").is_file()
    assert snapshot(source) == before
