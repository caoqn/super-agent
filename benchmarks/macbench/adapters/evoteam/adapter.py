"""Adapter for running Meta-Team pools on the native MAC-Bench catalogue."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from benchmarks.common.evoteam import require_upstream, REPO_ROOT

_BASE_DIR = require_upstream("macbench")

from benchmarks.adapter import BenchmarkAdapter, EnvContext, EvalResult
from core.execution_policy import BoundExecutionPolicy, ExecutionPolicy
from core.output_contract import OutputContract
from core.task_budget import TaskBudget, TaskBudgetExceeded


_DEFAULT_MAC_BENCH_ROOT = (
    REPO_ROOT / "external" / "MAC_Bench-main"
)






from benchmarks.macbench.adapters.evoteam.catalog import _catalog_paths, _load_catalog

class _TaskActionBudget(TaskBudget):
    """Shared dataset limits for calls, tokens, actions, and wall time."""

    def __init__(self, limits: dict[str, Any]):
        super().__init__(limits)
        self._on_exhausted = None

    def set_on_exhausted(self, callback) -> None:
        self._on_exhausted = callback

    def _stop(self, reason: str):
        self.reason = self.reason or reason
        if self._on_exhausted is not None:
            try:
                self._on_exhausted(self.reason)
            except Exception:
                pass
        raise TaskBudgetExceeded(self.reason)

    def admit(self) -> None:
        try:
            super().admit()
        except TaskBudgetExceeded:
            # MAC-Bench's native episode wrapper catches this exception type.
            from mac_bench.evaluation.episodes.budget import BudgetExceeded
            raise BudgetExceeded(self.reason)


def _set_runtime_pool_tools(team_dir: Path) -> None:
    """Expose only the adapter bridge on the disposable per-case pool copy."""
    configs = [path for path in team_dir.rglob("config.yaml") if path.parent != team_dir]
    if not configs:
        raise FileNotFoundError(f"No agent configs found in runtime team copy: {team_dir}")
    pool_config_path = team_dir / "pool.yaml"
    pool_config = yaml.safe_load(pool_config_path.read_text(encoding="utf-8")) or {}
    answer_agent = str((pool_config.get("settings") or {}).get("answer_agent") or "answer_agent")
    for config_path in configs:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid agent config: {config_path}")
        # The final-output service should only format the team's report. Keep
        # world-mutating task tools on the actual task-solving members.
        agent_name = str(payload.get("name") or config_path.parent.name)
        payload["tools"] = [] if agent_name == answer_agent else ["macbench"]
        config_path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


class MacBenchAdapter(BenchmarkAdapter):
    benchmark_name = "macbench"
    default_team = "pool_MACBENCH_OURS_INITIAL"
    default_timeout = 7200.0
    default_evolve_timeout = 7200.0
    default_effective_timeout = 7200.0
    default_split = "test"
    results_subdir = "macbench-results"
    split_choices = ["all", "dev", "test"]

    def output_contract(self) -> OutputContract:
        return OutputContract(
            name="macbench",
            submission_type="stateful_environment_completion",
            chairman_submission_hint=(
                "Submit a concise completion summary after verifying the requested "
                "postconditions in the MAC-Bench task environment."
            ),
            emergency_mode="tagged_output",
            emergency_instruction=(
                "Submit the best concise completion summary supported by the "
                "current MAC-Bench environment and evidence already collected. "
                "Do not call tools or claim unfinished changes are complete."
            ),
            emergency_prefix="FINAL OUTPUT:",
        )

    def default_source(self) -> Path:
        configured = os.environ.get("MAC_BENCH_ROOT", "").strip()
        return Path(configured).expanduser().resolve() if configured else _DEFAULT_MAC_BENCH_ROOT

    def add_extra_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--macbench-root",
            type=Path,
            default=self.default_source(),
            help="MAC-Bench repository root (or its data directory).",
        )
        parser.add_argument(
            "--category",
            choices=[
                "business_finance", "service_administration",
                "project_content_operations", "product_application",
                "platform_data", "developer_release",
            ],
            default=None,
            help="Optionally run only one MAC-Bench task category.",
        )
        parser.add_argument(
            "--max-items",
            type=int,
            default=0,
            help="Limit the number of loaded tasks (useful for smoke tests).",
        )

    def load_dataset(self, args: argparse.Namespace) -> list[dict]:
        root = Path(getattr(args, "macbench_root", None) or self.default_source())
        data_dir, rows, public_by_id = _load_catalog(root)
        project_root = (
            data_dir.parent
            if data_dir.name == "data" and (data_dir.parent / "src" / "mac_bench").is_dir()
            else Path(root).expanduser().resolve()
        )
        budgets_path = project_root / "configs" / "evaluation" / "task_budgets.json"
        if not budgets_path.is_file():
            raise FileNotFoundError(f"MAC-Bench task budgets are missing: {budgets_path}")
        budget_map = json.loads(budgets_path.read_text(encoding="utf-8"))
        split = str(getattr(args, "split", "test") or "test").lower()
        if split == "train":
            split = "dev"
        if split not in {"all", "dev", "test"}:
            raise ValueError("MAC-Bench split must be all, dev, or test")
        category = getattr(args, "category", None)

        selected = []
        for row in rows:
            if split != "all" and row.get("split") != split:
                continue
            if category and row.get("category") != category:
                continue
            task_id = str(row["task_id"])
            public = public_by_id[task_id]
            budget = budget_map.get(task_id)
            budget_source = "task_budgets.json"
            if budget is None and row.get("split") == "dev":
                # Native evaluation CLI defaults for dev tasks not in the test map.
                budget = dict(max_tokens=400000, max_model_calls=80,
                              max_tool_calls=160, max_env_actions=160, max_seconds=2400)
                budget_source = "native_evaluation_cli_defaults"
            required_budget_fields = (
                "max_tokens", "max_model_calls", "max_tool_calls",
                "max_env_actions", "max_seconds",
            )
            if not isinstance(budget, dict) or any(
                type(budget.get(field)) not in (int, float) or budget[field] <= 0
                for field in required_budget_fields
            ):
                raise ValueError(f"MAC-Bench task has an invalid or missing budget: {task_id}")
            selected.append({
                "task_id": task_id,
                "_macbench_split": str(row["split"]),
                "_macbench_category": str(row["category"]),
                "_macbench_revision": str(row["world_revision"]),
                "_macbench_bundle": str((data_dir / row["bundle"]).resolve()),
                "_macbench_data_dir": str(data_dir),
                "_macbench_project_root": str(project_root),
                "_macbench_instruction": public["instruction"],
                "_macbench_budget": {key: budget[key] for key in required_budget_fields},
                "_macbench_budget_source": budget_source,
            })

        max_items = int(getattr(args, "max_items", 0) or 0)
        return selected[:max_items] if max_items > 0 else selected

    def get_item_id(self, item: dict) -> str:
        return str(item.get("task_id") or "unknown")

    def execution_policy(self, item: dict) -> ExecutionPolicy:
        del item
        return ExecutionPolicy(
            name="macbench_native_task_api",
            version="1",
            environment_mode="macbench_task_api",
            allowed_tools=("macbench",),
            artifact_requirements=(
                "Complete the requested application or software changes in the MAC-Bench task world.",
                "Use the task's public tools and verify the resulting state before finalizing.",
            ),
            chairman_instructions=(
                "Inspect the task's native tool catalogue before assigning work. All team members share one MAC-Bench task environment.",
                "Use macbench(action='list_tools') to inspect tools, macbench(action='describe', tool_name=...) to read a tool schema, and macbench(action='call', tool_name=..., arguments={...}) to invoke it.",
                "Evaluate completion from the benchmark-visible application state; a final message alone does not complete the task.",
            ),
            special_rules=(
                "Only the public task instruction and task tools are exposed. Do not seek evaluator or private bundle data.",
                "All application changes are task-local and are graded by MAC-Bench's native evaluator.",
            ),
            infrastructure_failure_conditions=(
                "MAC-Bench task bundle is missing or has an invalid sealed revision",
                "task environment cannot be initialized or closed",
                "native MAC-Bench evaluator is unavailable",
            ),
        )

    def bind_execution_policy(
        self,
        policy: ExecutionPolicy,
        item: dict,
        env_ctx: EnvContext,
        session: Any,
    ) -> BoundExecutionPolicy:
        del item
        return policy.bind(
            workspace_paths={
                "workspace": str(session.workspace),
                "task_artifacts": str(env_ctx.data.get("artifact_dir", "")),
            },
        )

    def configure_native_runner(
        self, item: dict, env_ctx: EnvContext, runner: Any,
    ) -> None:
        budget = env_ctx.data.get("task_budget")
        if budget is None:
            return
        runner.max_seconds = min(runner.max_seconds, float(budget.limits["max_seconds"]))
        runner.wall_clock_seconds = min(
            runner.wall_clock_seconds, float(budget.limits["max_seconds"]),
        )
        runner.chairman_forced_submission_seconds = min(
            float(budget.submit_seconds), runner.max_seconds * 0.5,
        )
        loop = asyncio.get_running_loop()
        budget.set_on_exhausted(lambda reason: loop.call_soon_threadsafe(
            runner.request_forced_termination,
            f"macbench_budget_exceeded:{reason}",
        ))

    def setup_environment(self, item: dict, session: Any) -> EnvContext:
        source_root = Path(item["_macbench_project_root"]).expanduser().resolve()
        source_src = source_root / "src"
        if source_src.is_dir() and str(source_src) not in sys.path:
            sys.path.insert(0, str(source_src))
        try:
            from mac_bench.evaluation.tasks.catalog import open_task, task_record
        except ImportError as exc:
            raise RuntimeError(
                "MAC-Bench could not be imported. Install the MAC-Bench package "
                "or point --macbench-root at the repository containing src/mac_bench."
            ) from exc

        bundle = Path(item["_macbench_bundle"]).resolve()
        if not bundle.is_dir():
            raise FileNotFoundError(f"MAC-Bench task bundle is missing: {bundle}")
        record = task_record(
            bundle,
            category=item["_macbench_category"],
            split=item["_macbench_split"],
            task_id=item["task_id"],
            revision=item["_macbench_revision"],
        )
        case_dir = Path(session.dir)
        artifact_dir = case_dir / "macbench"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        handle = open_task(record, artifact_dir)
        token = None
        try:
            handle.open()
            _set_runtime_pool_tools(case_dir / "team")
            item["_macbench_world_instruction"] = handle.view.instruction
            item["_macbench_interaction_instructions"] = handle.view.interaction_instructions

            budget = dict(item["_macbench_budget"])
            action_budget = _TaskActionBudget(budget)
            (artifact_dir / "budget.json").write_text(
                json.dumps(action_budget.snapshot(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            event_lock = threading.Lock()
            event_path = artifact_dir / "events.jsonl"

            def emit_event(value: dict[str, Any]) -> None:
                with event_lock, event_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")

            handle.begin_episode(
                item["task_id"],
                admit_call=action_budget.admit,
                event=emit_event,
            )
            from benchmarks.macbench.adapters.evoteam.tool_bridge import set_active_task

            token = set_active_task(handle)
        except BaseException:
            if token is not None:
                from benchmarks.macbench.adapters.evoteam.tool_bridge import clear_active_task

                clear_active_task(token)
            handle.close()
            raise

        return EnvContext(data={
            "macbench_task": handle,
            "artifact_dir": str(artifact_dir),
            "action_budget": action_budget,
            "task_budget": action_budget,
            "task_id": item["task_id"],
            "task_category": item["_macbench_category"],
            "task_revision": item["_macbench_revision"],
            "context_token": token,
        })

    def build_task(self, item: dict, session: Any, workspace_files: list[str]) -> str:
        del session, workspace_files
        return (
            f"## MAC-Bench task ({item['_macbench_category']})\n\n"
            f"{item['_macbench_instruction']}\n\n"
            f"## Environment instructions\n\n{item.get('_macbench_world_instruction', '')}\n\n"
            f"{item.get('_macbench_interaction_instructions', '')}\n\n"
            "## Task tools\n\n"
            "Use the `macbench` tool to access the task's public environment. "
            "Call `macbench(action='list_tools')` first; inspect a tool with "
            "`macbench(action='describe', tool_name='...')`; invoke it with "
            "`macbench(action='call', tool_name='...', arguments={...})`. "
            "All agents share the same task state. Complete and verify the requested "
            "changes in that state, then submit a concise completion summary."
        )

    def evaluate(self, item: dict, predicted: str, session: Any, **ctx) -> EvalResult:
        del item, predicted, session
        handle = ctx.get("macbench_task")
        if handle is None:
            raise RuntimeError("MAC-Bench task was not initialized")
        task_budget = ctx.get("task_budget")
        if task_budget is not None:
            task_budget.finish()
        handle.end_episode()
        result = dict(handle.evaluate() or {})
        from benchmarks.macbench.adapters.evoteam.scoring import evaluation_fields

        fields = evaluation_fields(result)
        if fields["evaluation_status"] not in {"passed", "failed"} or fields["gc"] is None:
            raise RuntimeError(
                f"MAC-Bench evaluation unavailable: {fields['evaluation_status']}"
            )
        artifact_dir = Path(str(ctx.get("artifact_dir") or handle.directory))
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "evaluation.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        if task_budget is not None:
            (artifact_dir / "budget.json").write_text(
                json.dumps(task_budget.snapshot(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        handle.export_receipts(artifact_dir / "receipts")
        score = float(fields["gc"])
        success = fields["evaluation_status"] == "passed"
        return EvalResult(
            success=success,
            score=score,
            summary=f"MAC-Bench {fields['evaluation_status']} (GC={score:.3f})",
            details=json.dumps(result, ensure_ascii=False, default=str)[:4000],
            extra={
                "evaluation_status": fields["evaluation_status"],
                "goal_condition_score": fields["gc"],
                "task_category": ctx.get("task_category", ""),
                "task_revision": ctx.get("task_revision", ""),
                "tool_calls": getattr(ctx.get("action_budget"), "calls", None),
                "task_budget": (
                    ctx["task_budget"].snapshot()
                    if ctx.get("task_budget") is not None else None
                ),
            },
        )

    def teardown_environment(self, env_ctx: EnvContext) -> None:
        from benchmarks.macbench.adapters.evoteam.tool_bridge import clear_active_task

        handle = env_ctx.data.get("macbench_task")
        try:
            if handle is not None:
                handle.close()
        finally:
            clear_active_task(env_ctx.data.get("context_token"))

    def build_record(
        self,
        item: dict,
        idx: int,
        predicted: str,
        eval_result: EvalResult,
        session: Any,
        error: str,
    ) -> dict:
        return {
            "idx": idx,
            "task_id": item.get("task_id", ""),
            "split": item.get("_macbench_split", ""),
            "category": item.get("_macbench_category", ""),
            "world_revision": item.get("_macbench_revision", ""),
            "dataset_budget": item.get("_macbench_budget", {}),
            "dataset_budget_source": item.get("_macbench_budget_source", ""),
            "status": "error" if error else ("passed" if eval_result.success else "failed"),
            "success": bool(eval_result.success),
            "is_correct": bool(eval_result.success),
            "score": float(eval_result.score),
            "eval_summary": eval_result.summary,
            "eval_details": eval_result.details,
            "evaluation": eval_result.extra,
            "model_output": predicted,
            "run_error": error,
            "session_dir": str(session.dir),
        }

    def print_summary(self, records: list[dict]) -> None:
        scored = [row for row in records if not row.get("infrastructure_failure")]
        infra = [row for row in records if row.get("infrastructure_failure")]
        if not scored:
            print("No scored MAC-Bench results.")
            return
        passed = sum(bool(row.get("success")) for row in scored)
        print(f"\nMAC-Bench: {passed}/{len(scored)} passed ({100 * passed / len(scored):.1f}%)")
        if infra:
            print(f"Infrastructure failures (excluded): {len(infra)}")
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in scored:
            grouped[str(row.get("category") or "unknown")].append(float(row.get("score", 0.0)))
        for category, scores in sorted(grouped.items()):
            count = sum(score >= 1.0 for score in scores)
            print(f"  {category}: {count}/{len(scores)} passed, mean GC={sum(scores) / len(scores):.3f}")

    def dry_run_print(self, items: list[dict], args: argparse.Namespace) -> None:
        print(f"MAC-Bench ({args.split}): {len(items)} tasks")
        grouped: dict[str, int] = defaultdict(int)
        for item in items:
            grouped[item["_macbench_category"]] += 1
        for category, count in sorted(grouped.items()):
            print(f"  {category}: {count}")
        for index, item in enumerate(items[:10]):
            print(f"  [{index}] {item['task_id']} ({item['_macbench_category']})")
        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more")


if __name__ == "__main__":
    MacBenchAdapter().cli()
