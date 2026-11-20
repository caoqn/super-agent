"""MAC-Bench adapter for Skill-MAS's native tool runtime."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from .base import NativeBenchmarkAdapter, NativeEvalResult, NativeTask


class MacBenchAdapter(NativeBenchmarkAdapter):
    name = "macbench"
    aliases = ("mac_bench", "mac-bench")

    def default_source(self) -> Path:
        configured = os.environ.get("MAC_BENCH_ROOT", "").strip()
        return Path(configured).expanduser().resolve() if configured else (
            Path(__file__).resolve().parents[1] / "native_benchmarks" / "MAC_Bench-main"
        )

    def load_tasks(self, source: Path | None = None, *, split: str = "test", max_tasks: int = 0) -> list[NativeTask]:
        split = "dev" if split == "train" else split
        if split not in {"dev", "test"}:
            raise ValueError("MAC-Bench split must be dev or test (train aliases dev).")
        source_path = Path(source or self.default_source()).expanduser().resolve()
        tasks_path = source_path if source_path.is_file() else source_path / "tasks.json"
        if not tasks_path.is_file():
            tasks_path = source_path / "data" / "tasks.json"
        data_dir = tasks_path.parent
        root = data_dir.parent
        public_path = data_dir / "public_tasks.json"
        rows = json.loads(tasks_path.read_text(encoding="utf-8"))
        if rows.get("format") != "collaboration.tasks.v1":
            raise ValueError(f"Unsupported MAC-Bench task catalogue: {tasks_path}")
        public_rows = json.loads(public_path.read_text(encoding="utf-8"))
        public = {str(row.get("task_id")): row for row in public_rows.get("tasks", [])}
        selected: list[NativeTask] = []
        for row in rows.get("tasks", []):
            if split and str(row.get("split")) != split:
                continue
            task_id = str(row["task_id"])
            public_row = public[task_id]
            if public_row.get("world_revision") != row.get("world_revision"):
                raise ValueError(f"Public instruction revision mismatch: {task_id}")
            bundle = (data_dir / row["bundle"]).resolve()
            if data_dir not in bundle.parents:
                raise ValueError(f"Task bundle escapes catalogue directory: {task_id}")
            selected.append(NativeTask(
                task_id=task_id,
                prompt=str(public_row.get("instruction") or row.get("instruction") or ""),
                raw={**row, "source_root": str(root), "bundle_path": str(bundle)},
                split=str(row.get("split") or split),
            ))
        return selected[: int(max_tasks)] if max_tasks and max_tasks > 0 else selected

    def setup_runtime(self, task: NativeTask, context: dict[str, Any]) -> None:
        try:
            from mac_bench.evaluation.tasks.catalog import open_task, task_record
        except ImportError as exc:
            raise RuntimeError("MAC-Bench is not installed; install the Docker image or pip package") from exc
        root = Path(str(task.raw.get("source_root") or self.default_source()))
        bundle = Path(task.raw.get("bundle_path") or root / "data" / str(task.raw["bundle"]))
        if not bundle.is_dir():
            raise FileNotFoundError(f"MAC-Bench task bundle is missing: {bundle}. Run `mac-bench data download --split {task.split}`.")
        record = task_record(bundle, category=task.raw.get("category"), split=task.split,
                             task_id=task.task_id, revision=task.raw.get("world_revision"))
        task_dir = Path(str(context.get("artifact_dir") or context["workspace"])) / "macbench"
        task_dir.mkdir(parents=True, exist_ok=True)
        handle = open_task(record, task_dir)
        try:
            handle.open()
            event_lock = threading.Lock()

            def event(value):
                with event_lock, (task_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")

            handle.begin_episode(task.task_id, admit_call=lambda: None, event=event)
        except BaseException:
            handle.close()
            raise
        context["macbench_task"] = handle

    def build_prompt(self, task: NativeTask, context: dict[str, Any]) -> str:
        handle = context.get("macbench_task")
        if handle is None:
            return task.prompt
        view = handle.view
        return (
            f"{view.instruction}\n\n{view.interaction_instructions}\n\n"
            "All workflow roles share this task environment. Use only the supplied task tools. "
            "Coordinate changes with upstream roles and verify the final application or source state."
        )

    def tool_schemas(self, task: NativeTask, context: dict[str, Any] | None = None) -> list[dict[str, Any]] | None:
        del task
        handle = (context or {}).get("macbench_task")
        return [tool.as_function() for tool in handle.view.tools] if handle is not None else []

    async def execute_tool(self, name: str, args: dict[str, Any], context: dict[str, Any]) -> str | None:
        handle = context.get("macbench_task")
        if handle is None:
            return None
        result = handle.environment.call(name, args)
        return json.dumps(result, ensure_ascii=False, default=str)

    def evaluate(self, task: NativeTask, output: str, context: dict[str, Any]) -> NativeEvalResult:
        del task, output
        handle = context.get("macbench_task")
        if handle is None:
            return NativeEvalResult(False, 0.0, "MAC-Bench task was not initialized")
        handle.end_episode()
        result = dict(handle.evaluate() or {})
        from mac_bench.evaluation.reports.metrics import evaluation_fields
        fields = evaluation_fields(result)
        if fields["evaluation_status"] not in {"passed", "failed"} or fields["gc"] is None:
            raise RuntimeError(f"MAC-Bench evaluation unavailable: {fields['evaluation_status']}")
        (Path(handle.directory) / "evaluation.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        status = str(result.get("status") or "").lower()
        score = float(fields["gc"])
        handle.export_receipts(Path(handle.directory) / "receipts")
        return NativeEvalResult(status == "passed", score, f"MAC-Bench status={status or 'unknown'}", result)

    def teardown_runtime(self, task: NativeTask, context: dict[str, Any]) -> None:
        del task
        handle = context.pop("macbench_task", None)
        if handle is not None:
            handle.close()
