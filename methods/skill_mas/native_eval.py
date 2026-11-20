"""Native benchmark rollout for GAIA, MAC-Bench and MASBench.

This module keeps native benchmark state in disposable per-task workspaces and
uses Skill-MAS's existing three-stage build unchanged. It deliberately does
not import Meta-Team's Runner or any shared evolution registry.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
import uuid
from functools import partial
from dataclasses import asdict
from pathlib import Path
from typing import Any

from Skill_MAS.bench_adapters import get_adapter
from Skill_MAS.bench_adapters.base import make_workspace
from Skill_MAS.bench_adapters.native_runtime import NativeToolRuntime
from Skill_MAS.skill_mas.build import make_planner_call_fn, make_text_call_fn, run_mas_pipeline_with_retries
from Skill_MAS.skill_mas.openai_async_client import AsyncOpenAIClient, normalize_usage_tokens
from Skill_MAS.utils.llm_cost import empty_usage_totals, add_usage_totals, pricing_reference


class InfrastructureFailure(RuntimeError):
    """Fatal benchmark/runtime failure; abort the evaluation run."""


class APICircuitOpen(InfrastructureFailure):
    """Abort after a sustained upstream API outage."""


_NATIVE_TASK_TIMEOUT_S = {
    "gaia": 900.0,
    "macbench": 7200.0,
    "masbench": float(os.environ.get("MASBENCH_TASK_TIMEOUT_S", "7200")),
}
_TIMEOUT_WAIT_WINDOW_S = 800.0

_API_ERROR_STREAK_LIMIT = 3
_API_ERROR_MARKERS = (
    "api error", "apiconnectionerror", "service unavailable", "rate limit",
    "ratelimit", "http 429", "http 500", "http 502", "http 503",
    "status code: 429", "status code: 500", "status code: 502", "status code: 503",
)


def _is_api_failure(trace: dict[str, Any]) -> bool:
    """Classify upstream/provider failures without treating task outcomes as API outages."""
    if trace.get("failure_stage") == "task_timeout":
        return False
    text = " ".join(str(trace.get(k) or "") for k in ("failure_reason", "failure_stage")).lower()
    return any(marker in text for marker in _API_ERROR_MARKERS)


def _task_timeout(bench_backend: str) -> float:
    key = str(bench_backend or "").strip().lower()
    for prefix, seconds in _NATIVE_TASK_TIMEOUT_S.items():
        if key.startswith(prefix):
            return seconds
    return 1800.0


def _task_wall_timeout(bench_backend: str) -> float:
    """Wall-clock deadline: effective task budget plus excluded wait window."""
    return _task_timeout(bench_backend) + _TIMEOUT_WAIT_WINDOW_S


def _merge_usage(dst: dict[str, Any], src: dict[str, Any]) -> None:
    prompt, output, total = normalize_usage_tokens(src)
    add_usage_totals(dst, {**src, "prompt_tokens": prompt, "output_tokens": output, "total_tokens": total})


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Persist one complete trace so interruption cannot leave an empty checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _checkpoint_score(trace: dict[str, Any]) -> float | None:
    """Restore the same zero-score timeout outcome used on the original pass."""
    if trace.get("failure_stage") == "task_timeout":
        return 0.0
    evaluation = trace.get("evaluation") or {}
    raw_score = evaluation.get("score")
    return float(raw_score) if raw_score is not None else None


class _RecordedClient(AsyncOpenAIClient):
    """Keep build/repair usage, which the planner callback otherwise discards."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs):
        text, usage = await super().generate(**kwargs)
        self.calls.append(dict(usage))
        return text, usage


def run_native_evaluation_round(
    *,
    bench_backend: str,
    bench_id: str,
    run_id: str,
    round_idx: int,
    task_ids: list[str],
    agent_llm: str,
    runs_dir: Path,
    skills_evolution_dir: Path,
    jsonl_path: Path | None = None,
    max_rounds: int = 12,
    max_concurrency: int = 2,
    skill_file: Path | None = None,
    trajectory_tag: str = "",
    split: str = "test",
) -> tuple[Path, Path, dict[str, Any]]:
    adapter = get_adapter(bench_backend)
    source = Path(jsonl_path).resolve() if jsonl_path is not None and str(jsonl_path) else None
    tasks = adapter.load_tasks(source, split=split)
    wanted = {str(t) for t in task_ids}
    selected = [task for task in tasks if str(task.task_id) in wanted]
    missing = sorted(wanted - {str(task.task_id) for task in selected})
    if missing:
        raise RuntimeError(f"{bench_backend} source missing task ids: {missing[:10]}")

    suffix = f"_{trajectory_tag}" if trajectory_tag else ""
    root = runs_dir / bench_id / run_id / f"round_{round_idx:02d}" / "bench_rollouts" / f"{bench_backend}_native_r{round_idx:02d}{suffix}"
    root.mkdir(parents=True, exist_ok=True)
    trace_dir = root / "process_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    skill_path = Path(skill_file).resolve() if skill_file is not None else (
        skills_evolution_dir / bench_id / run_id / f"round_{round_idx:02d}" / "SKILL.md"
    )
    if not skill_path.is_file():
        raise FileNotFoundError(f"Skill workspace missing SKILL.md: {skill_path}")

    # Keep the requested task list separate from the pending subset.  The final
    # bundle must contain both newly-run rows and rows restored from checkpoints;
    # otherwise a resume overwrites the previous bundle with only the new rows.
    requested_tasks = list(selected)

    # Resume safely: every durable evaluation trace is a checkpoint.  A timeout
    # is a completed attempt by policy and must not be rerun automatically.
    pending = []
    for task in selected:
        checkpoint = trace_dir / f"{task.task_id}.json"
        try:
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            if saved.get("failure_stage") == "task_timeout" or (
                saved.get("evaluation") is not None
                and saved.get("final_output") is not None
                and saved.get("success") is True
            ):
                continue
        except Exception:
            pass
        pending.append(task)
    selected = pending

    async def one(task):
        started_at = time.monotonic()
        print(f"[task] starting task_id={task.task_id}", flush=True)
        workspace, context = make_workspace(bench_backend, task)
        context.update(adapter.prepare_workspace(task, workspace))
        context["artifact_dir"] = str(root / "task_artifacts" / task.task_id)
        client = _RecordedClient(model=agent_llm)
        planner = make_planner_call_fn(client, generation_kwargs={"response_format": {"type": "json_object"}, "temperature": 0.2})
        text_call = make_text_call_fn(client)
        runtime = None
        try:
            setup_retries = max(0, int(os.environ.get("MAC_BENCH_SETUP_RETRIES", "0")))
            for attempt in range(setup_retries + 1):
                try:
                    runtime = NativeToolRuntime(adapter, task, context, agent_llm)
                    break
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    transient = ("unable to connect", "failed to fetch", "502 bad gateway",
                                 "connection reset", "connection refused", "timed out",
                                 "temporary failure", "failed to install browsers")
                    retryable = any(marker in message.lower() for marker in transient)
                    if not retryable or attempt >= setup_retries:
                        raise InfrastructureFailure(
                            f"infrastructure failure during {bench_backend} setup for {task.task_id}: {exc}"
                        ) from exc
                    delay = min(30, 5 * (attempt + 1))
                    print(f"[task] setup retry {attempt + 1}/{setup_retries} task_id={task.task_id} "
                          f"wait={delay}s reason={message[:300]}", flush=True)
                    await asyncio.sleep(delay)
            # Some stateful adapters create the authoritative task instruction
            # while attaching MCP/Docker services, so build the prompt after
            # runtime setup has completed.
            prompt = adapter.build_prompt(task, context)
            result = await run_mas_pipeline_with_retries(
                task_text=prompt,
                class_name="Native" + re.sub(r"\W", "_", f"{bench_backend}_{task.task_id}") + "Workflow",
                init_skill_path=skill_path,
                planner_call_fn=planner,
                text_call_fn=text_call,
                tool_call_fn=partial(runtime.run, max_rounds=max_rounds),
                dataset_name=bench_backend,
                max_generation_attempts=6,
                max_execution_attempts=4,
            )
            evaluation = adapter.evaluate(task, result.final_output, context)
            usage = empty_usage_totals()
            for call_usage in [*client.calls, *runtime.llm_usage]:
                _merge_usage(usage, call_usage)
            trace = {
                "schema": "skill_mas_native_trace_v1",
                "bench_backend": bench_backend,
                "task_id": task.task_id,
                "split": task.split,
                "prompt": prompt,
                "task_description": prompt,
                "build_stages": [asdict(s) for s in result.artifacts.stage_traces] if result.artifacts else [],
                "state": result.state,
                "mas_code": result.mas_code,
                "success": result.success,
                "failure_stage": result.failure_stage,
                "failure_reason": result.failure_reason,
                "generation_attempts_used": result.generation_attempts_used,
                "execution_attempts_used": result.execution_attempts_used,
                "retry_events": result.retry_events,
                "final_output": result.final_output,
                "evaluation": {"success": evaluation.success, "score": evaluation.score, "summary": evaluation.summary, "details": evaluation.details},
                "native_tool_trace": runtime.trace,
                "usage_totals": usage,
                "llm_usage": {"build_and_text": client.calls, "native_execution": runtime.llm_usage},
                "usage_scope": "All returned API usage, including build, repair, and execution; missing provider usage cannot be recovered.",
                "elapsed_seconds": round(time.monotonic() - started_at, 3),
                "effective_elapsed_seconds": round(time.monotonic() - started_at, 3),
            }
            _atomic_write_json(trace_dir / f"{task.task_id}.json", trace)
            print(f"[task] finished task_id={task.task_id} score={evaluation.score} "
                  f"passed={evaluation.success} elapsed={trace['elapsed_seconds']}s", flush=True)
            return task.task_id, float(evaluation.score), usage, trace
        finally:
            try:
                if runtime is not None:
                    await runtime.close()
            finally:
                try:
                    await client.aclose()
                finally:
                    adapter.cleanup_workspace(context)

    async def gather_all():
        semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))

        async def bounded(task):
            started_at = time.monotonic()
            async with semaphore:
                try:
                    return await asyncio.wait_for(one(task), timeout=_task_wall_timeout(bench_backend))
                except asyncio.TimeoutError:
                    # A task timeout is an ordinary benchmark outcome. Persist it
                    # as a checkpoint and let the remaining tasks continue.
                    effective_budget = _task_timeout(bench_backend)
                    trace = {
                        "schema": "skill_mas_native_trace_v1",
                        "bench_backend": bench_backend,
                        "task_id": task.task_id,
                        "split": task.split,
                        "success": False,
                        "failure_stage": "task_timeout",
                        "completion_status": "completed_timeout",
                        "failure_reason": f"task exceeded effective budget {effective_budget:.0f}s plus wait window {_TIMEOUT_WAIT_WINDOW_S:.0f}s",
                        "evaluation": {"success": None, "score": None, "summary": "task timeout; score unavailable", "details": {}},
                        "status": "timeout",
                        "final_output": "",
                        "usage_totals": empty_usage_totals(),
                        "native_tool_trace": [],
                        "elapsed_seconds": effective_budget + _TIMEOUT_WAIT_WINDOW_S,
                        "effective_elapsed_seconds": 0.0,
                        "elapsed_excluded_reason": "task_timeout_wait",
                        "effective_budget_seconds": effective_budget,
                        "wait_window_seconds": _TIMEOUT_WAIT_WINDOW_S,
                    }
                    _atomic_write_json(trace_dir / f"{task.task_id}.json", trace)
                    # Include timeout as a zero-valued outcome in aggregate
                    # scoring while keeping score=None in the durable trace.
                    return task.task_id, 0.0, empty_usage_totals(), trace
                except InfrastructureFailure as exc:
                    # A single BeyondSWE repository/image can be unavailable
                    # (for example a missing Docker image or a failed git
                    # repack). Record it and keep the other repositories moving.
                    # It remains retryable on a later resume because this
                    # checkpoint is intentionally not treated as completed.
                    trace = {
                        "schema": "skill_mas_native_trace_v1",
                        "bench_backend": bench_backend,
                        "task_id": task.task_id,
                        "split": task.split,
                        "success": False,
                        "failure_stage": "infrastructure_failure",
                        "completion_status": "skipped_infrastructure_failure",
                        "failure_reason": str(exc),
                        "evaluation": {"success": None, "score": None, "summary": "infrastructure failure; retryable", "details": {}},
                        "status": "infrastructure_failure",
                        "final_output": "",
                        "usage_totals": empty_usage_totals(),
                        "native_tool_trace": [],
                        "elapsed_seconds": round(time.monotonic() - started_at, 3),
                        "effective_elapsed_seconds": 0.0,
                    }
                    _atomic_write_json(trace_dir / f"{task.task_id}.json", trace)
                    print(f"[task] infrastructure failure task_id={task.task_id}: {str(exc)[:400]}", flush=True)
                    return task.task_id, 0.0, empty_usage_totals(), trace

                except OSError as exc:
                    # Local disk/Docker socket failures are retryable setup
                    # failures, never benchmark scores or model generations.
                    trace = {
                        "schema": "skill_mas_native_trace_v1",
                        "bench_backend": bench_backend,
                        "task_id": task.task_id,
                        "split": task.split,
                        "success": False,
                        "failure_stage": "infrastructure_failure",
                        "completion_status": "skipped_infrastructure_failure",
                        "failure_reason": f"local I/O failure: {type(exc).__name__}: {exc}",
                        "evaluation": {"success": None, "score": None, "summary": "local I/O failure; retryable", "details": {}},
                        "status": "infrastructure_failure",
                        "final_output": "",
                        "usage_totals": empty_usage_totals(),
                        "native_tool_trace": [],
                        "elapsed_seconds": round(time.monotonic() - started_at, 3),
                        "effective_elapsed_seconds": 0.0,
                    }
                    _atomic_write_json(trace_dir / f"{task.task_id}.json", trace)
                    return task.task_id, 0.0, empty_usage_totals(), trace

                except Exception as exc:
                    # Persist generation/API/evaluation failures and continue;
                    # a later resume will retry them because only success and
                    # timeout checkpoints are terminal.
                    trace = {
                        "schema": "skill_mas_native_trace_v1",
                        "bench_backend": bench_backend,
                        "task_id": task.task_id,
                        "split": task.split,
                        "success": False,
                        "failure_stage": "generation",
                        "completion_status": "completed_failure",
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                        "evaluation": {"success": None, "score": None, "summary": "generation failure; retryable", "details": {}},
                        "status": "generation_failure",
                        "final_output": "",
                        "usage_totals": empty_usage_totals(),
                        "native_tool_trace": [],
                        "elapsed_seconds": round(time.monotonic() - started_at, 3),
                        "effective_elapsed_seconds": round(time.monotonic() - started_at, 3),
                    }
                    _atomic_write_json(trace_dir / f"{task.task_id}.json", trace)
                    return task.task_id, 0.0, empty_usage_totals(), trace

        if bench_backend == "macbench" and int(max_concurrency) == 1:
            rows = []
            minimum_free = float(os.environ.get("SKILL_MAS_MIN_FREE_GB", "0"))
            for task in selected:
                free_gb = shutil.disk_usage(trace_dir).free / (1024 ** 3)
                if free_gb < minimum_free:
                    raise RuntimeError(f"Storage guard: only {free_gb:.1f} GiB free; "
                                       f"need {minimum_free:.1f} GiB before {task.task_id}")
                row = await bounded(task)
                rows.append(row)
                reason = str(row[3].get("failure_reason") or "").lower()
                if "input/output error" in reason or "no space left on device" in reason:
                    raise InfrastructureFailure(f"Local storage failure; stopping before the next task: {reason}")
            return rows
        return await asyncio.gather(*(bounded(task) for task in selected))

    rows = asyncio.run(gather_all())

    # Rehydrate completed checkpoint rows omitted from ``pending``.  This makes
    # the aggregate bundle idempotent across interrupted/resumed executions.
    rows_by_task = {str(tid): (tid, score, usage, trace) for tid, score, usage, trace in rows}
    for task in requested_tasks:
        tid_s = str(task.task_id)
        if tid_s in rows_by_task:
            continue
        checkpoint = trace_dir / f"{tid_s}.json"
        try:
            trace = json.loads(checkpoint.read_text(encoding="utf-8"))
            score = _checkpoint_score(trace)
            usage = trace.get("usage_totals") or empty_usage_totals()
            rows_by_task[tid_s] = (tid_s, score, usage, trace)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            # A malformed/incomplete checkpoint remains absent and is visible as
            # missing from the bundle instead of silently becoming score 0.
            continue
    rows = [rows_by_task[str(task.task_id)] for task in requested_tasks if str(task.task_id) in rows_by_task]
    aggregate = empty_usage_totals()
    scores: dict[str, float] = {}
    per_task: list[dict[str, Any]] = []
    for tid, score, usage, trace in rows:
        scores[str(tid)] = score
        _merge_usage(aggregate, usage)
        per_task.append({"task_id": str(tid), "score": score, "status": trace.get("status", "completed"), **usage})
    timeout_count = sum(1 for _, _, _, trace in rows if trace.get("failure_stage") == "task_timeout")
    bundle = {"bench_backend": bench_backend, "round_idx": round_idx, "bench_id": bench_id, "run_id": run_id, "task_ids": [str(t.task_id) for t in requested_tasks], "process_trace_dir": str(trace_dir.resolve()), "merged_skill_dir": str(skill_path.parent.resolve()), "per_task_scores": scores, "timeout_count": timeout_count}
    bundle_path = root / f"{bench_backend}_native_bundle_r{round_idx:02d}.json"
    _atomic_write_json(bundle_path, bundle)
    section = {"phase": "native_skill_mas_rollout", "description": f"Skill-MAS native {bench_backend} rollout", "model": agent_llm, "aggregate_usage": aggregate, "per_task": per_task, "pricing_reference": pricing_reference()}
    return bundle_path, root, section


def _collect_state_usage(state: dict[str, Any]) -> dict[str, Any]:
    total = empty_usage_totals()
    seen: set[int] = set()
    for key, value in (state or {}).items():
        if not str(key).startswith("usage_") or not isinstance(value, dict):
            continue
        marker = id(value)
        if marker in seen:
            continue
        seen.add(marker)
        add_usage_totals(total, value)
    return total
