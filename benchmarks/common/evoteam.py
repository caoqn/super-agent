"""Source checks for EvoTeam integrations; contains no method implementation."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = json.loads((REPO_ROOT / "methods/evoteam/upstream.json").read_text())


def inspect_upstream(root: Path, benchmark: str) -> dict:
    """Inspect source without importing it, loading credentials, or calling models."""
    root = root.expanduser().resolve()
    issues = []
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    if revision != SOURCE["revision"]:
        issues.append(f"Expected a Git checkout at {SOURCE['revision']}; found {revision or 'no revision'}")
    required = ["benchmarks/adapter.py", "core/runner.py", "runtime_pool.py",
                "tools/primitives.py"]
    if benchmark == "gaia":
        required += [f"tools/{name}.py" for name in
                     ("read_file", "write_file", "bash", "web_search", "web_fetch")]
    if benchmark == "macbench":
        required += ["core/task_budget.py"]
    for name in required:
        if not (root / name).is_file():
            issues.append(f"Missing upstream file: {name}")
    if benchmark in {"macbench", "masbench"} and (root / "benchmarks/adapter.py").is_file():
        try:
            tree = ast.parse((root / "benchmarks/adapter.py").read_text())
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BenchmarkAdapter")
            names = {n.name for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            hooks = set()
            if benchmark == "macbench":
                hooks |= {"configure_native_runner", "native_tool_schemas", "execute_native_tool"}
            if benchmark == "masbench":
                hooks |= {"native_tool_schemas", "execute_native_tool", "validate_native_message",
                          "runtime_team_capacity_members", "filter_reflection_handoff_trace",
                          "result_for_evolution_reflection", "eval_result_for_evolution_reflection",
                          "configure_native_runner"}
            for hook in sorted(hooks - names):
                issues.append(f"Missing BenchmarkAdapter hook: {hook}")
        except (SyntaxError, StopIteration) as exc:
            issues.append(f"Cannot inspect upstream BenchmarkAdapter: {exc}")
    return {"benchmark": benchmark, "source": SOURCE["repository"],
            "expected_revision": SOURCE["revision"], "actual_revision": revision,
            "issues": issues, "execution_status": "blocked_upstream_compatibility"}


def require_upstream(benchmark: str) -> Path:
    """Keep preserved adapters from silently executing against incompatible code."""
    value = os.environ.get("EVOTEAM_ROOT")
    if not value:
        raise RuntimeError("Set EVOTEAM_ROOT to the separately installed official source. See methods/evoteam/README.md.")
    report = inspect_upstream(Path(value), benchmark)
    detail = "\n".join(report["issues"]) or "End-to-end compatibility has not been validated."
    raise RuntimeError(
        "EvoTeam experiment execution is not released for this official revision:\n"
        + detail + "\nUse scripts/run_evoteam.py --check or --dry-run. See docs/evoteam-reproducibility.md."
    )
