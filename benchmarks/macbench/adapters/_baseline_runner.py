"""Thin launcher for baseline integrations supplied by an external MAC-Bench checkout."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
MAC_BENCH_REVISION = "c70d962ee8e428a6d9bae42af8d92e086581ee0d"
AORCHESTRA_REVISION = "14a1a2051d6b03c479b706f8f555a60b8419e3b5"


def verify_revision(root: Path, expected: str, label: str) -> None:
    try:
        actual = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"{label} must be a Git checkout at {expected}") from exc
    if actual != expected:
        raise ValueError(f"{label} revision mismatch: expected {expected}, found {actual}")


def build_command(method: str, config: Path, args: argparse.Namespace) -> list[str]:
    command = [
        str(args.python), "-m", "mac_bench", "evaluate",
        "--task-id", args.task_id,
        "--method", method,
        "--method-config", str(config),
        "--repetitions", str(args.repetitions),
        "--max-tokens", str(args.max_tokens),
        "--max-model-calls", str(args.max_model_calls),
        "--max-tool-calls", str(args.max_tool_calls),
        "--max-seconds", str(args.max_seconds),
        "--out", str(args.output_dir),
    ]
    if args.dry_run:
        command.append("--dry-run")
    return command


def resolve_budgets(root: Path, task_id: str, args: argparse.Namespace,
                    parser: argparse.ArgumentParser) -> dict[str, int]:
    fields = ("max_tokens", "max_model_calls", "max_tool_calls", "max_seconds")
    explicit = {field: getattr(args, field) for field in fields}
    if all(value is not None for value in explicit.values()):
        budgets = explicit
    else:
        path = root / "configs/macbench/task_budgets.json"
        try:
            task_budgets = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"cannot load MAC-Bench task budgets at {path}: {exc}")
        if task_id not in task_budgets:
            parser.error(
                f"no paper budget is recorded for {task_id}; specify all four "
                "budget flags explicitly for a development task"
            )
        row = task_budgets[task_id]
        budgets = {
            "max_tokens": row["max_tokens"],
            "max_model_calls": row["max_model_calls"],
            "max_tool_calls": row["max_tool_calls"],
            "max_seconds": row["max_seconds"],
        }
        for field, value in explicit.items():
            if value is not None:
                budgets[field] = value
    if min(budgets.values()) <= 0:
        parser.error("all budgets must be positive")
    return budgets


def run(method: str, config_relative: str, argv=None) -> int:
    parser = argparse.ArgumentParser(description=f"Run MAC-Bench {method} baseline")
    parser.add_argument("--upstream", type=Path, required=True,
                        help="External MAC_Bench checkout pinned by this adapter")
    parser.add_argument("--python", type=Path, default=Path(sys.executable),
                        help="Python interpreter with the selected MAC-Bench extras installed")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-model-calls", type=int, default=None)
    parser.add_argument("--max-tool-calls", type=int, default=None)
    parser.add_argument("--max-seconds", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    upstream = args.upstream.expanduser().resolve()
    if not (upstream / "src/mac_bench").is_dir():
        parser.error("--upstream must be a MAC-Bench source checkout")
    try:
        verify_revision(upstream, MAC_BENCH_REVISION, "MAC-Bench")
        if method == "aorchestra":
            source = upstream / ".cache/baselines/AOrchestra"
            verify_revision(source, AORCHESTRA_REVISION, "AOrchestra")
    except ValueError as exc:
        parser.error(str(exc))

    config = ROOT / config_relative
    if not config.is_file():
        parser.error(f"missing tracked method configuration: {config}")
    if args.repetitions <= 0:
        parser.error("repetitions must be positive")
    budgets = resolve_budgets(ROOT, args.task_id, args, parser)
    for field, value in budgets.items():
        setattr(args, field, value)

    output = args.output_dir.expanduser().resolve()
    command = build_command(method, config, args)
    command[command.index("--out") + 1] = str(output)
    env = os.environ.copy()
    env["MAC_BENCH_ROOT"] = str(upstream)
    source_path = str(upstream / "src")
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_path, existing_path) if part
    )
    try:
        return subprocess.run(command, cwd=upstream, env=env, check=False).returncode
    except OSError as exc:
        print(f"Cannot start MAC-Bench: {exc}", file=sys.stderr)
        return 2
