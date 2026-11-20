"""Validate the native public and private catalogues without exposing them to agents."""
import json
from pathlib import Path
from typing import Any

def _catalog_paths(root: Path) -> tuple[Path, Path]:
    root = Path(root).expanduser().resolve()
    data_dir = root / "data" if (root / "data" / "tasks.json").is_file() else root
    return data_dir / "tasks.json", data_dir / "public_tasks.json"

def _load_catalog(root: Path) -> tuple[Path, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    tasks_path, public_path = _catalog_paths(root)
    if not tasks_path.is_file() or not public_path.is_file():
        raise FileNotFoundError(
            "MAC-Bench needs data/tasks.json and data/public_tasks.json under "
            f"{Path(root).expanduser().resolve()}"
        )
    catalogue = json.loads(tasks_path.read_text(encoding="utf-8"))
    public_catalogue = json.loads(public_path.read_text(encoding="utf-8"))
    if catalogue.get("format") != "collaboration.tasks.v1":
        raise ValueError(f"Unsupported MAC-Bench catalogue format: {tasks_path}")
    if public_catalogue.get("format") != "mac_bench.public_tasks.v1":
        raise ValueError(f"Unsupported MAC-Bench public catalogue format: {public_path}")

    rows = catalogue.get("tasks")
    public_rows = public_catalogue.get("tasks")
    if not isinstance(rows, list) or not isinstance(public_rows, list):
        raise ValueError("MAC-Bench catalogues must contain a tasks array")

    tasks_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in tasks_by_id:
            raise ValueError(f"Missing or duplicate MAC-Bench task ID: {task_id!r}")
        tasks_by_id[task_id] = row

    public_by_id: dict[str, dict[str, Any]] = {}
    for row in public_rows:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in public_by_id:
            raise ValueError(f"Missing or duplicate public MAC-Bench task ID: {task_id!r}")
        public_by_id[task_id] = row
    if tasks_by_id.keys() != public_by_id.keys():
        missing = sorted(tasks_by_id.keys() - public_by_id.keys())
        extra = sorted(public_by_id.keys() - tasks_by_id.keys())
        raise ValueError(f"MAC-Bench public/task catalogue mismatch: missing={missing}, extra={extra}")

    data_dir = tasks_path.parent.resolve()
    for task_id, row in tasks_by_id.items():
        public_row = public_by_id[task_id]
        if row.get("world_revision") != public_row.get("world_revision"):
            raise ValueError(f"MAC-Bench task revision mismatch for {task_id}")
        relative = Path(str(row.get("bundle") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"MAC-Bench bundle path escapes data directory: {task_id}")
        bundle = (data_dir / relative).resolve()
        if not bundle.is_relative_to(data_dir) or not bundle.is_dir():
            raise FileNotFoundError(f"MAC-Bench task bundle is unavailable: {task_id} ({bundle})")
        instruction = public_row.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"MAC-Bench public instruction is missing for {task_id}")

    return data_dir, rows, public_by_id
