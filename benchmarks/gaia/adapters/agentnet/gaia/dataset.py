"""Load GAIA (2023, parquet layout of Oct-2025) into AgentNet Task objects."""
import json
import os

from src.task import Task

ABILITIES = ["reasoning", "knowledge", "web_search", "file_processing", "calculation"]


def task_type_of(level, has_file):
    return f"gaia_L{level}_{'file' if has_file else 'web'}"


def register_task_types():
    """Inject GAIA task types into AgentNet's global maps (same dict objects, updated in place)."""
    from config.setting import BASIC_PROBLEM_COMPLEXITY, task_to_ability_map
    for level in (1, 2, 3):
        task_to_ability_map[task_type_of(level, False)] = ["reasoning", "knowledge", "web_search"]
        task_to_ability_map[task_type_of(level, True)] = ["reasoning", "file_processing", "calculation"]
        BASIC_PROBLEM_COMPLEXITY[task_type_of(level, False)] = 1.0
        BASIC_PROBLEM_COMPLEXITY[task_type_of(level, True)] = 1.0


def load_gaia(root, split="validation", levels=None, data_file=None):
    """Returns list of dict rows. `data_file` may be a .parquet or .jsonl."""
    path = data_file or os.path.join(root, "2023", split, "metadata.parquet")
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
    else:
        import pandas as pd
        rows = pd.read_parquet(path).to_dict("records")
    out = []
    for r in rows:
        level = int(r.get("Level"))
        if levels and level not in levels:
            continue
        fp = r.get("file_path") or ""
        if fp and not os.path.isabs(fp):
            fp = os.path.join(root, fp)
        if not r.get("file_name"):
            fp = ""
        out.append({
            "task_id": r["task_id"],
            "question": r["Question"],
            "level": level,
            "answer": str(r.get("Final answer", "")),
            "file_name": r.get("file_name") or "",
            "file_path": fp,
        })
    return out


def to_agentnet_task(i, row):
    q = row["question"]
    if row["file_name"]:
        q += (f"\n\n[An attached file is provided with this task: '{row['file_name']}'. "
              f"Agents executing the task can read it with the read_attachment tool or open it in the python tool.]")
    return Task(
        task_id=i,
        task_type=task_type_of(row["level"], bool(row["file_name"])),
        complexity=0.3,
        major_problem=q,
        correct_answer=row["answer"],
    )
