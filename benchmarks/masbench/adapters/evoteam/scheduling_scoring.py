"""Native 20-minute meeting slot scoring."""
import json
from pathlib import Path

def score_schedule(answer: str, task_path: str):
    details = {}
    score = 0.0
    try:
        obj = json.loads(answer)
        day, start, end = (obj[key] for key in ("day", "start_minute", "end_minute"))
        if any(type(x) is not int for x in (day, start, end)):
            raise ValueError("day/start_minute/end_minute must be integers")
        if not (1 <= day <= 5 and 0 <= start < end <= 1440 and end - start == 20 and start % 10 == 0):
            raise ValueError("answer must be a valid 20-minute, 10-minute-grid slot")
        plans = json.loads((Path(task_path) / "task_plans").read_text())["meeting"]["solutions"]
        start_slot = (day - 1) * 144 + start // 10
        end_slot = (day - 1) * 144 + end // 10
        score = float(any(p["start_time"] == start_slot and p["end_time"] == end_slot for p in plans))
    except (ValueError, TypeError, KeyError) as exc:
        details["invalid_prediction"] = str(exc)
    return score, details
