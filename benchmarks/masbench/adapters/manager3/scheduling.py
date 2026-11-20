"""Load private calendars and check the selected meeting slot."""
from __future__ import annotations

import json
from pathlib import Path


def private_context(path: Path) -> dict:
    return {
        f"Member{i}": json.loads((path / f"Member{i}_final_schedule.json").read_text(encoding="utf-8"))
        for i in range(1, 6)
    }


def evaluate(path: Path, answer: str) -> tuple[float, dict]:
    try:
        prediction = json.loads(answer)
        day, start, end = (prediction[key] for key in ("day", "start_minute", "end_minute"))
        if any(type(value) is not int for value in (day, start, end)):
            raise ValueError("integer values required")
        if not (1 <= day <= 5 and 0 <= start < end <= 1440 and end - start == 20 and start % 10 == 0):
            raise ValueError("invalid 20-minute slot")
        plans = json.loads((path / "task_plans").read_text(encoding="utf-8"))["meeting"]["solutions"]
        absolute_start = (day - 1) * 144 + start // 10
        absolute_end = (day - 1) * 144 + end // 10
        matches = any(
            plan["start_time"] == absolute_start and plan["end_time"] == absolute_end for plan in plans
        )
        return float(matches), {"prediction": prediction}
    except (ValueError, TypeError, KeyError) as exc:
        return 0.0, {"invalid_prediction": str(exc)}
