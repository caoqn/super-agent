"""Scoring helpers aligned with the MASBench release."""
import json
import re
from collections import Counter
from pathlib import Path


def reasoning_f1(reference: str, prediction: str) -> float:
    """MASBench's SQuAD-style normalized token F1."""
    def tokens(value):
        value = str(value).lower().translate(str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"))
        value = re.sub(r"\b(a|an|the)\b", " ", value)
        return " ".join(value.split()).split()

    ref, pred = tokens(reference), tokens(prediction)
    if not ref or not pred:
        return 1.0 if ref == pred else 0.0
    overlap = sum((Counter(ref) & Counter(pred)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def scheduling_correct(data_dir: Path, scenario_id: str, day: int, start_minute: int, end_minute: int) -> bool:
    """Match a proposed slot to a released task_plans solution."""
    if not isinstance(day, int) or isinstance(day, bool) or day < 1 or not (0 <= start_minute < end_minute <= 1440):
        return False
    scenario = data_dir / "Scheduling/scenarios" / scenario_id
    plans = json.loads((scenario / "task_plans").read_text(encoding="utf-8"))["meeting"]
    schedule = json.loads((scenario / "Member1_final_schedule.json").read_text(encoding="utf-8"))
    minutes_per_unit = 60 if schedule.get("time_unit") == "hours" else int(schedule["time_unit"].split("_")[1])
    if day > schedule.get("total_days", 0) or start_minute % minutes_per_unit or end_minute - start_minute != 20:
        return False
    units_per_day = 24 * 60 // minutes_per_unit
    start = (day - 1) * units_per_day + start_minute // minutes_per_unit
    duration = (end_minute - start_minute) // minutes_per_unit
    return any(solution["start_time"] == start and solution["end_time"] == start + duration
               for solution in plans.get("solutions", []))


def game_score(map_path: Path, game_log_path: Path) -> dict:
    """Compare final collected quantities against the map's scoring targets."""
    map_data = json.loads(map_path.read_text(encoding="utf-8"))
    log = json.loads(game_log_path.read_text(encoding="utf-8"))
    scoring = map_data["task"]["scoring"]
    totals = log.get("final_table_items")
    if not isinstance(totals, dict):
        totals = log.get("items_on_tables")
    if not isinstance(totals, dict):
        raise ValueError("Game log must expose final_table_items from the original environment")
    earned = {kind: min(int(totals.get(kind, 0)), int(target)) for kind, target in scoring.items()}
    return {"earned": sum(earned.values()), "maximum": sum(map(int, scoring.values())), "by_resource": earned}
