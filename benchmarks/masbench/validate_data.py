#!/usr/bin/env python3
"""Validate the selected MASBench subset against its selection manifest."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def validate(data_dir: Path) -> dict:
    manifest = json.loads((data_dir / "selection_manifest.json").read_text(encoding="utf-8"))
    counts = {}

    reasoning = json.loads((data_dir / "Reasoning/reasoning_subset.json").read_text(encoding="utf-8"))
    expected_reasoning = manifest["reasoning"]["ids"]
    actual_reasoning = [row.get("id") for row in reasoning]
    if len(reasoning) != manifest["reasoning"]["n"] or actual_reasoning != expected_reasoning:
        raise ValueError("Reasoning IDs/count/order do not match selection_manifest.json")
    for row in reasoning:
        if not all(row.get(key) for key in ("question", "answer", "context", "supporting_facts")):
            raise ValueError(f"Reasoning sample {row.get('id')} is missing required fields")
        if not all(row["context"].get(agent) for agent in ("agent_a", "agent_b")):
            raise ValueError(f"Reasoning sample {row['id']} must have private context for both agents")
    counts["reasoning"] = len(reasoning)

    game_ids = manifest["game"]["ids"]
    for name in game_ids:
        path = data_dir / "Game/maps" / name
        row = json.loads(path.read_text(encoding="utf-8"))
        if not row.get("map", {}).get("agents") or not row.get("task", {}).get("scoring"):
            raise ValueError(f"Game map {name} lacks agents or scoring targets")
    counts["game"] = len(game_ids)

    scheduling_ids = manifest["scheduling"]["ids"]
    for scenario_id in scheduling_ids:
        scenario = data_dir / "Scheduling/scenarios" / scenario_id
        plans = json.loads((scenario / "task_plans").read_text(encoding="utf-8"))
        if not plans.get("meeting", {}).get("solutions"):
            raise ValueError(f"Scheduling scenario {scenario_id} has no meeting solutions")
        for member in range(1, 6):
            info = json.loads((scenario / f"Member{member}_info.json").read_text(encoding="utf-8"))
            schedule = json.loads((scenario / f"Member{member}_final_schedule.json").read_text(encoding="utf-8"))
            if info.get("member_id") != member or not schedule.get("activities"):
                raise ValueError(f"Scheduling scenario {scenario_id} has invalid member {member} data")
    counts["scheduling"] = len(scheduling_ids)

    if any(value != 30 for value in counts.values()):
        raise ValueError(f"Expected 30 examples per category, got {counts}")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    args = parser.parse_args()
    counts = validate(args.data_dir.resolve())
    print("Validated selected MASBench data: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    print(f"Total: {sum(counts.values())}")


if __name__ == "__main__":
    main()
