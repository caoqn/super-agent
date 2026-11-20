"""Load the fixed split against an explicitly supplied upstream tree."""
import json
from pathlib import Path

def load_dataset(args, split_dir: Path, source_root: Path):
    split = getattr(args, "split", "test")
    manifest_path = split_dir / f"{split}_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reasoning_path = split_dir / f"reasoning_{split}.json"
    reasoning = {str(row["id"]): row for row in
                 json.loads(reasoning_path.read_text(encoding="utf-8"))}
    items = []
    for category in ("reasoning", "scheduling", "game"):
        for task_id in manifest[category]["ids"]:
            item = {"category": category, "id": str(task_id),
                    "source_root": str(source_root), "split": split}
            if category == "reasoning":
                item["row"] = reasoning[str(task_id)]
            elif category == "scheduling":
                a, b = str(task_id).split("_", 1)
                item["path"] = str(source_root / "Scheduling/meeting/minutes_10/5days" / a / b)
                if not (Path(item["path"]) / "task_plans").is_file():
                    raise FileNotFoundError(f"Scheduling task not found: {task_id}")
            else:
                item["path"] = str(source_root / "Game/custom_map" / str(task_id))
                if not Path(item["path"]).is_file():
                    raise FileNotFoundError(f"Game map not found: {task_id}")
            items.append(item)
    max_items = getattr(args, "max_items", None)
    return items[:max_items] if max_items else items
