"""Rebuild fixed Reasoning split files from the locally provisioned upstream dataset."""
import argparse, json, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--source", required=True, type=Path)
p.add_argument("--output", type=Path, default=ROOT/"data/masbench_split18_test90")
a = p.parse_args()
rows = json.loads((a.source/"Reasoning/reasoning.json").read_text())
if not isinstance(rows, list):
    raise ValueError("Expected a list in Reasoning/reasoning.json")
by_id = {str(row["id"]): row for row in rows}
a.output.mkdir(parents=True, exist_ok=True)
for split in ("train", "test"):
    manifest = ROOT/"benchmarks/masbench"/f"{split}_manifest.json"
    ids = json.loads(manifest.read_text())["reasoning"]["ids"]
    selected = [by_id[str(task_id)] for task_id in ids]
    target = a.output/f"reasoning_{split}.json"
    if target.exists() or (a.output/manifest.name).exists():
        raise FileExistsError(f"Output already exists for {split}; use an empty directory")
    target.write_text(json.dumps(selected, ensure_ascii=False, indent=2))
    shutil.copy2(manifest, a.output/manifest.name)
    print(f"{split}: {len(selected)} Reasoning cases")
