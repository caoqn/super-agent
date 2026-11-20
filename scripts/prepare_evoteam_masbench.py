"""Rebuild EvoTeam's 18/90 fixed split from separately obtained MASBench data."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / ".artifacts/evoteam/masbench_split18_test90")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    rows = json.loads((args.source / "Reasoning/reasoning.json").read_text())
    by_id = {str(row["id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate Reasoning IDs in source")
    manifests = {split: json.loads((ROOT / f"data/manifests/evoteam/masbench/{split}_manifest.json").read_text())
                 for split in ("train", "test")}
    for category in ("reasoning", "scheduling", "game"):
        train, test = (manifests[s][category]["ids"] for s in ("train", "test"))
        if set(train) & set(test):
            raise ValueError(f"Train/test overlap: {category}")
        for split in manifests:
            entry = manifests[split][category]
            if len(set(entry["ids"])) != entry["n"] or len(entry["ids"]) != entry["n"]:
                raise ValueError(f"Invalid {split} {category} manifest")
    selected = {split: [by_id[str(i)] for i in manifests[split]["reasoning"]["ids"]]
                for split in manifests}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        staged = Path(temporary) / "split"
        staged.mkdir()
        for split in manifests:
            (staged / f"reasoning_{split}.json").write_text(json.dumps(selected[split], ensure_ascii=False, indent=2) + "\n")
            shutil.copy2(ROOT / f"data/manifests/evoteam/masbench/{split}_manifest.json", staged)
        staged.rename(args.output)
    print(f"Prepared {args.output}; Reasoning: train=6, test=30; total manifest cases: train=18, test=90")


if __name__ == "__main__":
    main()
