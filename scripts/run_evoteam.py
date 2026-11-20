"""Validate preserved EvoTeam configs and inspect local data without model calls."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.common.evoteam import inspect_upstream


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Report source compatibility; exits 2 for blockers")
    mode.add_argument("--dry-run", action="store_true", help="Load and validate local data; no models or environment actions")
    parser.add_argument("--evoteam-root", type=Path, default=os.environ.get("EVOTEAM_ROOT"))
    parser.add_argument("--split")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--data-dir", type=Path, help="GAIA split/attachment directory, or prepared MASBench directory")
    parser.add_argument("--macbench-root", type=Path, default=os.environ.get("MAC_BENCH_ROOT"))
    parser.add_argument("--masbench-root", type=Path, default=os.environ.get("MASBENCH_ROOT"))
    args = parser.parse_args(argv)
    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be positive")
    config = json.loads(args.config.read_text())
    benchmark = config["benchmark"]
    if benchmark not in {"gaia", "macbench", "masbench"}:
        parser.error("unsupported benchmark")
    flags = config["args"]
    team = flags[flags.index("--team") + 1]
    if Path(team).name != team or team in {".", ".."}:
        parser.error("invalid configured pool name")
    pool = ROOT / "configs/evoteam/pools" / team
    if not (pool / "pool.yaml").is_file():
        parser.error(f"missing pool: {pool}")
    if args.check:
        if not args.evoteam_root:
            parser.error("--check requires --evoteam-root or EVOTEAM_ROOT")
        report = inspect_upstream(args.evoteam_root, benchmark)
        report["pool"] = str(pool.relative_to(ROOT))
        print(json.dumps(report, indent=2))
        return 2
    if not args.dry_run:
        parser.error("Experiment execution is blocked by upstream compatibility; use --check or --dry-run. See methods/evoteam/README.md.")
    args.split = args.split or flags[flags.index("--split") + 1]
    module = importlib.import_module(f"benchmarks.{benchmark}.adapters.evoteam.data" if benchmark != "macbench"
                                    else "benchmarks.macbench.adapters.evoteam.catalog")
    if benchmark == "gaia":
        if args.data_dir:
            module.DATA_DIR = args.data_dir.expanduser().resolve()
            module.VAL_FILES_DIR = module.DATA_DIR / "val_files"
        items = module.load_gaia_data(args.split)
        ids = [str(row["task_id"]) for row in items]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate GAIA task IDs")
        if args.split == "test_100":
            expected = {str(row["task_id"]) for row in json.loads((ROOT / "benchmarks/gaia/manifest.json").read_text())["tasks"]}
            if set(ids) != expected:
                raise ValueError("GAIA test_100 task IDs differ from the checked-in manifest")
        for row in items:
            name = row.get("file_name")
            if name:
                candidates = [module.VAL_FILES_DIR / name, module.VAL_FILES_DIR / str(row["task_id"]) / name]
                if not any(p.is_file() for p in candidates):
                    raise FileNotFoundError(f"Missing GAIA attachment: {name}")
    elif benchmark == "macbench":
        if not args.macbench_root:
            parser.error("set MAC_BENCH_ROOT or --macbench-root")
        if args.split not in {"all", "dev", "test"}:
            parser.error("MAC-Bench split must be all, dev, or test")
        _, rows, _ = module._load_catalog(args.macbench_root)
        items = [row for row in rows if args.split == "all" or row["split"] == args.split]
        ids = [str(row["task_id"]) for row in items]
    else:
        if not args.masbench_root:
            parser.error("set MASBENCH_ROOT or --masbench-root")
        if args.split not in {"train", "test"}:
            parser.error("MASBench split must be train or test")
        split_dir = args.data_dir or Path(os.environ.get("MASBENCH_SPLIT_DIR", ROOT / ".artifacts/evoteam/masbench_split18_test90"))
        expected = json.loads((ROOT / f"data/manifests/evoteam/masbench/{args.split}_manifest.json").read_text())
        if json.loads((split_dir / f"{args.split}_manifest.json").read_text()) != expected:
            raise ValueError("Prepared MASBench manifest differs from the checked-in selection")
        limit = args.max_items
        args.max_items = None
        items = module.load_dataset(args, split_dir, args.masbench_root.expanduser().resolve())
        args.max_items = limit
        ids = [f"{row['category']}__{row['id']}" for row in items]
    selected_ids = ids[:args.max_items] if args.max_items else ids
    print(json.dumps({"benchmark": benchmark, "split": args.split, "pool": team,
                      "total_items": len(items), "selected_items": len(selected_ids),
                      "task_ids": selected_ids, "mode": "data_only_no_model_calls",
                      "execution_status": "blocked_upstream_compatibility"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
