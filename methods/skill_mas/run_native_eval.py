"""Run Skill-MAS native evaluation from the public experiment repository."""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True, help="Generated skill_mas_<dataset>_run directory")
    parser.add_argument("--backend", choices=("gaia", "macbench", "masbench"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task-ids", required=True, help="Comma-separated task IDs")
    parser.add_argument("--skill-file", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test", "dev"), default="test")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data/manifests/skill_mas/masbench")
    args = parser.parse_args()

    runtime = args.runtime.expanduser().resolve()
    sys.path.insert(0, str(runtime))
    os.environ["SUPER_AGENT_ROOT"] = str(ROOT)
    os.environ["SKILL_MAS_MANIFEST_DIR"] = str(args.manifest_dir.expanduser().resolve())
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    from methods.skill_mas.native_eval import run_native_evaluation_round

    bundle, run_root, usage = run_native_evaluation_round(
        bench_backend=args.backend,
        bench_id=f"{args.backend}_native",
        run_id="skill_mas",
        round_idx=0,
        task_ids=[value.strip() for value in args.task_ids.split(",") if value.strip()],
        agent_llm=args.model,
        runs_dir=args.runs_dir.expanduser().resolve(),
        skills_evolution_dir=ROOT / "runs" / "skills",
        skill_file=args.skill_file.expanduser().resolve(),
        max_rounds=args.max_rounds,
        max_concurrency=args.max_concurrency,
        split=args.split,
    )
    print(f"bundle={bundle}\nrun_root={run_root}\nusage={usage}")


if __name__ == "__main__":
    main()
