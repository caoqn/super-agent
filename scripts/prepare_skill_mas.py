"""Assemble Skill-MAS native adapters against a pinned upstream checkout."""

import argparse
from pathlib import Path
import shutil
import subprocess

UPSTREAM = "f6cf17ab3ddbd3893d12b75f08c4b06a23ead873"
ROOT = Path(__file__).resolve().parents[1]
METHOD = ROOT / "methods" / "skill_mas"
PACKAGES = ("skill_mas", "template", "utils", "evolution", "init_skill")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, help="Separate official Skill-MAS Git checkout")
    parser.add_argument("--dataset", choices=("gaia", "macbench", "masbench"), required=True)
    parser.add_argument("--output-dir", type=Path, help="Optional output directory; must not exist")
    args = parser.parse_args()

    checkout = args.checkout.expanduser().resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if commit != UPSTREAM:
        parser.error(f"Check out Skill-MAS commit {UPSTREAM} first.")

    destination = (args.output_dir.expanduser().resolve() if args.output_dir
                   else ROOT / f"skill_mas_{args.dataset}_run")
    if destination.exists():
        parser.error(f"Existing run directory: {destination.name}")
    package = destination / "Skill_MAS"
    tracked = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", UPSTREAM, *PACKAGES, "LICENSE"],
        cwd=checkout, text=True,
    ).splitlines()
    for name in tracked:
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.check_output(["git", "show", f"{UPSTREAM}:{name}"], cwd=checkout))

    for patch in sorted((METHOD / "patches").glob("*.patch")):
        subprocess.run(["git", "apply", "--check", "--directory=Skill_MAS", str(patch)], check=True, cwd=destination)
        subprocess.run(["git", "apply", "--directory=Skill_MAS", str(patch)], check=True, cwd=destination)

    adapters = package / "bench_adapters"
    adapters.mkdir()
    common = ROOT / "benchmarks" / "common" / "skill_mas"
    registry = (common / "__init__.py").read_text()
    for dataset in ("gaia", "macbench", "masbench"):
        registry = registry.replace(f"benchmarks.{dataset}.adapters.skill_mas.adapter", f".{dataset}")
    (adapters / "__init__.py").write_text(registry, encoding="utf-8")
    shutil.copy2(common / "base.py", adapters / "base.py")
    shutil.copy2(METHOD / "native_runtime.py", adapters / "native_runtime.py")
    source = ROOT / "benchmarks" / args.dataset / "adapters" / "skill_mas" / "adapter.py"
    (adapters / f"{args.dataset}.py").write_text(
        source.read_text().replace("from benchmarks.common.skill_mas.base import", "from .base import"),
        encoding="utf-8",
    )
    if args.dataset == "masbench":
        shutil.copytree(
            ROOT / "benchmarks" / "masbench" / "adapters" / "skill_mas" / "masbench_support",
            adapters / "masbench_support",
        )

    print(destination)


if __name__ == "__main__":
    main()
