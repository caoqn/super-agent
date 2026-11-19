"""Assemble GAIA and MAC-Bench adapters against a pinned upstream AgentNet."""
import argparse
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

UPSTREAM = "325d39f2a940be5fa903d28c411bd3426b8007f5"
ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "methods/agentnet/patches/agentnet-core.patch"
EMBEDDING = ROOT / "methods/agentnet/embedding.py"
ADAPTERS = (
    (Path("benchmarks/gaia/adapters/agentnet"), "agentnet_gaia_run"),
    (Path("benchmarks/macbench/adapters/agentnet"), "agentnet_macbench_run"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, help="Path to the official AgentNet Git checkout")
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if commit != UPSTREAM:
        parser.error(f"Check out upstream AgentNet commit {UPSTREAM} first.")
    upstream = checkout / "AgentNet_Code"
    for adapter_rel, destination_name in ADAPTERS:
        adapter = ROOT / adapter_rel
        destination = ROOT / destination_name
        if destination.exists():
            parser.error(f"Existing run directory: {destination.name}")
        with TemporaryDirectory() as tmp:
            assembled = Path(tmp)
            for part in ("config", "evaluator", "models", "prompt", "src", "utils"):
                shutil.copytree(upstream / part, assembled / part, ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copy2(upstream / "requirements.txt", assembled)
            subprocess.run(["git", "apply", "--unidiff-zero", str(PATCH)], check=True, cwd=assembled)
            shutil.copy2(EMBEDDING, assembled / "src" / "embedding.py")
            shutil.copytree(adapter, assembled, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
            runtime = assembled / "gaia"
            runtime.mkdir(exist_ok=True)
            for module in ("envfile.py", "llm.py", "tracker.py"):
                shutil.copy2(ROOT / "methods/agentnet" / module, runtime / module)
            method_runtime = assembled / "methods" / "agentnet"
            method_runtime.mkdir(parents=True)
            for module in ("__init__.py", "envfile.py", "llm.py", "tracker.py"):
                shutil.copy2(ROOT / "methods/agentnet" / module, method_runtime / module)
            shutil.copytree(assembled, destination)
        print(destination.name)


if __name__ == "__main__":
    main()
