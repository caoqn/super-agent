"""Assemble the GAIA and MAC-Bench adapters against a pinned upstream AgentNet."""
import argparse
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

UPSTREAM = "325d39f2a940be5fa903d28c411bd3426b8007f5"
ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, help="Path to the official AgentNet Git checkout")
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if commit != UPSTREAM:
        parser.error(f"Check out upstream AgentNet commit {UPSTREAM} first.")
    upstream = checkout / "AgentNet_Code"
    for name in ("agentnet_gaia", "agentnet_macbench"):
        adapter = ROOT / name
        destination = ROOT / f"{name}_run"
        if destination.exists():
            parser.error(f"Existing run directory: {destination.name}")
        with TemporaryDirectory() as tmp:
            assembled = Path(tmp)
            for part in ("config", "evaluator", "models", "prompt", "src", "utils"):
                shutil.copytree(upstream / part, assembled / part, ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copy2(upstream / "requirements.txt", assembled)
            subprocess.run(["git", "apply", "--unidiff-zero", str(adapter / "agentnet-core.patch")],
                           check=True, cwd=assembled)
            shutil.copytree(adapter, assembled, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("agentnet-core.patch", "__pycache__"))
            shutil.copytree(assembled, destination)
        print(destination.name)


if __name__ == "__main__":
    main()
