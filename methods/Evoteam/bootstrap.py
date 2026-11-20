"""Connect benchmark adapters to an external, experiment-compatible EvoTeam checkout."""
import importlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_URL = "https://github.com/caoqn/EvoTeam"


def source_root():
    value = os.environ.get("EVOTEAM_ROOT", "").strip()
    if not value:
        raise RuntimeError(f"Set EVOTEAM_ROOT to an experiment-compatible source checkout; see {SOURCE_URL} and methods/Evoteam/README.md")
    root = Path(value).expanduser().resolve()
    required = ("benchmarks/adapter.py", "core/runner.py", "core/task_budget.py",
                "core/team_selector.py", "main.py", "tools/macbench.py")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise RuntimeError("Incompatible EvoTeam source checkout: missing " + ", ".join(missing)
                           + ". These adapters require the historical individual-benchmark experiment API; see methods/Evoteam/upstream.yaml.")
    return root


def configure_source():
    root = source_root()
    # Keep shared benchmark data/scorers in this evaluation repository.
    sys.path.insert(0, str(REPO_ROOT))
    benchmarks = importlib.import_module("benchmarks")
    local = str(REPO_ROOT / "benchmarks")
    external = str(root / "benchmarks")
    benchmarks.__path__[:] = [local, external]
    sys.path.insert(0, str(root))
    return root
