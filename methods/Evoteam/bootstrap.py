"""Connect benchmark adapters to an external, experiment-compatible EvoTeam checkout."""
import importlib
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_URL = "https://github.com/DEEP-JLU/EvoTeam"


def source_root():
    value = os.environ.get("EVOTEAM_ROOT", "").strip() or str(REPO_ROOT / "external/EvoTeam")
    if not Path(value).expanduser().exists():
        raise RuntimeError("Set EVOTEAM_SOURCE_REPOSITORY and run python scripts/setup_evoteam.py "
                           "or set EVOTEAM_ROOT to the pinned runtime checkout; see methods/Evoteam/README.md. "
                           f"Official method repository: {SOURCE_URL}.")
    root = Path(value).expanduser().resolve()
    required = ("benchmarks/adapter.py", "core/runner.py", "core/task_budget.py",
                "core/team_selector.py", "main.py", "tools/macbench.py")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise RuntimeError("Incompatible EvoTeam source checkout: missing " + ", ".join(missing)
                           + ". These adapters require the historical individual-benchmark experiment API; see methods/Evoteam/upstream.yaml.")
    lock = json.loads((REPO_ROOT / "methods/Evoteam/source-lock.json").read_text())
    for name, expected in lock["files"].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Incompatible EvoTeam source: {name} differs from pinned {lock['revision']}. Run scripts/setup_evoteam.py into a fresh directory.")
    return root


def configure_source():
    root = source_root()
    os.environ.setdefault("EVAL_RESULTS_DIR", str(REPO_ROOT / "results/Evoteam"))
    # Keep shared benchmark data/scorers in this evaluation repository.
    sys.path.insert(0, str(REPO_ROOT))
    benchmarks = importlib.import_module("benchmarks")
    local = str(REPO_ROOT / "benchmarks")
    external = str(root / "benchmarks")
    benchmarks.__path__[:] = [local, external]
    sys.path.insert(0, str(root))
    return root
