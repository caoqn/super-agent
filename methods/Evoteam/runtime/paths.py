from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = Path(os.environ.get("EVAL_RESULTS_DIR", str(REPO_ROOT / "results" / "Evoteam"))).expanduser()
