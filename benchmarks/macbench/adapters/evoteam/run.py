"""Data/compatibility entry point; model execution awaits upstream porting."""
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[4]
    sys.argv[1:1] = ["--config", str(root / "configs/evoteam/macbench.json")]
    runpy.run_path(str(root / "scripts/run_evoteam.py"), run_name="__main__")
