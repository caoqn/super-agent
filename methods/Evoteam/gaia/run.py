"""Run the preserved method adapter from the repository root."""
import sys
from pathlib import Path

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(RUNTIME))
from benchmarks.adapter_gaia import GAIAAdapter

if __name__ == "__main__":
    GAIAAdapter().cli()
