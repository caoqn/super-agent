"""Run the masbench adapter using the external EvoTeam method."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from benchmarks.masbench.adapters.evoteam.adapter import MASBenchAdapter

if __name__ == "__main__":
    MASBenchAdapter().cli()
