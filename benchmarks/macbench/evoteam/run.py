"""Run the macbench adapter using the external EvoTeam method."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from benchmarks.macbench.evoteam.adapter import MacBenchAdapter

if __name__ == "__main__":
    MacBenchAdapter().cli()
