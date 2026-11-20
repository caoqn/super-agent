"""Run the gaia adapter using the external EvoTeam method."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from methods.Evoteam.gaia.adapter import GAIAAdapter

if __name__ == "__main__":
    GAIAAdapter().cli()
