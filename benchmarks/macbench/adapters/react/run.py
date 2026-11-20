#!/usr/bin/env python3
"""Run MAC-Bench's pinned ReAct baseline and official evaluator."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _baseline_runner import run


if __name__ == "__main__":
    raise SystemExit(run("react_single", "configs/react/macbench.yaml"))
