#!/usr/bin/env python3
"""Run MAC-Bench's AOrchestra integration with the pinned external method source."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _baseline_runner import run


if __name__ == "__main__":
    raise SystemExit(run("aorchestra", "configs/aorchestra/macbench.yaml"))
