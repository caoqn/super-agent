"""Expose Evoteam runtime imports without replacing the shared benchmarks package."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "methods/Evoteam/runtime"
sys.path.insert(0, str(RUNTIME))
import benchmarks
for directory in (RUNTIME / "benchmarks", ROOT / "benchmarks"):
    if str(directory) not in benchmarks.__path__:
        benchmarks.__path__.append(str(directory))
os.environ.setdefault("DEV_API_BASE", "https://example.invalid/v1")
os.environ.setdefault("DEV_API_KEY", "test-placeholder")
