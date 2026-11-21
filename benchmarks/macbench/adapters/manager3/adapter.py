"""Benchmark-facing entry point for the manager3 MAC-Bench baseline."""
import sys

from methods.manager3.macbench.runner import main


if __name__ == '__main__':
    sys.exit(main())
