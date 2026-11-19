#!/bin/sh
set -eu
BENCH_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec python3 "$BENCH_ROOT/benchmarks/gaia/adapters/super_agent/adapter.py" "$@"
