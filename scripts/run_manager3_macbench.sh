#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
env_file="${MAC_BENCH_ENV_FILE:-configs/manager3/macbench.env}"
if [[ -f "$env_file" ]]; then
  set -a
  source "$env_file"
  set +a
elif [[ -n "${MAC_BENCH_ENV_FILE:-}" ]]; then
  printf 'MAC-Bench config file not found: %s\n' "$env_file" >&2
  exit 1
fi
if [[ -n "${MAC_BENCH_ROOT:-}" ]]; then
  export PYTHONPATH="${MAC_BENCH_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
fi
exec "${MAC_BENCH_PYTHON:-python3}" -m methods.manager3.macbench.runner "$@"
