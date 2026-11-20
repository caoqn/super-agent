#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
env_file="${MASBENCH_ENV_FILE:-configs/manager3/masbench.env}"
if [[ -f "$env_file" ]]; then
  set -a
  source "$env_file"
  set +a
elif [[ -n "${MASBENCH_ENV_FILE:-}" ]]; then
  printf 'MASBench config file not found: %s\n' "$env_file" >&2
  exit 1
fi
exec "${MASBENCH_PYTHON:-python3}" -m methods.manager3.masbench.runner "$@"
