#!/usr/bin/env bash
# Usage: bash run_gaia.sh            # Run 3 smoke tasks, then ask to run all 165 tasks
#       bash run_gaia.sh full       # Run the full validation set
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install -U pip
  ./.venv/bin/pip install -r requirements_bench.txt
fi
PY=./.venv/bin/python
if [ ! -f ../.env ]; then echo "First copy ../.env.example to ../.env and configure LLM_API_KEY"; exit 1; fi
if [ "$1" != "full" ]; then
  $PY run_gaia.py --limit 3 --out save/gaia_smoke_$(date +%Y%m%d-%H%M%S)
  read -p "Smoke check passed? Run all 165 tasks [y/N] " yn; [ "$yn" = "y" ] || exit 0
fi
$PY run_gaia.py
