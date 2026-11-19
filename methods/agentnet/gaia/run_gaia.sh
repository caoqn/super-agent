#!/usr/bin/env bash
# 用法: bash run_gaia.sh            # 先冒烟 3 题，再问你是否跑全量 165 题
#       bash run_gaia.sh full       # 直接跑全量
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install -U pip
  ./.venv/bin/pip install -r requirements_bench.txt
fi
PY=./.venv/bin/python
if [ ! -f ../.env ]; then echo "请先: cp ../.env.example ../.env 并填写 LLM_API_KEY 等"; exit 1; fi
if [ "$1" != "full" ]; then
  $PY run_gaia.py --limit 3 --out save/gaia_smoke_$(date +%Y%m%d-%H%M%S)
  read -p "冒烟通过？跑全量 165 题 [y/N] " yn; [ "$yn" = "y" ] || exit 0
fi
$PY run_gaia.py
