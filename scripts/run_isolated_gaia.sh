#!/bin/sh
set -eu

# Start each benchmark with a fresh DSH home and state directory. This keeps
# prior orchestrator/worker sessions, tickets, and model histories out of the run.
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PROJECT_DIR=${SUPERAGENT_PROJECT_DIR:?Set SUPERAGENT_PROJECT_DIR to a built external Super Agent checkout}
DSH_SOURCE_HOME=${DSH_SOURCE_HOME:-$HOME/.dsh}
export DSH_SOURCE_HOME
[ -f "$PROJECT_DIR/package.json" ] || { echo "SuperAgent project not found: $PROJECT_DIR" >&2; exit 2; }
RUN_ID=${RUN_ID:-gaia-8parallel-1200s-clean-$(date +%Y%m%d-%H%M%S)}
PORT=${PORT:-3083}
RUN_ROOT=${RUN_ROOT:-$ROOT_DIR/runs/$RUN_ID}
DSH_HOME_NEW=$RUN_ROOT/dsh-home
STATE_DIR=$RUN_ROOT/state
PROFILE_DIR=$DSH_HOME_NEW/profiles/superagent

case "$RUN_ID" in *[!A-Za-z0-9_-]*|'') echo 'Invalid RUN_ID' >&2; exit 2;; esac
case "$PORT" in *[!0-9]*|'') echo 'Invalid PORT' >&2; exit 2;; esac
if [ -e "$RUN_ROOT" ]; then echo "Run directory already exists: $RUN_ROOT" >&2; exit 2; fi
mkdir -p "$PROFILE_DIR" "$STATE_DIR"

# Copy only the profile definition. Do not copy credentials, sessions,
# storages, logs, or the old orchestrator SQLite database.
cp "$DSH_SOURCE_HOME"/profiles/superagent/package.json "$PROFILE_DIR/package.json"
cp "$DSH_SOURCE_HOME"/profiles/superagent/pnpm-workspace.yaml "$PROFILE_DIR/pnpm-workspace.yaml"
cp "$DSH_SOURCE_HOME"/profiles/superagent/cordis.yml "$PROFILE_DIR/cordis.yml"
cp "$DSH_SOURCE_HOME"/profiles/superagent/cordis.patch.yml "$PROFILE_DIR/cordis.patch.yml"
mkdir -p "$PROFILE_DIR/node_modules"
ln -s "$PROJECT_DIR" "$PROFILE_DIR/node_modules/dsh-superagent"

# Point the copied profile at this run's private state and port.
python3 - "$PROFILE_DIR/cordis.patch.yml" "$STATE_DIR" "$PORT" <<'PY'
from pathlib import Path
import sys
import re
p = Path(sys.argv[1])
s = p.read_text()
s, n = re.subn(r'(?m)^(\s*stateDir:)\s*.*$', lambda m: m.group(1) + ' ' + sys.argv[2], s)
if n != 1:
    raise SystemExit('Expected exactly one stateDir in copied DSH profile')
s, n = re.subn(r'(?m)^(\s*port:)\s*\d+\s*$', lambda m: m.group(1) + ' ' + sys.argv[3], s)
if n != 1:
    raise SystemExit('Expected exactly one port in copied DSH profile')
p.write_text(s)
PY

SANDBOX_PROFILE=$RUN_ROOT/session-isolation.sb
{
  printf '(version 1)\n(allow default)\n'
  for denied in \
    "$DSH_SOURCE_HOME" \
    "$ROOT_DIR/benchmarks/gaia" \
    "$ROOT_DIR/results" \
    "$ROOT_DIR/results/analysis"; do
    printf '(deny file-read* (subpath "%s"))\n' "$denied"
  done
  for previous_run in "$ROOT_DIR"/runs/*; do
    [ -e "$previous_run" ] || continue
    [ "$previous_run" = "$RUN_ROOT" ] || printf '(deny file-read* (subpath "%s"))\n' "$previous_run"
  done
} > "$SANDBOX_PROFILE"

export DSH_HOME="$DSH_HOME_NEW"
export RUN_ROOT STATE_DIR PORT RUN_ID
python3 - "$SANDBOX_PROFILE" <<'PY'
import os
import subprocess
import sys
import yaml

with open(os.path.join(os.environ.get('DSH_SOURCE_HOME', os.path.expanduser('~/.dsh')), '.credentials.yaml'), encoding='utf-8') as source:
    key = yaml.safe_load(source).get('refs', {}).get('A1_API_KEY')
if not key:
    raise SystemExit('A1_API_KEY is not configured in the DSH credential store')
environment = os.environ.copy()
environment['A1_API_KEY'] = key
os.execve('/usr/bin/sandbox-exec', [
    'sandbox-exec', '-f', sys.argv[1], 'dsh', '--profile', 'superagent', '--no-open'
], environment)
PY
