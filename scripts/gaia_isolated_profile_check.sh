#!/bin/sh
set -eu
RUN_ROOT=${1:?run root required}
PORT=${2:?port required}
SERVICE_URL="http://127.0.0.1:$PORT"
python3 - "$SERVICE_URL" "$RUN_ROOT" <<'PY'
import json, sys, urllib.request
url, root = sys.argv[1:]
def get(path):
    req = urllib.request.Request(url + '/orchestrator/api/' + path, headers={'x-orchestrator-request':'1'})
    with urllib.request.urlopen(req, timeout=5) as r: return json.load(r)
state = get('state')
settings = get('settings')
print(json.dumps({
    'setup_problem': state.get('setup', {}).get('problem'),
    'paused': state.get('paused'),
    'state_dir': state.get('setup', {}).get('machine', {}).get('stateDir'),
    'max_workers': settings.get('blocks', {}).get('executor', {}).get('value', {}).get('maxWorkers'),
    'benchmark_enabled': next((x.get('value', {}).get('enabled') for x in settings.get('sources', []) if x.get('id') == 'benchmark'), None),
    'run_root': root,
}, ensure_ascii=False))
PY
