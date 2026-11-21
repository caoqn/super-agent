"""Run MAC-Bench with one prompt-driven manager and three workers.

The MAC-Bench checkout and its task environments remain external to this
repository.  Set ``MAC_BENCH_ROOT`` to that checkout and use the official
per-task budget table unless explicit overrides are supplied.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import argparse

from .agent import (PromptManager3, MANAGER_SYSTEM, WORKER_SYSTEM, WORKERS,
                    DELEGATE, MAX_OUTPUT_TOKENS)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _positive_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f'{name} must be positive')
    return value


ROOT = Path(os.environ.get('MAC_BENCH_ROOT', '/mnt/macbench')).expanduser().resolve()
SPLIT = os.environ.get('MAC_BENCH_SPLIT', 'test').strip().lower()
if SPLIT != 'test':
    raise ValueError('The manager3 MAC-Bench baseline is configured for the 48-task test split')
EXPECTED_TASKS = 48
OUT = Path(os.environ.get(
    'MAC_BENCH_OUT_DIR',
    str(REPO_ROOT / 'results/manager3/macbench' / ('test48' if SPLIT == 'test' else SPLIT)),
)).expanduser().resolve()
MODEL_TIMEOUT = float(os.environ.get('MAC_BENCH_LLM_TIMEOUT_SECONDS', '180'))
MODEL_RETRIES = int(os.environ.get('MAC_BENCH_LLM_MAX_RETRIES', '0'))
MAX_MODEL_CALLS = _positive_int('MAC_BENCH_MAX_MODEL_CALLS', None)
MAX_TOOL_CALLS = _positive_int('MAC_BENCH_MAX_TOOL_CALLS', None)
MAX_TOKENS = _positive_int('MAC_BENCH_MAX_TOKENS', None)
MAX_WALL_SECONDS = _positive_int('MAC_BENCH_TASK_TIMEOUT_S', None)


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def model_config():
    from mac_bench.runtime.llm.config import LLMConfig
    return LLMConfig.from_mapping({
        'api_key_env': 'MAC_BENCH_LLM_API_KEY',
        'timeout_seconds': MODEL_TIMEOUT,
        'max_retries': MODEL_RETRIES,
        'thinking': os.environ.get('MAC_BENCH_LLM_THINKING', 'disabled'),
    }, require_api_key=True)


def load_budgets() -> dict:
    path = ROOT / 'configs/evaluation/task_budgets.json'
    try:
        budgets = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f'Cannot load official MAC-Bench budget table: {path}') from exc
    if not isinstance(budgets, dict):
        raise ValueError(f'Invalid budget table: {path}')
    return budgets


def task_budget(task_id: str, budgets: dict) -> dict[str, int]:
    try:
        row = dict(budgets[task_id])
    except KeyError as exc:
        raise KeyError(f'MAC-Bench budget missing for task {task_id}') from exc
    required = ('max_model_calls', 'max_tool_calls', 'max_env_actions', 'max_tokens', 'max_seconds')
    if any(int(row.get(key, 0)) <= 0 for key in required):
        raise ValueError(f'Invalid budget for {task_id}: {row}')
    overrides = {
        'max_model_calls': MAX_MODEL_CALLS,
        'max_tool_calls': MAX_TOOL_CALLS,
        'max_tokens': MAX_TOKENS,
        'max_seconds': MAX_WALL_SECONDS,
    }
    row.update({key: value for key, value in overrides.items() if value is not None})
    return {key: int(row[key]) for key in required}


def agent_usage(task_out):
    usage = {agent: {'model_attempts': 0, 'completed_calls': 0, 'input_tokens': 0,
                     'output_tokens': 0, 'model_seconds': 0.0}
             for agent in ('manager', *WORKERS)}
    for path in sorted((task_out / 'episode/model_calls').glob('*.json')):
        call = json.loads(path.read_text())
        agent = call.get('identity', {}).get('agent_id')
        if agent not in usage:
            raise RuntimeError('Missing or unexpected agent attribution')
        row = usage[agent]
        row['model_attempts'] += 1
        row['completed_calls'] += call.get('status') == 'completed'
        row['input_tokens'] += int(call.get('usage', {}).get('prompt_tokens', 0))
        row['output_tokens'] += int(call.get('usage', {}).get('completion_tokens', 0))
        row['model_seconds'] += float(call.get('elapsed_seconds', 0))
    return usage


def run_one(task_id):
    from mac_bench.evaluation.tasks.catalog import load_catalog, open_task
    from mac_bench.runtime.llm.client import LiteLLMChatClient
    from mac_bench.evaluation.episodes.runner import run_episode
    from mac_bench.runtime.agents.workflow import AgentBudget
    record = load_catalog(ROOT / 'data/tasks.json', task_ids=[task_id])[0]
    if record.split != SPLIT:
        raise ValueError(f'Task {task_id} belongs to split {record.split!r}, expected {SPLIT!r}')
    b = task_budget(task_id, load_budgets())
    budget = AgentBudget(max_model_calls=b['max_model_calls'], max_tool_calls=b['max_tool_calls'],
                         max_env_actions=b['max_env_actions'], max_tokens=b['max_tokens'],
                         max_wall_seconds=b['max_seconds'])
    task_out = OUT / 'tasks' / task_id
    task_out.mkdir(parents=True, exist_ok=False)
    save(task_out / 'run_config.json', {'max_output_tokens_per_call': MAX_OUTPUT_TOKENS,
                                       'agents': ['manager', *WORKERS], 'split': SPLIT,
                                       'budget': b, 'started_at': now()})
    client = LiteLLMChatClient(model_config())
    try:
        result = run_episode(open_task(record, task_out / 'runtime'), PromptManager3(), client,
                             output=task_out / 'episode', budget=budget, close=True, phase=SPLIT)
        # Save the terminal outcome even if auxiliary statistics fail.
        save(task_out / 'summary.json', result)
        usage = agent_usage(task_out)
        save(task_out / 'agent_usage.json', usage)
        if (sum(row['input_tokens'] for row in usage.values()) != result.get('input_tokens', 0)
                or sum(row['output_tokens'] for row in usage.values()) != result.get('output_tokens', 0)):
            raise RuntimeError('Per-agent usage does not match episode accounting')
    finally:
        client.close()
    return 0 if result.get('status') == 'completed' and result.get('closed') else 1


def manifest(records, budgets):
    connection = model_config()
    source = Path(__file__).resolve().parent
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
              for name in ('runner.py', 'agent.py')}
    runtime_files = ('evaluation/episodes/runner.py', 'evaluation/episodes/budget.py',
                     'evaluation/episodes/recording.py', 'runtime/llm/client.py',
                     'runtime/llm/gateway.py')
    runtime_hashes = {name: hashlib.sha256((ROOT / 'src/mac_bench' / name).read_bytes()).hexdigest()
                      for name in runtime_files}
    return {'method': PromptManager3.name, 'manager_prompt': MANAGER_SYSTEM,
            'worker_prompt': WORKER_SYSTEM, 'delegation_tool': DELEGATE,
            'agents': ['manager', *WORKERS], 'model': connection.model,
            'temperature': 0.0, 'thinking': connection.thinking_mode,
            'timeout_seconds': connection.timeout_seconds,
            'provider_max_retries': connection.max_retries,
            'recorded_gateway_transport_attempts': 1 + connection.max_retries,
            'max_output_tokens_per_call': MAX_OUTPUT_TOKENS,
            'split': SPLIT, 'repetitions': 1, 'episode_concurrency': 1,
            'worker_concurrency': 3, 'environment_tool_calls': 'serialized',
            'training': False, 'trained_skills': False,
            'sandbox': os.environ.get('MAC_BENCH_SANDBOX_BACKEND', 'docker'),
            'supersedes_pilot': None,
            'superseded_pilot_included_in_scores': False,
            'history_truncation': False, 'history_summarization': False,
            'budget_scope': 'one official shared budget across all four agents',
            'control_tool_accounting': 'event count; environment tools use official tool budget',
            'source_sha256': hashes, 'runtime_sha256': runtime_hashes,
            'budget_source': 'MAC-Bench task_budgets.json',
            'tasks': {r.task_id: {'revision': r.world_revision,
                                  'budget': task_budget(r.task_id, budgets)} for r in records}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', dest='task_id', help=argparse.SUPPRESS)
    parser.add_argument('--preflight', action='store_true',
                        help='Verify the selected MAC-Bench split without running tasks')
    args = parser.parse_args(argv)
    if args.task_id:
        return run_one(args.task_id)
    if args.preflight:
        os.environ['MAC_BENCH_PREFLIGHT'] = '1'
    from mac_bench.evaluation.tasks.catalog import load_catalog
    records = [r for r in load_catalog(ROOT / 'data/tasks.json') if r.split == SPLIT]
    budgets = load_budgets()
    expected = EXPECTED_TASKS
    if len(records) != expected or len({r.task_id for r in records}) != expected:
        raise RuntimeError(f'Expected exactly {expected} distinct {SPLIT} tasks')
    OUT.mkdir(parents=True, exist_ok=True)
    config = manifest(records, budgets)
    path = OUT / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError('Existing output belongs to a different configuration')
    save(path, config)
    env = os.environ.copy()
    src = str(ROOT / 'src')
    env['PYTHONPATH'] = src + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    subprocess.run([sys.executable, '-m', 'mac_bench', 'data', 'verify', '--split', SPLIT],
                   cwd=ROOT, env=env, check=True)
    if os.environ.get('MAC_BENCH_PREFLIGHT') == '1':
        progress = OUT / 'progress.json'
        save(progress, {'status': 'preflight_ok', 'recorded': 0, 'completed': 0,
                        'failed': 0, 'split': SPLIT, 'updated_at': now()})
        print(f'[manager3] preflight passed for {len(records)} {SPLIT} tasks', flush=True)
        return
    rows = []

    def counts():
        completed = sum(r.get('status') == 'completed' and bool(r.get('closed')) for r in rows)
        return {'recorded': len(rows), 'completed': completed, 'failed': len(rows) - completed}

    def progress(status, **extra):
        save(OUT / 'progress.json', {'status': status, **counts(), **extra, 'updated_at': now()})

    def checkpoint():
        save(OUT / 'results.json', {'rows': rows, 'updated_at': now()})

    for record in records:
        task_id = record.task_id
        task_dir = OUT / 'tasks' / task_id
        summary = task_dir / 'summary.json'
        if task_dir.exists():
            if summary.exists():
                old = json.loads(summary.read_text())
                if old.get('closed'):
                    rows.append(old)
                    checkpoint()
                    print(f'[manager3] retaining recorded outcome task={task_id} '
                          f'status={old.get("status")} gc={old.get("gc")}', flush=True)
                    continue
            progress('needs_attention', current_task=task_id)
            raise RuntimeError(f'Incomplete output for {task_id}; inspect before resuming')
        if shutil.disk_usage(OUT).free < 80 * 1024**3:
            progress('needs_attention', reason='disk_space')
            raise RuntimeError('Less than 80 GiB disk space available')
        progress('running', current_task=task_id)
        print(f'[manager3] starting task={task_id} recorded={len(rows)}/{expected}', flush=True)
        proc = subprocess.run([sys.executable, '-m', 'methods.manager3.macbench.runner',
                               '--task', task_id], cwd=REPO_ROOT)
        row = json.loads(summary.read_text()) if summary.exists() else None
        if not row or not row.get('closed'):
            progress('needs_attention', current_task=task_id, exit_code=proc.returncode)
            raise RuntimeError(f'Task {task_id} has no closed terminal result; inspect before resuming')
        rows.append(row)
        checkpoint()
        # Stop for accounting bugs even if the task itself closed successfully.
        if proc.returncode and row.get('status') == 'completed':
            progress('needs_attention', current_task=task_id, reason='auxiliary_statistics')
            raise RuntimeError('Task closed but auxiliary statistics failed')
        print(f'[manager3] finished task={task_id} status={row.get("status")} '
              f'gc={row.get("gc")} success={row.get("task_success")}', flush=True)
    progress('completed_with_errors' if counts()['failed'] else 'completed')
    print(f'[manager3] finished all {expected} {SPLIT} tasks', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
