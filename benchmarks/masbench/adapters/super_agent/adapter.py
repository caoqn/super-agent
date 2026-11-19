#!/usr/bin/env python3
"""MASBench native episode loops, with every inference routed through SuperAgent.

The upstream repository is imported, never executed with its API credentials.
Each episode runs in its own process; worker tickets get only that turn's view.
"""
import argparse
import asyncio
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REPO = ROOT
sys.path.insert(0, str(ROOT))
from benchmarks.gaia.adapters.super_agent import adapter as transport
from benchmarks.masbench.scoring import reasoning_f1, game_score
from benchmarks.common.session_usage import read_session_usage


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


class EpisodeFailure(RuntimeError):
    pass


class Bridge:
    def __init__(self, args, out):
        self.args, self.out = args, out
        self.deadline = time.monotonic() + args.timeout_seconds
        self.lock = threading.Lock()
        self.counter = 0
        self.failed = False
        self.worker_usage_by_session = {}
        (out / 'payloads').mkdir(parents=True, exist_ok=True)
        self.client = transport.Client(args.service_url)
        self.calls = out / 'calls.jsonl'
        # run_item owns the full intake -> router -> ticket -> report -> cleanup lifecycle.
        transport.build_task = lambda item, workspace, attachment=None: item['prompt']
        transport.extract_final_answer = self.extract
        original = self.client.call
        def audited_call(method, path, body=None):
            result = original(method, path, body)
            if method == 'POST' and path == 'benchmark/tasks':
                self.append({'event': 'submitted', 'at': utc(), 'task_id': body['task_id'], **result})
            return result
        self.client.call = audited_call

    @staticmethod
    def extract(summary):
        match = re.search(r'MASBENCH_RESPONSE_BEGIN\s*\n(.*?)\nMASBENCH_RESPONSE_END', summary, re.S)
        return match.group(1).strip() if match else ''

    def append(self, row):
        with self.lock:
            with self.calls.open('a') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()

    def generate(self, messages, role='agent'):
        for attempt in range(2):
            try:
                return self.generate_once(messages, role)
            except EpisodeFailure:
                if attempt or self.failed or time.monotonic() >= self.deadline:
                    raise
                self.append({'event': 'retry', 'at': utc(), 'role': str(role)})

    def generate_once(self, messages, role='agent'):
        with self.lock:
            self.counter += 1
            number = self.counter
        if self.failed or self.deadline - time.monotonic() < 15:
            raise EpisodeFailure('episode_timeout_or_previous_failure')
        task_id = f'{self.args.category}-{self.args.item_id.replace(".", "_")}-turn{number}'
        payload_path = self.out / 'payloads' / f'{task_id}.json'
        response_path = self.out / 'payloads' / f'{task_id}.response'
        instructions = (
            'Generate one next response as the specified role; do not simulate other roles. '
            'Read only this turn input, not other benchmark, result, or ticket data. '
            'Do not create a GitHub issue or PR. Write the exact complete response as UTF-8 '
            f'text to {response_path}, preserving tags and newlines. Then report ready. '
            'Also repeat the response verbatim in the report summary between '
            'MASBENCH_RESPONSE_BEGIN and MASBENCH_RESPONSE_END on separate lines.')
        payload_path.write_text(json.dumps({'instructions': instructions, 'role': str(role),
                                           'messages': messages}, ensure_ascii=False), encoding='utf-8')
        # Keep the source trigger shorter than the worker's inline trigger limit;
        # the full task payload is an explicit, per-ticket local input file.
        prompt = (f'MASBench inference turn. Read the authorized input file {payload_path} '
                  'and execute its instructions using its role and messages. '
                  'Save the exact response to the specified response file, then report ready.')
        if len(prompt) > 600:
            raise EpisodeFailure('trigger_path_exceeds_inline_limit')
        self.append({'event': 'request', 'task_id': task_id, 'at': utc(), 'role': str(role), 'messages': messages})
        item = {'task_id': task_id, '_level': 0, 'Final answer': '', 'prompt': prompt}
        record = transport.run_item(self.client, item, self.args.run_id, self.out / 'scratch',
                                    self.out, max(.1, self.deadline-time.monotonic()), 2)
        usage = read_session_usage(record.get('session_id'))
        if usage is not None:
            record.update(input_tokens=usage['inputTokens'], output_tokens=usage['outputTokens'],
                          total_tokens_including_cache=usage['totalTokens'],
                          cache_read_tokens=usage['cacheReadTokens'],
                          cache_write_tokens=usage['cacheWriteTokens'],
                          usage_events=usage['usage_events'], token_measurement='provider_reported_worker_session')
            self.worker_usage_by_session[record['session_id']] = usage
        ticket_state = self.client.call('GET', 'tickets/' + quote(record['ticket_id'], safe='')) if record.get('ticket_id') else {}
        # Some workers use Path.with_suffix() and save .response rather than .response.txt.
        # Only accept this turn's exact known filename, never a glob or another turn's output.
        if not response_path.exists() and response_path.with_suffix('.response.txt').is_file():
            response_path = response_path.with_suffix('.response.txt')
        if record.get('worker_state') == 'ready' and (ticket_state.get('ended') or {}).get('outcome') == 'completed' \
                and response_path.is_file():
            exact = response_path.read_text(encoding='utf-8').strip()
            if exact:
                record['prediction'] = exact
                record['status'] = 'completed'
                record['response_source'] = 'worker_written_response_file'
            else:
                record['status'] = 'invalid_answer'
        record.pop('correct', None)
        record['benchmark'] = 'MASBench selected 90'
        self.append({'event': 'response', 'at': utc(), **record})
        if record['status'] != 'completed' or record.get('cleanup_failed'):
            self.failed = bool(record.get('cleanup_failed'))
            raise EpisodeFailure(record['status'])
        return record['prediction']

    def json_generate(self, prompt, **meta):
        for attempt in range(2):
            request = prompt if not attempt else (
                prompt + '\n\nReturn only one valid JSON object matching the requested schema. '
                'Correct the previous invalid response: ' + text[:3000])
            text = self.generate([{'role': 'user', 'content': request}], meta.get('phase', 'json'))
            try:
                return json.loads(text)
            except ValueError:
                start, end = text.find('{'), text.rfind('}')
                if start >= 0 and end > start:
                    try:
                        return json.loads(text[start:end+1])
                    except ValueError:
                        pass
        raise EpisodeFailure('invalid_json_after_retry')


def reasoning(args, bridge, out):
    sys.path.insert(0, str(args.upstream / 'Reasoning'))
    from src.agents import BaseAgent
    from src.conversation import run_conversation
    from src.data_loader_v4_mas import V4MasDataLoader
    from src.prompts import QA_INFO_EXCHANGE_PROMPT
    loader = V4MasDataLoader(str(args.data_dir / 'Reasoning/reasoning_subset.json'), mode='context')
    index = next(i for i, row in enumerate(loader._items) if str(row['id']) == args.item_id)
    sample = loader.sample(index)
    class TicketAgent(BaseAgent):
        def step(self):
            messages = [{'role': 'system', 'content': self.system_prompt}]
            messages += [{'role': 'assistant' if m.content.startswith(self.name+':') else 'user',
                          'content': m.content} for m in self.memory]
            text = bridge.generate(messages, self.name)
            if not text.startswith(self.name + ':'):
                text = self.name + ': ' + text
            self.add_memory(text)
            return text
    agents = [TicketAgent('Alice'), TicketAgent('Bob')]
    for i, agent in enumerate(agents):
        agent.init_system_prompt(QA_INFO_EXCHANGE_PROMPT, dict(
            name=agent.name, partner=agents[1-i].name, question=sample.question,
            information=[sample.context_first, sample.context_second][i]))
    transcript, answer = run_conversation(agents, max_rounds=8)
    write_json(out / 'trajectory.json', {'transcript': transcript, 'prediction': answer})
    return dict(prediction=answer, reference=sample.answer, score=reasoning_f1(sample.answer, answer or ''),
                rounds=len(transcript), status='completed' if answer else 'invalid_answer',
                settings={'mode': 'context', 'conversation_style': 'base', 'max_turns': 8})


def scheduling(args, bridge, out):
    spec = importlib.util.spec_from_file_location('masbench_scheduling', args.upstream / 'Scheduling/simulate_meeting_gpt.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.model_name = 'superagent-configured-model'
    module.ensure_logs_dir = lambda _: str(out / 'native_logs')
    (out / 'native_logs').mkdir()
    module.call_llm = lambda prompt, **meta: bridge.generate([{'role': 'user', 'content': prompt}], meta.get('agent_id', meta.get('phase', 'agent')))
    for name in ('call_llm_stream', 'call_llm_json', 'call_llm_json_stream'):
        setattr(module, name, bridge.json_generate)
    episode = module.Orchestrator(str(args.data_dir / 'Scheduling/scenarios' / args.item_id), style='protocol', use_memory=True)
    episode.run_task(duration_minutes=20, max_rounds=5, conversion=1, mode='parallel')
    meta = json.loads(Path(episode.meta_file_path).read_text())
    # Native counters describe direct API calls, which this bridge deliberately replaces.
    meta['token_usage'] = None
    meta['token_usage_unavailable_reason'] = 'SuperAgent management API does not expose aggregate model usage'
    write_json(Path(episode.meta_file_path), meta)
    evaluation = meta.get('evaluation') or {}
    return dict(status='completed', score=float(evaluation.get('status') == 'ok'),
                prediction=evaluation.get('normalized_result'), native_evaluation=evaluation,
                rounds=meta['rounds_used'], settings={'style': 'protocol', 'use_memory': True,
                                                     'mode': 'parallel', 'max_rounds': 5})


def game(args, bridge, out):
    sys.path.insert(0, str(args.upstream / 'Game'))
    import framework.framework as factory
    from framework.llm import Chat
    class TicketBrain(Chat):
        async def generate(self, content, sampling_override=None):
            self.msg.append({'role': 'user', 'content': content})
            messages = [self.sys_prompt] + (self.msg if self.memory else self.msg[-1:])
            result = await asyncio.to_thread(bridge.generate, messages, 'game-agent')
            self.msg.append({'role': 'assistant', 'content': result})
            return result
    factory.Chat = TicketBrain
    from masbench.framework import MASFramework
    map_path = args.data_dir / 'Game/maps' / args.item_id
    data = json.loads(map_path.read_text())
    cfg = data['map']
    acfg = cfg.get('agent_config', {})
    framework = MASFramework('masbench-' + Path(args.item_id).stem)
    # Match upstream eval.py's released base preset and 50-round budget.
    framework.run_task(MASFramework.model_config('superagent-configured-model', 'bridge-no-api-key',
                                                'http://127.0.0.1', provider='SUPERAGENT'),
        'Foraging', log_dir=str(out / 'native_logs'),
        num_agents=max(len(cfg.get('agents', [])), acfg.get('agent_number', 0)),
        max_round=args.game_rounds, width=cfg['width'], height=cfg['height'], seed=42,
        view_size=acfg.get('view_size', 9), communication_size=acfg.get('communication_size', 15),
        custom_map=data, custom_map_path=str(map_path), agent_view_memory=1, agent_history_memory=0,
        agent_carrying_capacity=acfg.get('item_number', 4), mechanism_preset='base')
    env = framework.framework.env
    totals = env.tasks[env.task].count_items_on_tables(env)
    # The released game maps set targets in the task configuration. Preserve
    # those targets when calculating the per-resource score.
    target = data.get('task', {}).get('scoring', {})
    write_json(out / 'final_tables.json', {'items_on_tables': totals,
        'final_table_items': {key: min(int(totals.get(key, 0)), int(value)) for key, value in target.items()}})
    metric = game_score(map_path, out / 'final_tables.json')
    # Mark native placeholder counters unavailable instead of presenting zero usage as measured.
    meta_path = out / 'native_logs/meta_log.json'
    if meta_path.exists():
        native = json.loads(meta_path.read_text())
        for entry in native.values():
            if isinstance(entry, dict) and 'finish' in entry:
                entry['finish']['tokens'] = None
                entry['finish']['token_usage_unavailable_reason'] = 'SuperAgent usage not exposed by management API'
        write_json(meta_path, native)
    return dict(status='completed', score=metric['earned']/metric['maximum'], metric=metric,
                rounds=env.round, settings={'mechanism_preset': 'base', 'max_rounds': args.game_rounds,
                                           'seed': 42, 'messages_enabled': False})


def episode_main(args):
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    bridge = Bridge(args, out)
    start = time.monotonic()
    result = dict(category=args.category, task_id=args.item_id, run_id=args.run_id,
                  started_at=utc(), input_tokens=None, output_tokens=None, cost_usd=None,
                  usage_scope='unavailable: requires SuperAgent worker + router usage extraction')
    try:
        result.update(globals()[args.category](args, bridge, out))
    except Exception as error:
        result.update(status='failed', score=None, error_type=type(error).__name__)
        if isinstance(error, EpisodeFailure):
            result['status'] = str(error)
        # No provider exception strings or tracebacks, which can contain credentials.
    usage_values = list(bridge.worker_usage_by_session.values())
    usage_keys = ('inputTokens', 'outputTokens', 'totalTokens', 'cacheReadTokens', 'cacheWriteTokens', 'usage_events')
    usage_totals = {key: sum(v[key] for v in usage_values) for key in usage_keys}
    result.update(ended_at=utc(), latency_seconds=round(time.monotonic()-start, 3), inference_calls=bridge.counter,
                  input_tokens=usage_totals['inputTokens'] if usage_values else None,
                  output_tokens=usage_totals['outputTokens'] if usage_values else None,
                  total_tokens_including_cache=usage_totals['totalTokens'] if usage_values else None,
                  cache_read_tokens=usage_totals['cacheReadTokens'] if usage_values else None,
                  cache_write_tokens=usage_totals['cacheWriteTokens'] if usage_values else None,
                  usage_events=usage_totals['usage_events'] if usage_values else None,
                  usage_scope='provider-reported per-ticket worker sessions; orchestrator usage excluded')
    write_json(out / 'result.json', result)


def cleanup(client, episode, run_id):
    """Stop only tickets belonging to this episode; unresolved routing stops the run."""
    path = episode / 'calls.jsonl'
    if not path.exists():
        return True
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    tasks = {r['task_id'] for r in rows if r['event'] == 'request'}
    responses = {r['task_id']: r for r in rows if r['event'] == 'response'}
    safe = True
    for task in tasks:
        if task in responses and not responses[task].get('cleanup_failed'):
            continue
        try:
            thread = client.call('GET', 'threads/'+quote(f'benchmark:{run_id}/{task}', safe=''))
            ids = {t['id'] for t in thread.get('tickets', [])}
            terminal = False
            for event in thread.get('timeline', []):
                d = event.get('disposition', {}).get('decision') or {}
                if d.get('ticketId'):
                    ids.add(d['ticketId'])
                terminal |= d.get('outcome') in ('ignore', 'unavailable')
            if not ids and not terminal:
                safe = False
            for ticket in ids:
                current = client.call('GET', 'tickets/'+quote(ticket, safe=''))
                if current.get('state') != 'ended':
                    client.call('POST', 'tickets/'+quote(ticket, safe='')+'/stop')
                safe &= client.call('GET', 'tickets/'+quote(ticket, safe='')).get('state') == 'ended'
        except Exception:
            safe = False
    return safe


def run_suite(args):
    from benchmarks.masbench.validate_data import validate
    validate(args.data_dir)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    client = transport.Client(args.service_url)
    state = client.call('GET', 'state')
    if state['setup']['problem'] or state['paused']:
        raise RuntimeError('SuperAgent unavailable')
    manifest = json.loads((args.data_dir / 'selection_manifest.json').read_text())
    # Interleave categories so each category's score develops throughout the run.
    tasks = [(category, str(manifest[category]['ids'][i])) for i in range(30)
             for category in ('reasoning', 'scheduling', 'game')]
    launch_path = out / 'launch.json'
    previous_launch = json.loads(launch_path.read_text()) if args.resume and launch_path.exists() else {}
    if previous_launch:
        args.run_id = previous_launch['run_id']
    write_json(launch_path, dict(run_id=args.run_id, started_at=previous_launch.get('started_at', utc()), resumed_at=utc(), pid=os.getpid(),
        total=90, episode_concurrency=1, timeout_seconds=args.timeout_seconds,
        native_settings={'reasoning': 'context/base/8 turns', 'scheduling': 'protocol/memory/5 rounds',
                         'game': 'base/50 rounds/seed42'},
        failure_policy='record individual failures; retry pending cleanup between episodes and continue',
        service_url=args.service_url))
    start = time.monotonic()
    existing = {}
    if args.resume and (out / 'results.jsonl').exists():
        for line in (out / 'results.jsonl').read_text().splitlines():
            row = json.loads(line)
            existing[(row['category'], row['task_id'])] = row
    canonical = dict(existing)
    results = [existing[t] for t in tasks if t in existing]
    pending_path = out / 'pending_cleanup.json'
    pending = json.loads(pending_path.read_text()) if pending_path.exists() else []
    # Recover delayed routing from older attempts, including the one that stopped a prior run.
    for result_path in out.glob('*/result.json'):
        prior = json.loads(result_path.read_text())
        entry = {'episode': str(result_path.parent), 'run_id': prior.get('run_id', args.run_id)}
        if entry not in pending:
            if not cleanup(client, result_path.parent, entry['run_id']):
                pending.append(entry)
    def sweep_pending():
        pending[:] = [entry for entry in pending
                      if not cleanup(client, Path(entry['episode']), entry['run_id'])]
        write_json(pending_path, pending)
    sweep_pending()
    for index, (category, item) in enumerate(tasks):
        if (category, item) in existing and (
                not args.retry_failed or existing[(category, item)]['status'] == 'completed'):
            continue
        sweep_pending()
        episode = out / f'{category}-{Path(item).stem}'
        attempt = 1
        while episode.exists():
            attempt += 1
            episode = out / f'{category}-{Path(item).stem}-attempt{attempt}'
        episode.mkdir()
        episode_run_id = f'{args.run_id}-e{index+1}-a{attempt}'
        command = [sys.executable, str(Path(__file__).resolve()), '--episode', '--category', category,
                   '--item-id', item, '--run-id', episode_run_id, '--timeout-seconds', str(args.timeout_seconds),
                   '--output-dir', str(episode), '--service-url', args.service_url,
                   '--upstream', str(args.upstream), '--data-dir', str(args.data_dir), '--game-rounds', str(args.game_rounds)]
        write_json(out / 'progress.json', dict(processed=len(results), valid=sum(r['status']=='completed' for r in results),
                                               total=90, active=f'{category}/{item}', phase='full', updated_at=utc()))
        print(f'START [{index+1}/90] {category}/{item}', flush=True)
        with (episode / 'episode.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=log)
            try:
                process.wait(timeout=args.timeout_seconds + 25)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        safe = cleanup(client, episode, episode_run_id)
        if not safe:
            pending.append({'episode': str(episode), 'run_id': episode_run_id})
            write_json(pending_path, pending)
        result_path = episode / 'result.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else dict(
            category=category, task_id=item, status='timeout_or_process_failure', score=None,
            input_tokens=None, output_tokens=None, cost_usd=None)
        result['cleanup_verified'] = safe
        canonical[(category, item)] = result
        results = [canonical[t] for t in tasks if t in canonical]
        # Preserve old attempts in their episode directories; canonical rows contain one result per item.
        (out / 'results.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in results))
        print(f'END [{index+1}/90] {category}/{item} {result["status"]} score={result.get("score")}', flush=True)
        summary = dict(processed=len(results), valid=sum(r['status']=='completed' for r in results),
                       failed=sum(r['status']!='completed' for r in results), total=90, updated_at=utc(),
                       wall_seconds=round(time.monotonic()-start, 3),
                       categories={c: dict(processed=sum(r['category']==c for r in results),
                           valid=sum(r['category']==c and r['status']=='completed' for r in results),
                           mean_score=(sum(r['score'] for r in results if r['category']==c and r['status']=='completed' and r.get('score') is not None)/
                           max(1,sum(r['category']==c and r['status']=='completed' and r.get('score') is not None for r in results))
                           if any(r['category']==c and r['status']=='completed' and r.get('score') is not None for r in results) else None))
                           for c in ('reasoning','scheduling','game')})
        write_json(out / 'summary.json', summary)
        if not safe:
            print('WARN: delayed ticket cleanup queued for retry; continuing next item.', flush=True)
    sweep_pending()
    write_json(out / 'progress.json', dict(processed=len(results),total=90,phase='completed',active=None,
        valid=sum(r['status']=='completed' for r in results),pending_cleanup=len(pending),updated_at=utc()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', action='store_true')
    parser.add_argument('--category', choices=['reasoning','scheduling','game'])
    parser.add_argument('--item-id')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, default=REPO / 'MASBench-main')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'benchmarks/masbench/data')
    parser.add_argument('--service-url', default='http://127.0.0.1:3082')
    parser.add_argument('--run-id', default='masbench-'+uuid.uuid4().hex[:12])
    parser.add_argument('--timeout-seconds', type=float, default=1200)
    parser.add_argument('--game-rounds', type=int, default=50)
    parser.add_argument('--resume', action='store_true', help='Continue unattempted items, keeping all previous results')
    parser.add_argument('--retry-failed', action='store_true', help='With --resume, retry previously unsuccessful items too')
    args = parser.parse_args()
    args.upstream, args.data_dir = args.upstream.resolve(), args.data_dir.resolve()
    if args.timeout_seconds <= 0 or not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}',args.run_id):
        parser.error('Invalid timeout or run ID')
    if args.episode and (not args.category or not args.item_id):
        parser.error('--episode requires --category and --item-id')
    (episode_main if args.episode else run_suite)(args)


if __name__ == '__main__':
    main()
