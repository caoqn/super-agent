#!/usr/bin/env python3
"""Evaluation-only EvoTeam GAIA adapter for the actual SuperAgent service."""
import argparse
import json
import re
import shutil
import sys
import tempfile
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from benchmarks.gaia.accuracy import correct


def load_dataset(data_dir):
    expected = [x['task_id'] for x in json.loads(
        (ROOT / 'benchmarks/gaia/manifest.json').read_text())['tasks']]
    items = []
    for level in (1, 2, 3):
        rows = json.loads((data_dir / f'level_{level}_test_100.json').read_text())
        items.extend(dict(row, _level=level) for row in rows)
    if len(expected) != 100 or len(set(expected)) != 100:
        raise ValueError('Manifest must contain 100 distinct task IDs')
    if [x['task_id'] for x in items] != expected:
        raise ValueError('Test IDs/order differ from frozen EvoTeam test_100 manifest')
    for item in items:
        if not item.get('Question') or item.get('Final answer') is None:
            raise ValueError(f"Missing question/reference: {item['task_id']}")
        if item.get('file_name'):
            attachment_path(data_dir, item)
    return items


def attachment_path(data_dir, item):
    name = Path(item['file_name'])
    if name.is_absolute() or '..' in name.parts:
        raise ValueError('Attachment must have a safe relative filename')
    root = (data_dir / 'val_files').resolve()
    for source in (root / name, root / item['task_id'] / name):
        source = source.resolve()
        if source.is_relative_to(root) and source.is_file():
            return source
    raise FileNotFoundError(f"Attachment missing: {item['task_id']} / {name}")


def build_task(item, workspace, attachment=None):
    text = (
        'Solve this GAIA benchmark question in the assigned SuperAgent worker ticket. '
        'Preserve the full question and any attachment path when routing this request. '
        'The assigned worker should use the question and provided attachment to produce the final answer. '
        'This is an evaluation task with a specified answer format. Do not create a GitHub issue or PR.\n\n'
        f"## Question (Level {item['_level']})\n{item['Question']}\n\n"
        'Keep the SuperAgent ticket execution directory as your working directory. '
        f'This additional per-question directory contains evaluation inputs and may hold scratch files: {workspace}\n'
    )
    if attachment:
        text += f'Attached file (absolute local path): {attachment}\nUse appropriate tools to analyze it.\n'
    return text + (
        '\n## Answer format\nReturn a number, a few words, or a comma-separated list, as requested. '
        'For numbers omit units and thousands separators unless requested. For strings omit articles '
        'and avoid abbreviations unless requested. Do not ask a person for help. '
        'Do not read other benchmark questions, references or result files. '
        'When finished, call orchestrator_report with state="ready" and put the final answer '
        'in its summary on a single line: FINAL ANSWER: <answer>. '
        'The summary must contain the answer itself, not just a path to a file.'
    )


class Client:
    def __init__(self, url):
        parsed = urlparse(url)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost') or parsed.username:
            raise ValueError('Use a local HTTP SuperAgent service URL')
        self.base = url.rstrip('/') + '/orchestrator/api/'

    def call(self, method, path, body=None):
        request = Request(self.base + path, method=method,
                          data=None if body is None else json.dumps(body).encode(),
                          headers={'x-orchestrator-request': '1', 'Content-Type': 'application/json'})
        try:
            with urlopen(request, timeout=20) as response:
                return json.load(response)
        except HTTPError as error:
            # Server errors may contain credential-bearing upstream messages: do not echo their body.
            raise RuntimeError(f'SuperAgent HTTP {error.code} for {method} {path}') from None


def extract_final_answer(summary):
    """Extract only the worker's answer line from a ready report.

    The benchmark prompt asks for ``FINAL ANSWER:``, while the current
    SuperAgent worker reports often use the shorter ``Answer:`` form.  Keep
    the extraction line-oriented so the method/reasoning text is never used
    as the prediction.
    """
    # Workers sometimes add a short label (for example ``GAIA L3 ... answer =``)
    # or wrap the answer in Markdown bold.  Accept those answer-line variants,
    # while still ignoring the surrounding research narrative.
    lines = summary.splitlines()

    def value_after_marker(line, marker):
        match = re.search(
            rf'\b{marker}\b[ \t]*[:=][ \t]*(\S.*?)[ \t]*$', line, re.I)
        if not match:
            return ''
        value = match.group(1).strip()
        # When the worker uses **answer**, keep only the bold span.  This avoids
        # treating a trailing source citation as part of the prediction.
        bold = re.match(r'\*\*(.+?)\*\*', value)
        if bold:
            value = bold.group(1).strip()
        elif '**' in value:
            # The opening ``**`` can be before the marker, as in
            # ``**Final answer: value** (source...)``.
            value = value.split('**', 1)[0].strip()
        return value.strip('`').strip()

    for line in lines:
        value = value_after_marker(line, r'FINAL[ \t]+ANSWER')
        if value:
            return value
    for line in lines:
        value = value_after_marker(line, r'ANSWER')
        if value:
            return value
    return ''


def run_item(client, item, run_id, workspace_root, data_dir, timeout, poll):
    start = time.monotonic()
    record = dict(agent='Super Agent (benchmark-adapted)', benchmark='EvoTeam GAIA test_100',
                  run_id=run_id, task_id=item['task_id'], level=item['_level'], status='failed',
                  prediction='', reference=item['Final answer'], ticket_id=None, session_id=None,
                  latency_seconds=None, human_interventions=None, human_action_requested=False, cost_usd=None,
                  input_tokens=None, output_tokens=None, correct=False)
    ticket_id = None
    submitted = None
    ticket = None
    submission_attempted = False
    routing_terminal = False
    try:
        workspace = workspace_root / item['task_id']
        workspace.mkdir(parents=True, exist_ok=False)
        attachment = None
        if item.get('file_name'):
            attachment = workspace / item['file_name']
            attachment.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(attachment_path(data_dir, item), attachment)
        submission_attempted = True
        submitted = client.call('POST', 'benchmark/tasks', dict(
            run_id=run_id, task_id=item['task_id'], prompt=build_task(item, workspace, attachment)))
        record['trigger_id'] = submitted['trigger_id']
        record['thread'] = submitted['thread']
        if submitted['result'] == 'duplicate':
            # Never score a trigger from an earlier run.  In particular, do not
            # stop the old ticket while attempting cleanup for this item.
            routing_terminal = True
            raise RuntimeError('Benchmark task already exists; use a new run ID')
        if submitted['result'] == 'filtered':
            routing_terminal = True
            raise RuntimeError('Benchmark source filtered the task')
        deadline = start + timeout
        while time.monotonic() < deadline:
            if not ticket_id:
                thread = client.call('GET', 'threads/' + quote(submitted['thread'], safe=''))
                tickets = thread.get('tickets', [])
                ids = {x['id'] for x in tickets}
                for event in thread.get('timeline', []):
                    decision = event.get('disposition', {}).get('decision') or {}
                    if decision.get('ticketId'):
                        ids.add(decision['ticketId'])
                    if decision.get('outcome') in ('ignore', 'unavailable'):
                        routing_terminal = True
                        record['status'] = 'routing_' + decision['outcome']
                        return record
                if len(ids) > 1:
                    raise RuntimeError('Task routed to multiple tickets; manual audit required')
                ticket_id = next(iter(ids), None)
                record['ticket_id'] = ticket_id
            if ticket_id:
                ticket = client.call('GET', 'tickets/' + quote(ticket_id, safe=''))
                record['session_id'] = ticket.get('sessionId') or None
                assignments = ticket.get('assignments', [])
                report = (assignments[-1].get('report') or {}) if assignments else {}
                state = report.get('state')
                record['worker_state'] = state
                record['ticket_stage'] = ticket.get('stage')
                record['ticket_directory'] = ticket.get('directory')
                record['human_action_requested'] = bool(ticket.get('attention'))
                if state == 'ready' and (ticket.get('ended') or {}).get('outcome') == 'completed':
                    record['prediction'] = extract_final_answer(report.get('summary', ''))
                    record['status'] = 'completed' if record['prediction'] else 'invalid_answer'
                    record['correct'] = record['status'] == 'completed' and correct(record['prediction'], record['reference'])
                    return record
                if ticket.get('attention') or ticket.get('stage') == 'person':
                    action = ticket.get('action')
                    record['status'] = ('needs_human' if ticket.get('attention') else
                                        {'delivery': 'delivery_failed', 'review': 'needs_review',
                                         'blocked': 'blocked', 'failed': 'worker_failed'}.get(action, 'needs_human'))
                    return record
                if ticket.get('stage') == 'ended':
                    record['status'] = 'ended_without_answer'
                    return record
            time.sleep(min(poll, max(0, deadline - time.monotonic())))
        record['status'] = 'timeout'
        return record
    except (RuntimeError, URLError, TimeoutError, OSError, ValueError, KeyError) as error:
        record['error_type'] = type(error).__name__
        # Never serialize raw provider errors or HTTP response bodies.
        return record
    finally:
        # A submit can reach the service even if its HTTP response is lost. Recover
        # ticket IDs using the deterministic thread, including tickets discovered
        # after the last poll. Unresolved routing must stop the run, not silently
        # overlap later questions. The management API has no pending-Eval cancel.
        cleanup_ids = {ticket_id} if ticket_id else set()
        if submission_attempted and not routing_terminal:
            try:
                thread_key = f"benchmark:{run_id}/{item['task_id']}"
                thread = client.call('GET', 'threads/' + quote(thread_key, safe=''))
                cleanup_ids.update(x['id'] for x in thread.get('tickets', []))
                for event in thread.get('timeline', []):
                    decision = event.get('disposition', {}).get('decision') or {}
                    if decision.get('ticketId'):
                        cleanup_ids.add(decision['ticketId'])
                    if decision.get('outcome') in ('ignore', 'unavailable'):
                        routing_terminal = True
                if not cleanup_ids and not routing_terminal:
                    record['cleanup_failed'] = True
            except Exception:
                record['cleanup_failed'] = True
        # End outstanding benchmark work so timeouts/human handoffs do not leak into the next task.
        for cleanup_id in cleanup_ids:
            try:
                current = client.call('GET', 'tickets/' + quote(cleanup_id, safe=''))
                if current.get('state') != 'ended':
                    client.call('POST', 'tickets/' + quote(cleanup_id, safe='') + '/stop')
                    current = client.call('GET', 'tickets/' + quote(cleanup_id, safe=''))
                    if current.get('state') != 'ended':
                        record['cleanup_failed'] = True
            except Exception:
                record['cleanup_failed'] = True
        record['latency_seconds'] = round(time.monotonic() - start, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['test_100'], default='test_100')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'benchmarks/gaia/data')
    parser.add_argument('--service-url', default='http://127.0.0.1:3082')
    parser.add_argument('--max-items', type=int, choices=range(1, 101), metavar='1..100')
    parser.add_argument('--task-ids-file', type=Path,
                        help='Run only task IDs listed one per line after dataset validation')
    parser.add_argument('--level', type=int, choices=[1, 2, 3])
    parser.add_argument('--concurrency', type=int, choices=range(1, 9), default=1)
    parser.add_argument('--timeout-seconds', type=float, default=480)
    parser.add_argument('--poll-seconds', type=float, default=2)
    parser.add_argument('--run-id', default='gaia-' + uuid.uuid4().hex[:16])
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / ('gaia-' + uuid.uuid4().hex[:8] + '.jsonl'))
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}', args.run_id):
        parser.error('Invalid run ID')
    if args.timeout_seconds <= 0 or args.poll_seconds <= 0:
        parser.error('Timeout and poll interval must be positive')
    items = load_dataset(args.data_dir.resolve())
    print(f'Validated {len(items)} frozen test tasks and attachments.', flush=True)
    if args.level:
        items = [x for x in items if x['_level'] == args.level]
    if args.task_ids_file:
        wanted = {line.strip() for line in args.task_ids_file.read_text().splitlines() if line.strip()}
        if not wanted:
            parser.error('--task-ids-file is empty')
        items = [x for x in items if x['task_id'] in wanted]
        if len(items) != len(wanted):
            missing = wanted - {x['task_id'] for x in items}
            parser.error('Unknown task IDs: ' + ','.join(sorted(missing)))
    if args.max_items:
        items = items[:args.max_items]
    if args.validate_only:
        return
    client = Client(args.service_url)
    state = client.call('GET', 'state')
    if state['setup']['problem'] or state['paused']:
        raise RuntimeError('SuperAgent setup incomplete or executor paused; open the Superagent panel')
    settings = client.call('GET', 'settings')
    policy = settings['blocks']['policy']['value']
    if not any(x['id'] == 'runner' for x in policy['sources'].get('benchmark', {}).get('allowed', [])):
        raise RuntimeError('Allow runner in the benchmark source policy in the Superagent panel')
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Outside the repository/data tree; only the question and its attachment enter the worker context.
    workspace_root = Path(tempfile.mkdtemp(prefix=args.run_id + '-'))
    retain_workspace = False
    try:
        with args.output.open('x', encoding='utf-8') as stream, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            pending = set()
            remaining = iter(items)
            completed = 0
            def submit_next():
                item = next(remaining, None)
                if item is not None:
                    pending.add(pool.submit(run_item, client, item, args.run_id, workspace_root,
                                            args.data_dir.resolve(), args.timeout_seconds, args.poll_seconds))
            for _ in range(args.concurrency):
                submit_next()
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    completed += 1
                    stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                    stream.flush()
                    print(f"[{completed}/{len(items)}] {result['task_id']} {result['status']} correct={result['correct']}", flush=True)
                    if result.get('cleanup_failed'):
                        retain_workspace = True
                if not retain_workspace:
                    for _ in done:
                        submit_next()
            if retain_workspace:
                raise RuntimeError('Could not stop benchmark worker; refused to start further tasks')
    finally:
        # A failed cleanup leaves the staging files available for audit because
        # a still-running worker may depend on them. Successful/terminal items
        # can release their private inputs after the run ends.
        if not retain_workspace:
            shutil.rmtree(workspace_root, ignore_errors=True)
    print(f'Results: {args.output}', flush=True)


if __name__ == '__main__':
    main()
