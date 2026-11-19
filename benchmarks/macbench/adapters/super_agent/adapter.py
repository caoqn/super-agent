#!/usr/bin/env python3
"""Run MAC-Bench's Superagent baseline and official evaluator."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True,
                        help='MAC_Bench checkout installed in the selected Python environment')
    parser.add_argument('--python', default=sys.executable,
                        help='Python interpreter with MAC-Bench installed')
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--max-tokens', type=int, default=1000000)
    parser.add_argument('--max-model-calls', type=int, default=3000)
    parser.add_argument('--max-tool-calls', type=int, default=9000)
    parser.add_argument('--max-seconds', type=int, default=7200)
    parser.add_argument('--method-config', type=Path,
                        help='MAC-Bench method-spec YAML (defaults to methods/super_agent/config/macbench.yaml)')
    parser.add_argument('--dry-run', action='store_true', help='Use the upstream evaluator dry run')
    args = parser.parse_args()
    root = args.upstream.expanduser().resolve()
    if not (root / 'src/mac_bench/baselines/superagent/tasks.py').is_file():
        parser.error('--upstream must point to a MAC_Bench checkout with the Superagent baseline')
    if min(args.repetitions, args.max_tokens, args.max_model_calls, args.max_tool_calls, args.max_seconds) <= 0:
        parser.error('repetitions and budgets must be positive')
    output = args.output_dir.expanduser().resolve()
    method_config = (args.method_config.expanduser().resolve() if args.method_config else
                     Path(__file__).resolve().parents[4] / 'methods/super_agent/config/macbench.yaml')
    if not method_config.is_file():
        parser.error('method config does not exist: ' + str(method_config))
    cmd = [args.python, '-m', 'mac_bench', 'evaluate', '--task-id', args.task_id,
           '--method', 'superagent', '--repetitions', str(args.repetitions),
           '--max-tokens', str(args.max_tokens), '--max-model-calls', str(args.max_model_calls),
           '--max-tool-calls', str(args.max_tool_calls), '--max-seconds', str(args.max_seconds),
           '--out', str(output)]
    cmd.extend(['--method-config', str(method_config)])
    if args.dry_run:
        cmd.append('--dry-run')
    # Force the selected checkout rather than silently importing another installed revision.
    env = os.environ.copy()
    env['PYTHONPATH'] = str(root / 'src')
    try:
        return subprocess.run(cmd, cwd=root, env=env, check=False).returncode
    except FileNotFoundError:
        print('Python interpreter not found: ' + args.python, file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
