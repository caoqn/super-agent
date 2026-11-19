#!/usr/bin/env python3
"""Analyze recorded GAIA episodes against the frozen manifest."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.gaia import accuracy
from benchmarks.common.evaluation import autonomy, cost, latency


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    with args.results.open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    manifest = json.loads((ROOT / 'benchmarks/gaia/manifest.json').read_text(encoding='utf-8'))
    expected = {item['task_id'] for item in manifest['tasks']}
    ids = [row['task_id'] for row in rows]
    if len(ids) != len(set(ids)) or not set(ids) <= expected:
        raise ValueError('Results contain duplicate or unknown task IDs')
    for row in rows:
        if not isinstance(row.get('status'), str) or not row['status']:
            raise ValueError('Each result needs a status')
        if row['status'] == 'completed' and (not isinstance(row.get('prediction'), str) or not isinstance(row.get('reference'), str)):
            raise ValueError('Completed results need prediction and reference strings')
        for field in ('cost_usd', 'latency_seconds', 'human_interventions'):
            value = row.get(field)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0):
                raise ValueError('%s must be a nonnegative number' % field)
    successes = sum(row['status'] == 'completed' and accuracy.correct(row['prediction'], row['reference']) for row in rows)
    result = {'expected': len(expected), 'recorded': len(rows), 'correct': successes,
              'accuracy_recorded': successes / len(rows) if rows else None,
              'cost': cost.summarize(rows), 'latency': latency.summarize(rows), 'autonomy': autonomy.summarize(rows)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
