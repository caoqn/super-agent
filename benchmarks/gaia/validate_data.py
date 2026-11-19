#!/usr/bin/env python3
"""Verify that locally supplied GAIA payloads match the frozen EvoTeam IDs."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--metadata', type=Path, help='GAIA metadata JSONL')
    source.add_argument('--evoteam-data', type=Path, help='EvoTeam data/gaia directory')
    parser.add_argument('--attachments', type=Path, help='GAIA attachment directory')
    args = parser.parse_args()
    manifest = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))['tasks']
    expected = [item['task_id'] for item in manifest]
    if len(expected) != 100 or len(set(expected)) != 100:
        raise ValueError('Manifest must contain exactly 100 distinct task IDs')
    if args.evoteam_data:
        items = []
        for level in (1, 2, 3):
            path = args.evoteam_data / ('level_%d_test_100.json' % level)
            items.extend(json.loads(path.read_text(encoding='utf-8')))
        attachments = args.attachments or args.evoteam_data / 'val_files'
        if [str(item.get('task_id', '')) for item in items] != expected:
            raise ValueError('EvoTeam split IDs or ordering do not match the frozen manifest')
    else:
        with args.metadata.open(encoding='utf-8') as stream:
            items = [json.loads(line) for line in stream if line.strip()]
        attachments = args.attachments
    ids = [str(item.get('task_id', '')) for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate task IDs in input data')
    by_id = dict(zip(ids, items))
    missing = [task_id for task_id in expected if task_id not in by_id]
    if missing:
        raise ValueError('Missing %d task IDs: %s' % (len(missing), ', '.join(missing)))
    for task_id in expected:
        item = by_id[task_id]
        if not item.get('Question') or item.get('Final answer') is None:
            raise ValueError('%s: missing question or reference answer' % task_id)
        filename = item.get('file_name') or ''
        if filename:
            if not attachments or not (attachments / filename).is_file():
                raise ValueError('%s: missing attachment %s' % (task_id, filename))
    print('Validated 100 frozen GAIA tasks and their attachments.')


if __name__ == '__main__':
    main()
