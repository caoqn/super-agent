"""Rebuild EvoTeam's fixed GAIA splits from an authorized official local download."""
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Official 2023/validation directory containing metadata.jsonl and attachments')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/gaia')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory to preserve existing data.')
    rows = [json.loads(line) for line in (args.source / 'metadata.jsonl').read_text().splitlines() if line.strip()]
    by_id = {row['task_id']: row for row in rows}
    manifests = json.loads((ROOT / 'benchmarks/gaia/evoteam/splits.json').read_text())['splits']
    outputs, files = {}, {}
    for split, levels in manifests.items():
        for level, ids in levels.items():
            selected = [by_id[task_id] for task_id in ids]
            for row in selected:
                assert str(row['Level']) == level, f"Level mismatch for {row['task_id']}"
                filename = row.get('file_name', '')
                if filename:
                    if Path(filename).name != filename:
                        raise ValueError('Expected a plain GAIA attachment filename')
                    source = args.source / filename
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    files[filename] = source
            outputs[f'level_{level}_{split}.json'] = selected
    (args.output / 'val_files').mkdir(parents=True)
    for filename, rows in outputs.items():
        (args.output / filename).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    for filename, source in files.items():
        shutil.copy2(source, args.output / 'val_files' / filename)
    print(f'Prepared train_20/test_100 with {len(files)} attachments at {args.output}')


if __name__ == '__main__':
    main()
