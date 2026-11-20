# Skill-MAS × GAIA

First [prepare Skill-MAS](../../../../methods/skill_mas/README.md) with `--dataset gaia`. Install from this repository root:

```sh
python -m pip install -r benchmarks/gaia/adapters/skill_mas/requirements.txt
export GAIA_DATA_ROOT=/path/to/gaia
```

Obtain access to [official GAIA data](https://huggingface.co/datasets/gaia-benchmark/GAIA) separately. This adapter expects the experiment's exported `level_1_test_100.json`, `level_2_test_100.json`, `level_3_test_100.json` (or `level_<n>_train_20.json`) under `GAIA_DATA_ROOT`, with `task_id`, `Question`, `Final answer` and optional `file_name`. Place attachments under `val_files/<file_name>` or `val_files/<task_id>/<file_name>`. Missing split files fail with an error instead of silently running zero tasks. The exact GAIA selection/export files are not released here yet.

```sh
python methods/skill_mas/run_native_eval.py \
  --runtime skill_mas_gaia_run --backend gaia --model YOUR_MODEL_ID \
  --skill-file /path/to/experiment/SKILL.md --task-ids TASK_ID --split test
```

The adapter exposes file, shell and web tools. It preserves the supplied evaluator: strip the first `FINAL ANSWER:` prefix, lowercase, remove everything except ASCII letters/digits, then compare. This differs from official GAIA number/list/string normalization; report the metric accurately and use the same metric across baselines when comparing results. Data, credentials and output traces are not checked in.
