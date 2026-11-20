# MASBench

Obtain task inputs from the [official MASBench repository](https://github.com/BUPT-GAMMA/MASBench). The selected task-ID manifest is tracked in `data/manifests/`; generated inputs and run outputs remain local. All MASBench-specific adapters are grouped in `adapters/<method>/`.

ReAct and Vanilla currently share scoring helpers from the ReAct adapter. Preserve that scoring behavior when refactoring.


## Super Agent adapter

Supply the selected 90 examples separately (30 Reasoning, 30 Scheduling, 30 Game). Their `selection_manifest.json` must match `data/manifests/masbench_seed42_selection_manifest.json`. No task payloads are bundled in this checkout. Install dependencies in an external Python environment with `pip install -r requirements/masbench.txt` and clone the official MASBench repository separately.

Validate local data from the project root:

```sh
python3 benchmarks/masbench/validate_data.py --data-dir /path/to/MASBench_selected_30
```

## Run Through SuperAgent

The adapter keeps each released task loop and routes every model inference through an independent SuperAgent benchmark ticket (normal intake, router, worker, report). Role prompts, private views, and turn boundaries are retained. Do not run upstream API scripts for this experiment.

```sh
python benchmarks/masbench/adapters/super_agent/adapter.py \
  --upstream /path/to/MASBench \
  --data-dir /path/to/MASBench_selected_30 \
  --timeout-seconds 1200 \
  --output-dir results/masbench-<run-id>
```

The runner validates the 90-item selection and runs one task at a time. It retries malformed worker responses once, checkpoints each selected item, and continues after individual task failures. If ticket cleanup is delayed, it queues cleanup for retry while continuing the suite. Use `--resume` with an existing output directory to continue unattempted items; add `--retry-failed` to retry previous failures too. Results are checkpointed to `results.jsonl`, with per-episode SuperAgent call/ticket traces, native trajectories, and `summary.json` under the output directory. Each episode has a 1200-second wall timeout by default.

Reasoning uses the released asymmetric context/base conversation (up to 8 turns) and normalized token F1. Scheduling uses the upstream protocol, memory initialization, five-agent negotiation, and 5-round cap; final intervals are checked against `task_plans`. Game uses the upstream base mechanism, selected map, seed 42, and 50 rounds; per-resource delivery is scored against each map's target. Input/output token counts are recovered from locally persisted DSH worker Session usage events; cost remains unavailable unless it is reported by the provider.

## Manager + 3 Workers baseline

This is a separate prompt-driven baseline with one manager and three fixed workers. Its implementation is under `methods/manager3/masbench/`, with the dataset adapter under `benchmarks/masbench/adapters/manager3/`. It uses the repository's checked-in 90-item `test_manifest.json`; raw MASBench data remains external. Prepare the local environment and configuration using [`methods/manager3/README.md`](../../methods/manager3/README.md). The Docker image and Compose file are `docker/manager3.Dockerfile` and `docker/manager3.compose.yaml`.
