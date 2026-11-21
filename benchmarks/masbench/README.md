# MASBench

Obtain task inputs from the [official MASBench repository](https://github.com/BUPT-GAMMA/MASBench). The shared task-selection manifest is tracked in `data/manifests/`; generated inputs and run outputs remain local. Adapters for this benchmark are grouped in `adapters/<method>/`.

ReAct and Vanilla currently share scoring helpers from the ReAct adapter. Preserve that scoring behavior when refactoring.


## Super Agent adapter

Prepare the task inputs listed in `data/manifests/masbench_seed42_selection_manifest.json` from an authorized MASBench data copy. The generated `selection_manifest.json` must match the checked-in manifest. Task payloads are not bundled. Install dependencies in an external Python environment with `pip install -r requirements/masbench.txt` and clone the official MASBench repository separately.

Validate local data from the project root:

```sh
python3 benchmarks/masbench/validate_data.py --data-dir /path/to/selected-masbench-data
```

## Run Through SuperAgent

This adapter runs the benchmark tasks through the SuperAgent service while preserving the task protocols and scoring expected by MASBench. Use the repository adapter for this integration:

```sh
python benchmarks/masbench/adapters/super_agent/adapter.py \
  --upstream /path/to/MASBench \
  --data-dir /path/to/selected-masbench-data \
  --timeout-seconds 1200 \
  --output-dir results/masbench-run
```

The runner validates the selection manifest and writes per-task results and a summary under the output directory. Use `--resume` to continue an existing run, or `--retry-failed` to retry failed tasks. Timeout and other run settings are configurable through the command-line options.

Task protocols and scoring follow the benchmark definitions. Consult the adapter configuration and upstream benchmark documentation for task-family settings and metric details.

## Manager + 3 Workers baseline

This is a separate prompt-driven baseline with one manager and three workers. Its implementation is under `methods/manager3/masbench/`, with the dataset adapter under `benchmarks/masbench/adapters/manager3/`. It uses the checked-in task-selection manifest; raw MASBench data remains external. Prepare the local environment and configuration using [`methods/manager3/README.md`](../../methods/manager3/README.md). The Docker image and Compose file are `docker/manager3.Dockerfile` and `docker/manager3.compose.yaml`.
