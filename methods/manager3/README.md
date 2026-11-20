# Manager + 3 Workers

This method is a prompt-driven MASBench baseline with one manager and three fixed workers. The manager delegates native benchmark work to `worker-1`, `worker-2`, and `worker-3`; all four agents share one per-task model, tool, token, and wall-clock budget.

The MASBench-specific runner is in `methods/manager3/masbench/`. Its benchmark adapter is in `benchmarks/masbench/adapters/manager3/`.

Prepare a dedicated environment and local configuration from the repository root:

```bash
python3.11 -m venv .venv-manager3
source .venv-manager3/bin/activate
pip install -r requirements/manager3-masbench.txt
cp configs/manager3/masbench.env.example configs/manager3/masbench.env
```

Edit `configs/manager3/masbench.env` with the local `MASBench-main` path, API base URL, and API key. The API-compatible service must expose the model configured by `MASBENCH_MODEL`.

Run an offline check and then the test split:

```bash
MASBENCH_ENV_FILE=configs/manager3/masbench.env scripts/run_manager3_masbench.sh --preflight
MASBENCH_ENV_FILE=configs/manager3/masbench.env scripts/run_manager3_masbench.sh
```

For Docker, keep the same settings file and run:

```bash
docker compose --env-file configs/manager3/masbench.env \
  -f docker/manager3.compose.yaml run --rm manager3 --preflight
docker compose --env-file configs/manager3/masbench.env \
  -f docker/manager3.compose.yaml run --rm manager3
```

The test runner checkpoints each task, records scores and token/time usage, and skips tasks with a closed summary when resumed. Results are written under `results/manager3/masbench/test90/`; generated data and credentials should remain untracked.
