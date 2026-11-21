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

## MAC-Bench

The same hierarchy is available for MAC-Bench in `methods/manager3/macbench/`.
It uses the official MAC-Bench episode runner and task budgets: one manager
delegates to three persistent workers, worker model calls run concurrently, and
shared environment calls are serialized because all four agents use one episode
world. MAC-Bench data, task environments, Modal credentials, and generated
results stay outside the repository.

Install the MAC-Bench extras in the selected Python environment, then point the
runner at the external checkout:

```bash
python3.11 -m venv .venv-manager3-macbench
source .venv-manager3-macbench/bin/activate
pip install -r requirements/manager3-macbench.txt
pip install -e "/path/to/MAC_Bench-main[agents,modal]"
cp configs/manager3/macbench.env.example configs/manager3/macbench.env
```

Set `MAC_BENCH_HOST_ROOT` to the local `MAC_Bench-main` checkout, set
`MAC_BENCH_LLM_BASE_URL`, `MAC_BENCH_LLM_API_KEY`, and
`MAC_BENCH_LLM_MODEL`, and provide `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET` when using Modal. `MAC_BENCH_ROOT` is the path visible to
the runner; it is `/mnt/macbench` in the Docker setup.

Run a data and environment preflight, then the 48-task test split with:

```bash
MAC_BENCH_ENV_FILE=configs/manager3/macbench.env \
  scripts/run_manager3_macbench.sh --preflight
MAC_BENCH_ENV_FILE=configs/manager3/macbench.env scripts/run_manager3_macbench.sh
```

For Docker, the compose file mounts the external checkout read-only, persists
results under `results/manager3/macbench/`, and passes Modal credentials through:

```bash
docker compose --env-file configs/manager3/macbench.env \
  -f docker/manager3-macbench.compose.yaml build
docker compose --env-file configs/manager3/macbench.env \
  -f docker/manager3-macbench.compose.yaml run --rm manager3-macbench --preflight
docker compose --env-file configs/manager3/macbench.env \
  -f docker/manager3-macbench.compose.yaml run --rm manager3-macbench
```

The runner writes a manifest, per-task summaries, official evaluation fields,
per-agent token/time usage, and a resumable `progress.json` under the selected
output directory. It refuses to mix an existing output directory with a
different split or method configuration.
