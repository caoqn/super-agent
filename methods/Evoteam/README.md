# EvoTeam Benchmark Adapter

The method source is obtained from the [upstream EvoTeam repository](https://github.com/caoqn/EvoTeam) at the revision recorded in `source-lock.json`. This repository stores benchmark adapters, experiment configurations, initial pools, and the source version lock; it does not include the method implementation or generated training artifacts.

## Install the Pinned Source Version

Use Python 3.11 and run the following commands from the evaluation repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python scripts/setup_evoteam.py
cp .env.example .env
# Set DEV_API_BASE and DEV_API_KEY in .env.
export META_TEAM_ENV_FILE="$PWD/.env"
```

The installer downloads the exact commit specified by `source-lock.json` from public GitHub, verifies each source file's SHA-256, and installs runtime dependencies; GitHub authentication is not required. By default it installs into the Git-ignored `external/EvoTeam/` directory. Add `--no-install` to download only, or use `--destination` to select another location. For a non-default location, export `EVOTEAM_ROOT` in the shell; setting it only in `.env` does not change source resolution. Existing directories are not overwritten, and incompatible or modified source is reported as an error.

The default model is `deepseek-chat`. The three benchmark adapters are in `benchmarks/{gaia,macbench,masbench}/adapters/evoteam/`; `methods/Evoteam/` contains only shared source setup, the version lock, and documentation. All three training configurations use `workers=1` and enable `--evolve`. Model variables in an explicit `META_TEAM_ENV_FILE` may override configuration defaults; check `META_TEAM_MODEL`, `TEAM_SELECTOR_MODEL`, and `TEAM_REFLECTION_MODEL` in `.env`. Set data paths with shell `export` commands.

## Prepare Data

Datasets and reference answers are not distributed with the code. Obtain access to the corresponding upstream data yourself.

**GAIA**: Obtain `2023/validation/` (`metadata.jsonl` and attachments) from [official GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA), then run:

```bash
python scripts/prepare_evoteam_gaia.py --source "$GAIA_VALIDATION_DIR"
```

This script prepares the inputs defined by the adapter split manifest. The default output is `data/gaia`. For another location, export `GAIA_DATA_DIR`; the directory must contain the generated split files and attachments. These files are local inputs for this adapter.

**MAC-Bench**: Obtain the [MAC-Bench](https://github.com/rzchen777-wq/MAC_Bench) source and data, including `data/`, `src/mac_bench/`, and `configs/evaluation/task_budgets.json`, then set:

```bash
export MAC_BENCH_ROOT="$PWD/external/MAC_Bench"
python -m pip install -e "${MAC_BENCH_ROOT}[meta_team]"
```

Training uses `dev`, and testing uses `test`. Upstream token, tool-call, and time budgets are retained, as is the chair's emergency submission behavior; the agent step limit is disabled. Native tool environments must also meet the MAC-Bench requirements, such as task-specific external commands.

**MASBench**: Obtain [official MASBench](https://github.com/BUPT-GAMMA/MASBench), including Reasoning, Scheduling, and Game data:

```bash
export MASBENCH_ROOT="$PWD/external/MASBench"
python scripts/prepare_masbench.py --source "$MASBENCH_ROOT"
export MASBENCH_SUPPORT_ROOT="$PWD/external/masbench-support"
python -m pip install --target "$MASBENCH_SUPPORT_ROOT/.masbench_deps" -r requirements/masbench-game.txt
```

The fixed split contains 18 training tasks and 90 test tasks. Export `MASBENCH_SPLIT_DIR` to use a custom split output location. Game subprocesses use a separate support directory and dependencies.

## Train from an Initial Pool

```bash
python scripts/run.py --config configs/Evoteam_gaia_train.json
python scripts/run.py --config configs/Evoteam_macbench_train.json
python scripts/run.py --config configs/Evoteam_masbench_train.json
```

Append `--dry-run --max-items 1` before a run to check data selection without calling the model. Additional CLI arguments override defaults, for example `--run-id my_mac_train` or `--cases 0`. Initial pools are in `benchmarks/{gaia,macbench,masbench}/adapters/evoteam/pools/initial`; see `initial_pools.json` for their source and file checksums.

Training copies the initial pool to `team/v000` in the run directory, then saves evolved versions. Later tasks use the latest version; the initial snapshot remains unchanged. The default output is `results/Evoteam/`; export `EVAL_RESULTS_DIR` to redirect it. Resume with `--run-id <existing-run-name> --resume`. Use the relevant entry point's `--help` for all options.

`configs/Evoteam_{gaia,macbench,masbench}.json` retain the original experiment test parameters, but the referenced frozen pools are not provided. After training, use `--team <complete-pool-directory-produced-by-training>` to select a test pool. Do not add `--evolve` during testing; an initial pool that has not undergone the corresponding training is not equivalent to the original experiment's frozen pool.

## Validation Scope

```bash
python -m pip install -r requirements/dev.txt
python -m pytest -q tests/test_evoteam_source.py tests/test_evoteam_training.py tests/test_evoteam_budget.py
python -m pytest -q external/EvoTeam/tests
```

Offline training tests for all three initial pools use the real team selection, collaboration, reflection, and version-saving logic while stubbing model responses and evaluation environments. They verify that the first task creates a version, the second reuses it, and the initial pool remains unchanged. Budget regression tests cover token limits and emergency submission. Integration tests are enabled automatically when the source is installed; real-data tests also require the data environment described above. Offline tests do not call paid models and do not reproduce complete benchmark scores.
