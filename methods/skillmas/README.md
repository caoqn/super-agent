# Skill-MAS Reproduction Adapter

This directory provides a pinned source reference for [Skill-MAS](https://github.com/linhh29/Skill_MAS), local compatibility patches, GAIA/MASBench/MACBench adapters, split manifests, and historical frozen skills. Prepare the upstream source separately at runtime; it is not copied into this repository. The commit in `upstream.json` comes from a comment in the original source ZIP, and its SHA-256 verifies the archive; the commit has not been independently confirmed against online Git history. `patches/runtime.patch` targets that archive. It does not prove that the runtime source used in the historical experiments exactly matches the pinned snapshot; the MACBench result-directory mapping was also made portable.

## Runtime Environment

Python 3.11 is recommended. `environment/python-current-snapshot.txt` is the current `pip freeze` from the experiment machine, not a historical runtime lockfile or a minimal requirements file for a clean environment. The historical MACBench Python environment has not been confirmed. For strict reproduction, create a fresh virtual environment on the target platform, install dependencies according to the upstream and benchmark documentation, and produce a separate lockfile that records Python, OS, and architecture. Additional MASBench Game dependencies are listed in `requirements/masbench-game.txt`; verify their versions when installing. MACBench requires its harness, task services, and Docker environment; the historical image digest is not recorded here.

## Data Preparation

Datasets and task packages are not included in this repository. Follow each benchmark's official project for data downloads, authentication, and licensing.

- **GAIA**: `--data-root` must contain `level_1_train_20.json`, `level_2_train_20.json`, `level_3_train_20.json`, and the corresponding `level_{1,2,3}_test_100.json` files. The frozen manifests aggregate 20 training tasks and 100 test tasks across levels. Place attachments under `val_files/` and preserve the original GAIA attachment layout. The repository does not contain raw GAIA data or attachment hashes; obtain the data from an authorized source and verify its version.
- **MASBench**: `--data-root` must point to a root containing `Reasoning/reasoning.json`, `Scheduling/`, and `Game/custom_map/`. The repository provides manifests for 18 training and 90 test IDs; the Reasoning split excludes test task families. See `splits/masbench/split_audit.json` for the audit. The local MASBench source directory has no clean Git revision because its Git status includes many untracked files, so retain source-file hashes yourself. The runner validates IDs and order and fails if data is missing.
- **MACBench**: `--data-root` may point to a task catalogue directory or the project root. This repository pins 48 test IDs, each task's world revision, and the train/test settings. After preparing the data, verify that the `mac-bench` CLI, required services, and Docker are available. The known data revision `425592652fc225292903c055e2c3a595b061a4a7` is not the harness code version or image digest; historical records for both are missing.

## Prepare and Run

Run these commands from the repository root. The archive must be a ZIP of the pinned upstream version; if `--archive` is omitted, the preparation script clones the pinned Git commit. The destination must be a new `Skill_MAS` directory.

```bash
python3 -m venv .venv-skillmas
source .venv-skillmas/bin/activate
python -m pip install -r requirements/masbench-game.txt
# Install other Python and system dependencies according to the Skill-MAS, GAIA, and MACBench documentation.
python scripts/prepare_skillmas.py --destination /path/to/work/Skill_MAS --archive /path/to/Skill_MAS-main.zip

export OPENAI_API_KEY='your-key'
export OPENAI_BASE_URL='https://your-openai-compatible-endpoint/v1'

# Validate paths and task manifests without making API requests.
python scripts/run_skillmas.py --upstream /path/to/work/Skill_MAS --benchmark gaia --phase train --data-root /path/to/gaia --output /path/to/results --run-id gaia-k3 --validate-only
# Test with the SKILL.md produced by training.
python scripts/run_skillmas.py --upstream /path/to/work/Skill_MAS --benchmark gaia --phase train --data-root /path/to/gaia --output /path/to/results --run-id gaia-k3
python scripts/run_skillmas.py --upstream /path/to/work/Skill_MAS --benchmark gaia --phase test --data-root /path/to/gaia --output /path/to/results --run-id gaia-test --skill /path/to/SKILL.md
```

Replace `gaia` with `masbench` or `macbench` and provide the matching data root. Ensure that the model ID exposed by the API server matches `deepseek-flash` in the configuration. Each experiment uses one evolution round: 20 GAIA tasks with k=3 per task, 18 MASBench tasks with k=3 per task, and 6 MACBench dev tasks with k=2 per task. The test sets contain 100, 90, and 48 tasks, respectively. Concurrency and timeout settings are in `config/`. Use different run IDs for training and testing; keep the configuration unchanged when resuming a run. The results directory stores the effective configuration and skill hash.

Historical frozen skills are in `artifacts/{gaia,masbench,macbench}/SKILL.md`; their SHA-256 values are in `experiment_evidence.json`. To test historical results, pass the corresponding file with `--skill`. A separate MACBench manager/worker experiment is not part of these Skill-MAS frozen-skill tests. Price settings are estimates and do not represent provider invoices.

## Budgets, Limitations, and Validation Status

The Skill-MAS pipeline uses `max_steps=300` during training; the default test tool limit is `max_rounds=12`. The timeout adds an 800-second wall-clock window to the base time; it does not pause precisely during API waits. `max_output_tokens` limits a single response, not cumulative tokens for one task. The adapter does not enforce MACBench's official token/call budgets, and MACBench's current `admit_call` does not count calls. If the study requires official budget compliance, add an auditable budget executor first.

Local checks passed for preparation with the pinned ZIP checksum, loading all six train/test splits, verifying IDs and order, and checking entry points with stubbed execution. No API requests were made, and Docker, external services, and real benchmarks have not been tested end to end on a clean machine. Strict reproduction still lacks the full historical Python lockfile, the MACBench harness commit and image digest, GAIA data and attachment hashes, and a clean MASBench source revision. See `environment/README.md` for the check commands.
