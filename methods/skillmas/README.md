# Skill-MAS Reproduction Adapter

This directory provides a pinned source reference for [Skill-MAS](https://github.com/linhh29/Skill_MAS), local compatibility patches, GAIA/MASBench/MACBench adapters, split manifests, and frozen skills. Prepare the upstream source separately at runtime; it is not copied into this repository. The source revision and archive checksum are recorded in `upstream.json`; `patches/runtime.patch` targets that source package.

## Runtime Environment

Python 3.11 is recommended. `environment/python-current-snapshot.txt` records the dependency versions used for validation. Create a virtual environment on the target platform and install dependencies according to the upstream and benchmark documentation. Additional MASBench Game dependencies are listed in `requirements/masbench-game.txt`. MACBench requires its harness, task services, and Docker environment.

## Data Preparation

Datasets and task packages are not included in this repository. Follow each benchmark's official project for data downloads, authentication, and licensing.

- **GAIA**: `--data-root` must contain the configured training and test files and the `val_files/` attachment directory. Obtain the data from an authorized source and preserve its original layout.
- **MASBench**: `--data-root` must point to a root containing `Reasoning/reasoning.json`, `Scheduling/`, and `Game/custom_map/`. The runner validates task IDs and order and fails if data is missing.
- **MACBench**: `--data-root` may point to a task catalogue directory or the project root. After preparing the data, verify that the `mac-bench` CLI, required services, and Docker are available.

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

Frozen skills are in `artifacts/{gaia,masbench,macbench}/SKILL.md`; their SHA-256 values are in `experiment_evidence.json`. To evaluate a frozen skill, pass the corresponding file with `--skill`. A separate MACBench manager/worker experiment is not part of these Skill-MAS evaluations. Price settings are estimates and do not represent provider invoices.

## Budgets, Limitations, and Validation Status

The Skill-MAS pipeline uses `max_steps=300` during training; the default test tool limit is `max_rounds=12`. The timeout adds an 800-second wall-clock window to the base time; it does not pause precisely during API waits. `max_output_tokens` limits a single response, not cumulative tokens for one task. The adapter does not enforce MACBench's official token/call budgets, and MACBench's current `admit_call` does not count calls. If the study requires official budget compliance, add an auditable budget executor first.

Local checks passed for preparation with the pinned ZIP checksum, loading all train/test splits, verifying IDs and order, and checking entry points with stubbed execution. The checks do not make API requests; full benchmark runs require the external data, services, and Docker environments described above. See `environment/README.md` for the check commands.
