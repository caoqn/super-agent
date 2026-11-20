# Skill-MAS × MAC-Bench

First [prepare Skill-MAS](../../../../methods/skill_mas/README.md) with `--dataset macbench`. In its dedicated Python 3.11 environment, run from this repository root:

```sh
python -m pip install -r benchmarks/macbench/adapters/skill_mas/requirements.txt
git clone https://github.com/rzchen777-wq/MAC_Bench.git /path/to/MAC_Bench
git -C /path/to/MAC_Bench checkout c70d962ee8e428a6d9bae42af8d92e086581ee0d
python -m pip install -e '/path/to/MAC_Bench[modal,agents]'
export MAC_BENCH_ROOT=/path/to/MAC_Bench
```

Run the upstream download command from `/path/to/MAC_Bench` (`mac-bench data download --split test`, or `dev`), then return to this repository root. Follow that checkout's environment setup instructions. This adapter expects `data/tasks.json`, `data/public_tasks.json` and the referenced task bundles. The catalogue must use `collaboration.tasks.v1`; public and private catalogue revisions must match. Modal-backed tasks require the upstream Modal account/authentication setup. This integration was migrated from a local MAC-Bench 1.2.0 tree; compatibility with the pinned public revision still needs an environment-backed smoke run.

```sh
python methods/skill_mas/run_native_eval.py \
  --runtime skill_mas_macbench_run --backend macbench --model YOUR_MODEL_ID \
  --skill-file /path/to/experiment/SKILL.md --task-ids TASK_ID --split test
```

`train` maps to upstream `dev`; `test` maps to upstream `test`. Task tools come from `handle.view.tools`; evaluation uses native `handle.evaluate()` and its `gc` field. All workflow roles share the same task environment. This adapter's `admit_call` is a no-op, so it does **not** enforce official shared token/call/action budgets. Document this as a custom Skill-MAS setting rather than an official budget-compliant run. Task service setup, model calls and full native evaluation have not been verified by the migration's offline checks.
