# EvoTeam on MAC-Bench

See [external EvoTeam source and compatibility status](../../../../methods/evoteam/README.md). This directory preserves the native catalogue integration, scoring delegation, and task-tool bridge. Model execution awaits porting the upstream runner/budget interfaces.

Clone and install the benchmark separately in a Python 3.11 environment:

```bash
git clone https://github.com/rzchen777-wq/MAC_Bench.git /path/to/MAC_Bench
git -C /path/to/MAC_Bench checkout c70d962ee8e428a6d9bae42af8d92e086581ee0d
python -m pip install -e '/path/to/MAC_Bench[meta_team]'
export MAC_BENCH_ROOT=/path/to/MAC_Bench
```

The pinned upstream declares the `meta_team` extra. Obtain its catalogue and task bundles through the benchmark's own setup instructions; private evaluators remain in the installed benchmark. A default install does not download task bundles or provide environment services automatically. Record service/image versions used for the paper separately.

```bash
python scripts/run_evoteam.py --config configs/evoteam/macbench.json --dry-run
```

The data-only check validates `data/tasks.json`, `data/public_tasks.json`, matching world revisions, and bundle paths. The preserved runtime adapter additionally expects `configs/evaluation/task_budgets.json` and native `open_task`, episode, evaluator, and receipt APIs; those are not exercised by the data check. Scoring delegates to `mac_bench.evaluation.reports.metrics.evaluation_fields`; the native evaluator determines the outcome.

The selected frozen pool is `pool_MACBENCH_OURS_V006_TEST`; the initial pool is also retained. `tool_bridge.py` is benchmark adapter code from the uploaded runtime's `tools/macbench.py`. It is no longer method source under `methods/`. Integrating the bridge with the official runtime's tool registry remains required.
