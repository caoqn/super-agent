# AOrchestra on MAC-Bench

The benchmark adapter is supplied by the external [MAC-Bench repository](https://github.com/rzchen777-wq/MAC_Bench); the method implementation is fetched from [FoundationAgents/AOrchestra](https://github.com/FoundationAgents/AOrchestra). Both revisions and the method configuration are recorded in [`methods/aorchestra/upstream.json`](../../../../methods/aorchestra/upstream.json) and [`configs/aorchestra/macbench.yaml`](../../../../configs/aorchestra/macbench.yaml). The shared baseline protocol and task-budget source are recorded in [`configs/macbench/upstream.json`](../../../../configs/macbench/upstream.json). This directory contains only a launcher and setup notes.

Install MAC-Bench and its AOrchestra/environment dependencies separately:

```bash
git clone https://github.com/rzchen777-wq/MAC_Bench.git /path/to/MAC_Bench
git -C /path/to/MAC_Bench checkout c70d962ee8e428a6d9bae42af8d92e086581ee0d
python -m venv /path/to/macbench-env
source /path/to/macbench-env/bin/activate
python -m pip install -e '/path/to/MAC_Bench[modal,agents,aorchestra]'
python /path/to/MAC_Bench/scripts/fetch_baselines.py --name AOrchestra
```

The fetch script installs the revision pinned in MAC-Bench's `configs/baselines/upstream.json`. Configure `MAC_BENCH_LLM_BASE_URL`, `MAC_BENCH_LLM_API_KEY`, and `MAC_BENCH_LLM_MODEL`; obtain task data and environment assets through the MAC-Bench setup guide. Choose task IDs with `mac-bench tasks list --split test`.

```bash
python benchmarks/macbench/adapters/aorchestra/run.py \
  --upstream /path/to/MAC_Bench --task-id TASK_ID \
  --output-dir /path/to/results/aorchestra --dry-run
```

The wrapper reads task budgets from the pinned MAC-Bench checkout by default; provide explicit budget flags when the selected task has no recorded defaults. Remove `--dry-run` to execute. The launcher verifies source revisions and delegates to MAC-Bench's evaluator. Record any changes to the checked-in method configuration with the run settings.
