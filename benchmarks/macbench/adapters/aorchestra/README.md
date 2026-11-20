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
  --upstream /path/to/MAC_Bench --task-id op26_business_finance_001 \
  --output-dir /path/to/results/aorchestra --dry-run
```

The wrapper reads paper task budgets from the pinned MAC-Bench checkout by default. For a development task absent from that file, provide all four budget flags explicitly. Remove `--dry-run` to execute. The launcher verifies both source revisions and delegates to MAC-Bench's native evaluator. Configure identical model and budget values when comparing baselines. The checked-in YAML retains MAC-Bench's initial method settings; paper-run configuration changes should be recorded explicitly.
