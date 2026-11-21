# ReAct on MAC-Bench

The ReAct method implementation and its MAC-Bench adapter are supplied by the external [MAC-Bench repository](https://github.com/rzchen777-wq/MAC_Bench), pinned in [`methods/react/upstream.json`](../../../../methods/react/upstream.json). This directory contains only a launcher. The method specification is preserved at [`configs/react/macbench.yaml`](../../../../configs/react/macbench.yaml); the task budget table and its provenance are in [`configs/macbench/`](../../../../configs/macbench/upstream.json).

Install MAC-Bench and its environment/model dependencies in a separate checkout and virtual environment. Keep data, caches, and output outside this repository.

```bash
git clone https://github.com/rzchen777-wq/MAC_Bench.git /path/to/MAC_Bench
git -C /path/to/MAC_Bench checkout c70d962ee8e428a6d9bae42af8d92e086581ee0d
python -m venv /path/to/macbench-env
source /path/to/macbench-env/bin/activate
python -m pip install -e '/path/to/MAC_Bench[modal,agents]'
```

Configure `MAC_BENCH_LLM_BASE_URL`, `MAC_BENCH_LLM_API_KEY`, and `MAC_BENCH_LLM_MODEL`, then acquire the task data and environment using the MAC-Bench setup guide. Choose task IDs with `mac-bench tasks list --split test`.

```bash
python benchmarks/macbench/adapters/react/run.py \
  --upstream /path/to/MAC_Bench --task-id TASK_ID \
  --output-dir /path/to/results/react --dry-run
```

The wrapper reads task budgets from the pinned MAC-Bench checkout by default; provide explicit budget flags when the selected task has no recorded defaults. Remove `--dry-run` to execute. The wrapper validates the MAC-Bench revision, passes the tracked method YAML to the evaluator, and uses MAC-Bench's result and scoring path. Explicit budget flags override task defaults.
