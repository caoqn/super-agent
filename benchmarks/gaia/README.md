# GAIA

Use the official [GAIA dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA); data is not stored here. The AgentNet-specific loader, scorer, and runner are in `adapters/agentnet/`.


## Super Agent adapter

此目录保存 EvoTeam `test_100` 的冻结任务清单、校验和答案评分。题目与附件需由本地数据副本提供，数据许可及参考答案不应默认随代码公开。使用 `validate_data.py --evoteam-data /path/to/benchmarks/gaia` 校验 100 道题及附件；适配器位于 `benchmarks/gaia/adapters/super_agent/adapter.py`。

Validate locally supplied data without starting a service:

```sh
python3 benchmarks/gaia/adapters/super_agent/adapter.py --data-dir /path/to/gaia/data --validate-only
```

With the external DSH benchmark service running, evaluate using explicit arguments:

```sh
./scripts/run_benchmark.sh --data-dir /path/to/gaia/data \
  --service-url http://127.0.0.1:3082 --concurrency 4 --timeout-seconds 1200 \
  --output results/gaia-run.jsonl
```

Service state and filesystem isolation must be configured separately; a fresh checkout alone does not isolate prior sessions or answer files. `scripts/run_isolated_gaia.sh` is a macOS-specific helper; review its allowed paths before using external datasets.
