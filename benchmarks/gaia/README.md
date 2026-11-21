# GAIA

Use the official [GAIA dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA); data is not stored here. The AgentNet-specific loader, scorer, and runner are in `adapters/agentnet/`.


## Super Agent adapter

This directory stores EvoTeam's frozen `test_100` task manifest, checksums, and answer-scoring data. Provide tasks and attachments from a local dataset copy; data licenses and reference answers should not be assumed to be redistributable with code. Run `validate_data.py --evoteam-data /path/to/benchmarks/gaia` to validate the 100 tasks and attachments. The adapter is at `benchmarks/gaia/adapters/super_agent/adapter.py`.

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
