# GAIA

Obtain benchmark data from the [official GAIA dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA); raw data is not stored here. Adapter-specific loaders, scorers, and runners are documented alongside each adapter.


## Super Agent adapter data

The task manifest and validation helper in this directory support the Super Agent adapter. They record the task IDs, checksums, and reference-answer metadata used by that adapter, but do not include task content or attachments. Supply those from an authorized local copy of GAIA and validate them against the manifest:

```sh
python3 benchmarks/gaia/validate_data.py --metadata /path/to/metadata.jsonl --attachments /path/to/attachments
```

The Super Agent adapter is at `benchmarks/gaia/adapters/super_agent/adapter.py`.

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
