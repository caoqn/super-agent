# AOrchestra method

For MAC-Bench, the benchmark integration is supplied by the pinned external [MAC-Bench repository](https://github.com/rzchen777-wq/MAC_Bench), while the method source is installed separately from [FoundationAgents/AOrchestra](https://github.com/FoundationAgents/AOrchestra). Revisions are recorded in [`upstream.json`](upstream.json). The method configuration is [`configs/aorchestra/macbench.yaml`](../../configs/aorchestra/macbench.yaml), and the [MAC-Bench adapter guide](../../benchmarks/macbench/adapters/aorchestra/README.md) documents setup and execution.

This repository does not copy AOrchestra method source. Its separate MASBench integration remains under `benchmarks/masbench/adapters/aorchestra/`.
