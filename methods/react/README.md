# ReAct method

For MAC-Bench, the implementation and benchmark integration are supplied by the pinned external [MAC-Bench repository](https://github.com/rzchen777-wq/MAC_Bench), under `src/mac_bench/baselines/react_single/`. See [`upstream.json`](upstream.json), the MAC-Bench method configuration at [`configs/react/macbench.yaml`](../../configs/react/macbench.yaml), and the [MAC-Bench launcher](../../benchmarks/macbench/adapters/react/README.md).

This repository does not add or copy the MAC-Bench ReAct implementation. Existing ReAct files under this directory are retained for the separate MASBench integration; they are not used by the MAC-Bench runner.
