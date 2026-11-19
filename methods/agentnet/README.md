# AgentNet method code

The complete AgentNet implementation is maintained in the [upstream repository](https://github.com/zoe-yyx/AgentNet) and checked out at the revision pinned in the root README. This folder contains the method-level core changes used by the released adapters:

- `patches/agentnet-core.patch` applies the shared AgentNet core changes.
- `embedding.py` supplies the configurable embedding implementation copied into the assembled GAIA and MAC-Bench run packages.
- `envfile.py`, `llm.py`, and `tracker.py` provide shared AgentNet runtime utilities; the GAIA and MAC-Bench assembly script copies them into the generated `gaia/` package.

Dataset-specific code and runners live under `benchmarks/<benchmark>/adapters/agentnet/`.
