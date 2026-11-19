# Super Agent

`upstream.yaml` records the external runtime repository and pinned commit, and `config/` records example benchmark settings. GAIA and MASBench YAML files document settings; the MAC-Bench method YAML is loaded by its adapter. Install and build the external runtime following the root README and its upstream instructions; do not place it under the AgentNet baseline.

GAIA and MASBench adapters submit tasks to the DSH Super Agent benchmark service. The MAC-Bench adapter invokes the separate Superagent baseline in MAC-Bench; see [MAC-Bench notes](../../benchmarks/macbench/README.md). The recorded commits are selected upstream versions, not proof of the versions used in historical experiments.

Adapter instructions: [GAIA](../../benchmarks/gaia/README.md), [MASBench](../../benchmarks/masbench/README.md), [MAC-Bench](../../benchmarks/macbench/README.md).


## DSH benchmark service prerequisite

The GAIA and MASBench adapters require the evaluation service's HTTP API, including `POST /benchmark/tasks`, ticket/thread inspection, and ticket cancellation. The pinned official DSH source alone does not include this project-specific benchmark bridge. Use an external runtime with that bridge installed; cloning the official repository alone is insufficient. This repository packages the Python adapters, not the runtime bridge or credentials.
