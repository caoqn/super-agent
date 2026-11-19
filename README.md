# Super Agent evaluation adapters

This repository contains the currently released evaluation adapters for AgentNet on GAIA, MAC-Bench, and MASBench, plus AOrchestra, ReAct, and vanilla adapters for MASBench. The current public code covers part of the full experiment matrix; add the remaining methods as they are released.

## Layout

- `methods/<method>/<benchmark>/` contains method-specific implementations, adapters, and run instructions.
- `benchmarks/<benchmark>/README.md` records benchmark data access and shared evaluation notes.
- `configs/matrix.yaml` lists the combinations currently represented here.
- `data/manifests/` contains the public MASBench sample-selection manifest. Obtain benchmark task data from its official source.

## Configuration

Copy `.env.example` to `.env` at the repository root and set your OpenAI-compatible base URL and API key. MASBench scripts default to DeepSeek Flash; set provider-specific cost rates when measuring cost. Do not commit `.env`.

## AgentNet on GAIA and MAC-Bench

Fetch AgentNet into a separate directory and check out the revision required by the adapters:

```sh
git clone https://github.com/zoe-yyx/AgentNet.git /path/to/AgentNet
git -C /path/to/AgentNet checkout 325d39f2a940be5fa903d28c411bd3426b8007f5
python scripts/prepare_agentnet.py /path/to/AgentNet
```

Follow the [GAIA instructions](methods/agentnet/gaia/README.md) from the generated `agentnet_gaia_run/` directory and the [MAC-Bench instructions](methods/agentnet/macbench/README.md) from `agentnet_macbench_run/`. Obtain GAIA access from the [official dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA). Install the [official MAC-Bench repository](https://github.com/rzchen777-wq/MAC_Bench) as described in its adapter instructions.

See [the file-by-file migration map](docs/file-migration-map.csv) for the old-to-new paths.

## MASBench

Clone [MASBench](https://github.com/BUPT-GAMMA/MASBench) into `MASBench-main/`. For AOrchestra, also clone [AOrchestra](https://github.com/FoundationAgents/AOrchestra) into `AOrchestra-main/` and install its dependencies. Generate selected inputs from official benchmark data and compare the manifest with the checked-in selection:

```sh
cp methods/agentnet/masbench/pick_30_each.py MASBench-main/pick_30_each.py
python MASBench-main/pick_30_each.py
diff -u data/manifests/masbench_seed42_selection_manifest.json MASBench-main/MASBench_selected_30/selection_manifest.json
cp -R MASBench-main/MASBench_selected_30/. methods/agentnet/masbench/MASBench_selected_30/
```

- AgentNet: follow [its MASBench instructions](methods/agentnet/masbench/README.md).
- AOrchestra: `python methods/aorchestra/masbench/run_masbench.py --task all`.
- ReAct: run the three `run_*.py` task scripts in `methods/react/masbench/`, then `run_all.py` there.
- Vanilla: follow [its MASBench instructions](methods/vanilla/masbench/README.md).

See `benchmarks/masbench/README.md` for dataset and selection notes. The MASBench license and task attribution remain alongside the AgentNet MASBench adapter; obtain task inputs from official benchmark sources.
