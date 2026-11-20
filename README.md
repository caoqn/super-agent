# Super Agent evaluation adapters

This repository separates **method implementations** from their **benchmark adapters**. `methods/` holds reusable method-level code; `benchmarks/` is grouped by dataset and contains the corresponding adapters, runners, task-specific prompts, scoring, and setup instructions. The currently public code covers part of the full experiment matrix; additional baselines can be added as they are released.

## Layout

- `methods/<method>/` contains method-level code, such as the ReAct loop, AgentNet core patch and embedding implementation, and method documentation.
- `benchmarks/<benchmark>/adapters/<method>/` contains that method's integration with one benchmark.
- `benchmarks/<benchmark>/README.md` contains shared dataset and evaluation notes.
- `configs/matrix.yaml` lists the method/benchmark combinations currently represented here.
- `data/manifests/` contains the public MASBench sample-selection manifest. Obtain benchmark task data from its official source.

## Code and dependencies

Keep this project's method implementations and benchmark adapters in Git; install their Python packages and upstream frameworks separately into the run environment. Put benchmark-specific setup instructions and dependency lists beside the adapter, and install only what the selected run needs instead of maintaining one large environment for every baseline. For upstream frameworks such as AgentNet and AOrchestra, record the source and revision and keep their checkout outside tracked source. Keep downloaded benchmark data, container images, and run outputs outside Git; track only small manifests needed to reproduce selections.

## Configuration

Copy `.env.example` to `.env` at the repository root and set your OpenAI-compatible base URL and API key. MASBench scripts default to DeepSeek Flash; set provider-specific cost rates when measuring cost. Do not commit `.env`.

## AgentNet on GAIA and MAC-Bench

Fetch AgentNet into a separate directory and check out the revision required by the adapters:

```sh
git clone https://github.com/zoe-yyx/AgentNet.git /path/to/AgentNet
git -C /path/to/AgentNet checkout 325d39f2a940be5fa903d28c411bd3426b8007f5
python scripts/prepare_agentnet.py /path/to/AgentNet
```

Follow the [GAIA adapter instructions](benchmarks/gaia/adapters/agentnet/README.md) from `agentnet_gaia_run/` and the [MAC-Bench adapter instructions](benchmarks/macbench/adapters/agentnet/README.md) from `agentnet_macbench_run/`. Obtain GAIA access from the [official dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA). Install the [official MAC-Bench repository](https://github.com/rzchen777-wq/MAC_Bench) as described in its adapter instructions.

See [the file-by-file migration map](docs/file-migration-map.csv) for the old-to-new paths.

## MASBench

Clone [MASBench](https://github.com/BUPT-GAMMA/MASBench) into `MASBench-main/`. For AOrchestra, also clone [AOrchestra](https://github.com/FoundationAgents/AOrchestra) into `AOrchestra-main/` and install its dependencies. Generate selected inputs from official benchmark data and compare the manifest with the checked-in selection:

```sh
cp benchmarks/masbench/adapters/agentnet/pick_30_each.py MASBench-main/pick_30_each.py
python MASBench-main/pick_30_each.py
diff -u data/manifests/masbench_seed42_selection_manifest.json MASBench-main/MASBench_selected_30/selection_manifest.json
cp -R MASBench-main/MASBench_selected_30/. benchmarks/masbench/adapters/agentnet/MASBench_selected_30/
```

Run the benchmark-specific adapters from the repository root:

- AgentNet: follow [`benchmarks/masbench/adapters/agentnet/README.md`](benchmarks/masbench/adapters/agentnet/README.md).
- AOrchestra: `python benchmarks/masbench/adapters/aorchestra/run_masbench.py --task all`.
- ReAct: run the three `run_*.py` task scripts in `benchmarks/masbench/adapters/react/`, then its `run_all.py` summary script.
- Vanilla: follow [`benchmarks/masbench/adapters/vanilla/README.md`](benchmarks/masbench/adapters/vanilla/README.md).

The benchmark-level notes are in [`benchmarks/masbench/README.md`](benchmarks/masbench/README.md). The MASBench license and task attribution remain alongside its AgentNet adapter; obtain task inputs from official benchmark sources.


## Super Agent

The GAIA, MASBench and MAC-Bench adapters are in their respective `benchmarks/<benchmark>/adapters/super_agent/` directories. Their external upstream revisions and YAML configurations are recorded in [methods/super_agent](methods/super_agent/README.md). GAIA and MASBench use the external DSH service; MAC-Bench invokes its own official Superagent implementation. Install runtimes and obtain benchmark data separately. GAIA and MASBench YAML files document settings passed as command-line arguments; the MAC-Bench method YAML is loaded by its adapter.

## ReAct and AOrchestra on MAC-Bench

These MAC-Bench integrations use its pinned external implementations and official evaluator. This repository stores method specifications and launchers only; see the [ReAct](benchmarks/macbench/adapters/react/README.md) and [AOrchestra](benchmarks/macbench/adapters/aorchestra/README.md) setup guides.
