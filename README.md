# Super Agent evaluation adapters

This repository contains the evaluation adapters available for AgentNet on GAIA, MAC-Bench, and MASBench and for AOrchestra, ReAct, and the vanilla baseline on MASBench.

## Configuration

Copy `.env.example` to `.env` in this directory and provide an OpenAI-compatible base URL and your own API key. The MASBench batch scripts use DeepSeek Flash. Set the cost rates to those for your provider when measuring cost.

## AgentNet on GAIA and MAC-Bench

Fetch [AgentNet](https://github.com/zoe-yyx/AgentNet) into a separate directory and check out the revision required by the adapters.

```sh
git clone https://github.com/zoe-yyx/AgentNet.git /path/to/AgentNet
git -C /path/to/AgentNet checkout 325d39f2a940be5fa903d28c411bd3426b8007f5
python prepare_agentnet.py /path/to/AgentNet
```

Follow the [GAIA setup](agentnet_gaia/README.md) from `agentnet_gaia_run/` and the [MAC-Bench setup](agentnet_macbench/README.md) from `agentnet_macbench_run/`. Obtain GAIA access from the [official dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA) and install the [official MAC-Bench repository](https://github.com/rzchen777-wq/MAC_Bench) in `MAC_Bench-1.0/` as described in the adapter instructions.

## MASBench

Clone [MASBench](https://github.com/BUPT-GAMMA/MASBench) into `MASBench-main/`. For the AOrchestra baseline, also clone [AOrchestra](https://github.com/FoundationAgents/AOrchestra) into `AOrchestra-main/` and install its dependencies. Generate the 90 selected inputs using the included sampling script and compare the output manifest against the checked-in selection before copying the inputs.

```sh
cp agentnet_masbench/pick_30_each.py MASBench-main/pick_30_each.py
python MASBench-main/pick_30_each.py
diff -u agentnet_masbench/MASBench_selected_30/selection_manifest.json MASBench-main/MASBench_selected_30/selection_manifest.json
cp -R MASBench-main/MASBench_selected_30/. agentnet_masbench/MASBench_selected_30/
```

- AgentNet adapter: follow [its MASBench setup](agentnet_masbench/README.md).
- AOrchestra: run `python masbench_aorchestra/run_masbench.py --task all`.
- ReAct: run the `run_reasoning.py`, `run_scheduling.py`, and `run_game.py` scripts in `react_adapter_MAsbnech/`. Then run `run_all.py` to summarize results.
- Vanilla baseline: follow [its MASBench setup](vanilla_adapter_MASbench/README.md).

The MASBench license and Reasoning data attribution are included alongside the MASBench adapter. The published sample manifest records the selected task IDs. Obtain task inputs from the corresponding official benchmarks.
