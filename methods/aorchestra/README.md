# AOrchestra method

The AOrchestra implementation is obtained from the [official upstream repository](https://github.com/FoundationAgents/AOrchestra). This repository contains only the MASBench integration in `benchmarks/masbench/adapters/aorchestra/`.

## Install the upstream dependency

The upstream instructions currently require Python 3.13 and install its complete dependency set from its own requirements file. Keep the checkout and environment outside the tracked source:

```bash
git clone https://github.com/FoundationAgents/AOrchestra.git AOrchestra-main
git -C AOrchestra-main checkout 14a1a2051d6b03c479b706f8f555a60b8419e3b5
python3.13 -m venv .venv-aorchestra
source .venv-aorchestra/bin/activate
python -m pip install -r AOrchestra-main/requirements.txt
```

From the repository root, run the adapter with that environment active:

```bash
python benchmarks/masbench/adapters/aorchestra/run_masbench.py --task all
```

The adapter also needs the external `MASBench-main/` checkout, the selected task data described in [`benchmarks/masbench/README.md`](../../benchmarks/masbench/README.md), and the root `.env` values documented in the main README. Do not install AOrchestra's full requirements into the AgentNet environment.
