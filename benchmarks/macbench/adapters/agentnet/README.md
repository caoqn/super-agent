# AgentNet × MAC-Bench

This is the MAC-Bench benchmark adapter for AgentNet. The run package is assembled by `scripts/prepare_agentnet.py` from the repository root.

## Directory layout

```
agentnet_macbench_run/ (generated run directory)
├── src/                # AgentNet core
├── models/             # AgentNet models
├── prompt/             # Prompt collection
├── config/             # Configuration
├── evaluator/          # Evaluators
├── utils/              # Utilities
├── gaia/               # MAC-Bench integration (macbench_adapter.py, etc.) and compatibility modules
├── methods/agentnet/   # Shared method components from methods/agentnet/
├── run_macbench.py     # Run entry point
├── monitor_macbench.py # Read-only run monitor
├── requirements.txt
├── requirements_bench.txt
└── README.md
```

## Setup

First assemble the AgentNet run directory as described in the repository-root README, then enter the generated directory.

1. **Dependencies** (Python 3.11+, required by MAC-Bench):

```bash
python3.11 -m venv .venv311 && source .venv311/bin/activate
pip install -r requirements_bench.txt
pip install -e '../MAC_Bench-1.0[modal,agents]'   # Or install the published package with pip install mac-bench.
modal token new
mac-bench data download                            # Download tasks and container images.
```

2. **MAC_Bench-1.0 location**: defaults to `MAC_Bench-1.0` next to the package; override it with an environment variable:

```bash
export MAC_BENCH_ROOT=/path/to/MAC_Bench-1.0
```

3. **Configure `.env`** (in the package root):

```
LLM_BASE_URL=...
LLM_API_KEY=...
LLM_MODEL=deepseek-v4.1-flash
LLM_PRICE_INPUT_PER_1M=...
LLM_PRICE_OUTPUT_PER_1M=...
```

## Run

```bash
python run_macbench.py --task-ids TASK_ID                           # Replace with an available task ID.
python run_macbench.py --split test                                  # Run the full test split.
python run_macbench.py --split test --warmup-dev                     # Evolve on 6 dev tasks before testing.
```

Outputs are saved under `save/macbench_<timestamp>/`: `results.jsonl` (per-task PS/RR, elapsed time, tokens, and cost) and `summary.json`.

Default task budgets are read from the installed MAC-Bench configuration and can be overridden with the runner options.

## Notes

- The official MAC-Bench metrics are **PS (Progress Score)** and **RR (Resolution Rate)**.
- Model requests use `context.client` and count toward MAC-Bench budgets and token statistics. Actions use `context.environment.call`.
- The assembly script copies shared components from `methods/agentnet/` into the run package. The `gaia/` directory retains modules required for compatible AgentNet imports.
