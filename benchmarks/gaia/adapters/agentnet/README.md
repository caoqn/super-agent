# AgentNet × GAIA

This is the GAIA benchmark adapter for AgentNet. The run package is assembled by `scripts/prepare_agentnet.py` from the repository root.

## Directory layout

```
agentnet_gaia_run/ (generated run directory)
├── src/                # AgentNet core (agentgraph / experiment / pool, etc.)
├── models/             # AgentNet models (agent_dag / experience_pool)
├── prompt/             # Prompt collection
├── config/             # Configuration (setting.py + experiment/*.yaml)
├── evaluator/          # Evaluators
├── utils/              # Executors, result extraction, and other utilities
├── gaia/               # GAIA integration (dataset / executor / llm / tools / scorer, etc.)
├── methods/agentnet/   # Shared method components from methods/agentnet/
├── run_gaia.py         # Run entry point
├── run_gaia.sh         # One-command setup, dependency installation, smoke test, and full run
├── monitor_gaia.py     # Read-only run monitor
├── requirements.txt          # Full dependencies
├── requirements_bench.txt    # Minimum benchmark runtime dependencies
└── README.md
```

## Setup

First assemble the AgentNet run directory as described in the repository-root README, then enter the generated directory.

1. **Dependencies** (Python 3.10+):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements_bench.txt
```

2. **GAIA dataset** (Hugging Face access required):

```bash
huggingface-cli download gaia-benchmark/GAIA --repo-type dataset --local-dir 2023
```

Place the data under the package root so that `agentnet_gaia_run/2023/validation/metadata.parquet` exists.

3. **Configure `.env`** (in the package root):

```
LLM_BASE_URL=...
LLM_API_KEY=...
LLM_MODEL=deepseek-v4.1-flash
LLM_PRICE_INPUT_PER_1M=...
LLM_PRICE_OUTPUT_PER_1M=...
SERPER_API_KEY=...    # Or TAVILY_API_KEY for search tools.
```

## Run

```bash
python run_gaia.py --limit 3                 # Smoke test on 3 tasks.
python run_gaia.py                            # Run all 165 validation tasks.
python run_gaia.py --levels 1                 # Run Level 1 only.
python run_gaia.py --start 50 --limit 20      # Run 20 tasks starting at index 50.
python run_gaia.py --resume save/gaia_xxx     # Resume an interrupted run.
```

Outputs are saved under `save/gaia_<timestamp>/`: `summary.json` (accuracy, tokens, elapsed time, and cost), `results.jsonl` (per-task results), and `calls.jsonl` (details for each LLM and tool call).

## Notes

- Set `LLM_SUPPORTS_VISION=1` for image tasks and configure `AUDIO_TRANSCRIBE_MODEL` for audio tasks.
- Search uses DuckDuckGo without an API key by default; a Serper or Tavily key is recommended.
- Scoring uses the official GAIA quasi-exact-match metric.
- The assembly script copies shared components from `methods/agentnet/` into the run package. The `gaia/` directory retains modules required for compatible AgentNet imports.
