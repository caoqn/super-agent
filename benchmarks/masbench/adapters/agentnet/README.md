# AgentNet × MAS-Bench

AgentNet run package for the three MAS-Bench task families: Reasoning, Scheduling, and Game.

Includes the AgentNet LLM client, official MAS-Bench code, adapter scripts, and a 90-task sample manifest (seed=42).

## Directory layout

```
benchmarks/masbench/adapters/agentnet/
├── Reasoning/            # Official MASBench Reasoning code and adapter
│   ├── src/              #   data_loader / prompts / metrics / conversation / agents
│   ├── api_key_loader.py
│   └── run_agentnet_reasoning.py   # Run Reasoning with the AgentNet LLM client
├── Scheduling/
│   └── simulate_meeting_qwen.py    # Official Scheduling runner (thinking disabled)
├── Game/
│   ├── eval.py           # Official Game entry point
│   ├── masbench/         #   Game framework
│   └── framework/        #   Framework layer
├── MASBench_selected_30/ # 90 tasks prepared as described in the repository README (30 per family)
│   ├── Reasoning/reasoning_subset.json
│   ├── Scheduling/scenarios/
│   ├── Game/maps/
│   └── selection_manifest.json
├── run_scheduling_30.py  # Run 30 Scheduling scenarios
├── run_game_30.py        # Run 30 Game maps
├── run_game_one_token.py # Run one map and collect input/output token counts
├── pick_30_each.py       # Resample 90 tasks from the complete dataset (requires full data)
└── README.md
```

Shared AgentNet clients, environment-variable handling, and token accounting are in `methods/agentnet/`; the Reasoning runner imports those modules directly. They are not part of the MASBench adapter directory.

## Setup

1. **Dependencies** (Python 3.10+):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install requests openai pydantic PyYAML python-dotenv
```

2. **Configure `.env`** (in the repository root or package root):

```
LLM_BASE_URL=...
LLM_API_KEY=...
LLM_MODEL=deepseek-v4.1-flash
```

## Run

After preparing the sampled data as described in the repository-root README, run these commands from the repository root:

```bash
# Reasoning (30 tasks, F1 metric)
python benchmarks/masbench/adapters/agentnet/Reasoning/run_agentnet_reasoning.py --num-samples -1 --seed 42

# Scheduling (30 scenarios, success-rate metric)
python benchmarks/masbench/adapters/agentnet/run_scheduling_30.py

# Game (30 maps, delivery-score metric)
python benchmarks/masbench/adapters/agentnet/run_game_30.py
```

Results:
- Reasoning: `Reasoning/results_agentnet_<timestamp>.jsonl`
- Scheduling: `MASBench_selected_30/scheduling_results.jsonl`
- Game: `MASBench_selected_30/game_results.jsonl`

## Official metrics

| Task | Metric |
|---|---|
| Reasoning | F1 (0–1, `cal_f1_score`) |
| Scheduling | Success rate (exact slot match against `task_plans` solutions) |
| Game | Delivery score (`sum(min(item count on table, target))`, maximum 100) |

## Notes

- `pick_30_each.py` requires the complete data source (`Reasoning/reasoning.json`, `Scheduling/meeting/`, and `Game/custom_map/`), which is not included. Download the data from official MASBench and generate `MASBench_selected_30/` as described in the repository-root README.
- Run the official Game entry point from the `Game/` directory; `eval.py` writes logs to `Game/.preset_logs/`.
