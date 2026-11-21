# Vanilla Agent Adapter

The **Vanilla (unmodified agent) baseline** implementation for 90 MASBench tasks.

## Protocol

Vanilla uses **a single LLM call, with no ReAct loop or tool calls**. All task information is provided in one prompt, and the model directly returns its final answer or plan:

- **Reasoning**: Provide both evidence sets and the question, then score the answer with the official `cal_f1_score`.
- **Scheduling**: Provide the free slots for all five people, then score the selected slot with the official exact-match evaluator.
- **Game**: Provide the complete map, replay the full action plan in `GridWorld`, and score with `sum(min(delivered, target))`.

Scoring matches ReAct **exactly**, isolating the mechanism difference. The shared single-call model client is in `methods/vanilla/llm.py`.

## File layout

```
benchmarks/masbench/adapters/vanilla/
├── run_vanilla.py   # MASBench task adapter, prompts, and evaluation
└── README.md
```

## Dependencies

Scoring functions are **reused** from the three runners in `../react/` (`run_reasoning.py`, `run_scheduling.py`, and `run_game.py`) so Vanilla and ReAct use the same scoring protocol:

- `_fmt_evidence` and official `cal_f1_score` from `run_reasoning.py`
- `_free_gaps`, `_evaluate`, and `_min_to_hhmm` from `run_scheduling.py`
- `GridWorld` from `run_game.py`

`run_vanilla.py` imports these modules through `sys.path`; scoring logic is not duplicated.

## Run

```bash
cd benchmarks/masbench/adapters/vanilla
python3 run_vanilla.py reasoning    # Reasoning (30 tasks, single call)
python3 run_vanilla.py scheduling   # Scheduling (30 scenarios, single call)
python3 run_vanilla.py game         # Game (30 maps, single call)
```

Resume is automatic: tasks already present in the result file are skipped.

## Results

See `results/vanilla_*.jsonl`.
