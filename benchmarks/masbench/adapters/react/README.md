# ReAct × MASBench

The ReAct loop and model client are local method code in `methods/react/`; its MASBench-specific prompts, tools, input handling, and scoring runner are in this directory.

## Dependencies

Use Python 3.10 or newer. The ReAct model client uses Python's standard library, so it needs no separate Python package install. The Reasoning adapter imports the official scorer from an external `MASBench-main/Reasoning/` checkout. All three tasks also need the selected inputs prepared as described in [`benchmarks/masbench/README.md`](../../README.md).

Set the root `.env` with `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` as described in the repository README. Run from the repository root:

```bash
python benchmarks/masbench/adapters/react/run_reasoning.py
python benchmarks/masbench/adapters/react/run_scheduling.py
python benchmarks/masbench/adapters/react/run_game.py
python benchmarks/masbench/adapters/react/run_all.py
```
