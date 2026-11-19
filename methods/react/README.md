# ReAct method core

`methods/react/react.py` implements the reusable Thought/Action/Observation loop and action parser. `methods/react/llm.py` provides the OpenAI-compatible model client used by the released ReAct baseline.

The MASBench-specific tools, prompts, data loading, and evaluation runners live in `benchmarks/masbench/adapters/react/`.
