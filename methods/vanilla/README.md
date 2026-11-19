# Vanilla method

Vanilla is the single-call baseline: it sends the task prompt to the model once, without a ReAct loop or tool calls. Its OpenAI-compatible client is in `methods/vanilla/llm.py`.

MASBench task-specific prompts, input handling, scoring, and the runner are in `benchmarks/masbench/adapters/vanilla/`.
