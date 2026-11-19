# Vanilla Agent 适配层

MASBench 90 题的 **Vanilla（原装智能体）基线** 实现。

## 口径

Vanilla = **单轮 LLM 调用、无 ReAct 循环、无工具调用**。把任务相关信息一次性全部放进 prompt，模型直接输出最终答案/计划：

- **Reasoning**：两方证据 + 问题一次性给出 → 直接作答 → 官方 `cal_f1_score`
- **Scheduling**：5 人空闲段一次性给出 → 直接出槽位 → 官方槽位精确匹配
- **Game**：完整地图一次性给出 → 输出整套动作计划 → 重放进 `GridWorld` 物理回放 → `sum(min(交付,目标))`

评分口径与 ReAct **完全一致**（纯「机制」差异对比）。

## 文件结构

```
methods/vanilla/masbench/
├── run_vanilla.py   # Vanilla 主实现（三个任务各一个 run_*_one 函数）
├── llm.py           # OpenAI 兼容客户端（独立副本，读取仓库根目录 .env）
├── README.md
└── results/         # 90 题逐题结果
    ├── vanilla_reasoning_results.jsonl
    ├── vanilla_scheduling_results.jsonl
    └── vanilla_game_results.jsonl
```

## 依赖

评分函数**复用** `../../react/masbench/` 里的三个 runner（`run_reasoning.py` / `run_scheduling.py` / `run_game.py`），以保证 Vanilla 和 ReAct 用同一套评分口径：

- `run_reasoning.py` 的 `_fmt_evidence`、`cal_f1_score`（官方）
- `run_scheduling.py` 的 `_free_gaps`、`_evaluate`、`_min_to_hhmm`
- `run_game.py` 的 `GridWorld`

`run_vanilla.py` 通过 `sys.path` 引用上述模块，不复制评分逻辑。

## 运行

```bash
cd methods/vanilla/masbench
python3 run_vanilla.py reasoning    # Reasoning（30 题，单轮）
python3 run_vanilla.py scheduling   # Scheduling（30 场景，单轮）
python3 run_vanilla.py game         # Game（30 地图，单轮）
```

带断点续跑：结果文件已存在时自动跳过已完成的题。

## 结果

见 `results/vanilla_*.jsonl`。
