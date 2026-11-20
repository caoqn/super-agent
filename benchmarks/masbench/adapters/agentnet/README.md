# AgentNet × MAS-Bench

AgentNet 在 MAS-Bench 三个任务（Reasoning / Scheduling / Game）上的运行包。

含 AgentNet 的 LLM 调用层 + MAS-Bench 官方代码 + 适配脚本 + 90 题的抽样清单（seed=42）。

## 目录结构

```
benchmarks/masbench/adapters/agentnet/
├── Reasoning/            # MASBench 官方 Reasoning 代码 + 适配脚本
│   ├── src/              #   data_loader / prompts / metrics / conversation / agents
│   ├── api_key_loader.py
│   └── run_agentnet_reasoning.py   # 用 AgentNet LLM 层跑 Reasoning
├── Scheduling/
│   └── simulate_meeting_qwen.py    # 官方 Scheduling runner（关思考）
├── Game/
│   ├── eval.py           # 官方 Game 入口
│   ├── masbench/         #   Game 框架
│   └── framework/        #   framework 层
├── MASBench_selected_30/ # 按根目录 README 准备的 90 题（各 30）
│   ├── Reasoning/reasoning_subset.json
│   ├── Scheduling/scenarios/
│   ├── Game/maps/
│   └── selection_manifest.json
├── run_scheduling_30.py  # 批量跑 Scheduling 30 场景
├── run_game_30.py        # 批量跑 Game 30 地图
├── run_game_one_token.py # 单地图，收集输入/输出 token 拆分
├── pick_30_each.py       # 从完整数据集重新抽取 90 题（需完整数据）
└── README.md
```

共享的 AgentNet 调用、环境变量和 token 统计实现位于 `methods/agentnet/`；Reasoning runner 直接导入这些模块。它们不属于 MASBench 适配器目录。

## 准备

1. **依赖**（Python 3.10+）：

```bash
python3 -m venv .venv-agentnet-masbench && source .venv-agentnet-masbench/bin/activate
python -m pip install -r benchmarks/masbench/adapters/agentnet/requirements.txt
```

依赖文件只列这三个 MASBench runner 实际使用的 Python 包。若使用内置 Gemini 客户端，再安装可选依赖：

```bash
python -m pip install -r benchmarks/masbench/adapters/agentnet/requirements-gemini.txt
```

MASBench 源码和任务数据需按仓库根目录说明另行获取；不需要克隆完整 AgentNet 仓库才能运行这里的适配器。

2. **配置 `.env`**（仓库根目录，或本包根目录）：

```
LLM_BASE_URL=...
LLM_API_KEY=...
LLM_MODEL=deepseek-v4.1-flash
```

## 运行

按仓库根目录 README 准备抽样数据后，从仓库根目录执行：

```bash
# Reasoning（30 题，F1 评分）
python benchmarks/masbench/adapters/agentnet/Reasoning/run_agentnet_reasoning.py --num-samples -1 --seed 42

# Scheduling（30 场景，成功率评分）
python benchmarks/masbench/adapters/agentnet/run_scheduling_30.py

# Game（30 地图，交付分评分）
python benchmarks/masbench/adapters/agentnet/run_game_30.py
```

结果输出：
- Reasoning：`Reasoning/results_agentnet_<时间戳>.jsonl`
- Scheduling：`MASBench_selected_30/scheduling_results.jsonl`
- Game：`MASBench_selected_30/game_results.jsonl`

## 指标口径（官方）

| 任务 | 指标 |
|---|---|
| Reasoning | F1（0–1，`cal_f1_score`） |
| Scheduling | 成功率（槽位精确匹配 `task_plans` 的 solutions） |
| Game | 交付分（`sum(min(桌上各类型数量, target))`，满分 100） |

## 说明

- `pick_30_each.py` 需要完整数据源（`Reasoning/reasoning.json`、`Scheduling/meeting/`、`Game/custom_map/`），未包含在本包；从官方 MASBench 下载数据后，按根目录 README 生成 `MASBench_selected_30/`。
- 直接运行官方 Game 入口时需在 `Game/` 目录下；`eval.py` 会写日志到 `Game/.preset_logs/`。
