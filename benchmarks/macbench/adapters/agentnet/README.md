# AgentNet × MAC-Bench

AgentNet 在 MAC-Bench 上的基准适配器源码。运行包由仓库根目录的 `scripts/prepare_agentnet.py` 组装生成。

## 目录结构

```
agentnet_macbench_run/（生成后的运行目录）
├── src/                # AgentNet 核心
├── models/             # AgentNet 模型
├── prompt/             # prompt 集
├── config/             # 配置
├── evaluator/          # 评测器
├── utils/              # 工具
├── gaia/               # MAC-Bench 适配层（macbench_adapter.py 等）及上游兼容模块
├── methods/agentnet/   # 仓库 methods/agentnet/ 中的共享方法组件
├── run_macbench.py     # 运行入口
├── monitor_macbench.py # 运行监视器（只读）
├── requirements.txt
├── requirements_bench.txt
└── README.md
```

## 准备

先按仓库根目录 README 组装 AgentNet 运行目录，再进入对应的运行目录。

1. **依赖**（Python 3.11+，MAC-Bench 要求）：

```bash
python3.11 -m venv .venv311 && source .venv311/bin/activate
pip install -r requirements_bench.txt
pip install -e '../MAC_Bench-1.0[modal,agents]'   # 或 pip install mac-bench
modal token new
mac-bench data download                            # 下载任务与镜像
```

2. **MAC_Bench-1.0 位置**：默认取本包父目录下的 `MAC_Bench-1.0`，也可用环境变量覆盖：

```bash
export MAC_BENCH_ROOT=/path/to/MAC_Bench-1.0
```

3. **配置 `.env`**（本包根目录）：

```
LLM_BASE_URL=...
LLM_API_KEY=...
LLM_MODEL=deepseek-v4.1-flash
LLM_PRICE_INPUT_PER_1M=...
LLM_PRICE_OUTPUT_PER_1M=...
```

## 运行

```bash
python run_macbench.py --task-ids op26_service_administration_007   # 先跑 1 个 dev 任务
python run_macbench.py --split test                                  # 全部 test 任务
python run_macbench.py --split test --warmup-dev                     # 先用 6 个 dev 任务进化
```

输出在 `save/macbench_<时间>/`：`results.jsonl`（每任务 PS/RR/耗时/token/成本）、`summary.json`。

默认预算与官方一致：1M tokens / 80 次模型调用 / 160 次工具调用 / 2400 秒。

## 说明

- MAC-Bench 的官方指标是 **PS（Progress Score）** 与 **RR（Resolution Rate）**。
- 模型请求走 `context.client`（计入 MAC-Bench 预算与 token 统计），动作走 `context.environment.call`。
- `methods/agentnet/` 中的共享实现由组装脚本带入运行包；`gaia/` 中保留上游 AgentNet 兼容导入所需的模块。
