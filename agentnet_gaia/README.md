# AgentNet × GAIA

AgentNet 框架在 GAIA benchmark 上的完整运行包（通过补丁适配 AgentNet 核心代码 + GAIA 适配层）。

## 目录结构

```
agentnet_gaia/
├── src/                # AgentNet 核心（agentgraph / experiment / pool 等）
├── models/             # AgentNet 模型（agent_dag / experience_pool）
├── prompt/             # prompt 集
├── config/             # 配置（setting.py + experiment/*.yaml）
├── evaluator/          # 评测器
├── utils/              # 执行器 / 结果提取等工具
├── gaia/               # GAIA 适配层（dataset / executor / llm / tools / scorer 等）
├── run_gaia.py         # 运行入口
├── run_gaia.sh         # 一键脚本（建 venv + 装依赖 + 冒烟 + 全量）
├── monitor_gaia.py     # 运行监视器（只读）
├── requirements.txt          # 完整依赖
├── requirements_bench.txt    # benchmark 运行所需最小依赖
└── README.md
```

## 准备

先按仓库根目录 README 组装 AgentNet 运行目录，再进入对应的运行目录。

1. **依赖**（Python 3.10+）：

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements_bench.txt
```

2. **GAIA 数据集**（需 Hugging Face 授权）：

```bash
huggingface-cli download gaia-benchmark/GAIA --repo-type dataset --local-dir 2023
```

放在本包根目录下，使结构为 `agentnet_gaia_run/2023/validation/metadata.parquet`。

3. **配置 `.env`**（本包根目录）：

```
LLM_BASE_URL=...
LLM_API_KEY=...
LLM_MODEL=deepseek-v4.1-flash
LLM_PRICE_INPUT_PER_1M=...
LLM_PRICE_OUTPUT_PER_1M=...
SERPER_API_KEY=...    # 或 TAVILY_API_KEY，搜索工具用
```

## 运行

```bash
python run_gaia.py --limit 3                 # 冒烟 3 题
python run_gaia.py                            # 全量 validation 165 题
python run_gaia.py --levels 1                 # 只跑 L1
python run_gaia.py --start 50 --limit 20      # 从第 50 题起跑 20 题
python run_gaia.py --resume save/gaia_xxx     # 中断续跑
```

输出在 `save/gaia_<时间>/`：`summary.json`（准确率/token/耗时/成本）、`results.jsonl`（逐题）、`calls.jsonl`（每次 LLM/工具调用明细）。

## 说明

- 图片题需 `LLM_SUPPORTS_VISION=1`；音频题需 `AUDIO_TRANSCRIBE_MODEL`。
- 搜索默认用免 key 的 DuckDuckGo，建议填 Serper/Tavily key。
- 评分用 GAIA 官方 quasi-exact-match。
