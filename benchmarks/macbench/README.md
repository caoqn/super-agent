# MAC-Bench

Install and download data from the [official MAC-Bench repository](https://github.com/rzchen777-wq/MAC_Bench). The AgentNet adapter is in `adapters/agentnet/`. For AgentNet, install the official MAC-Bench Python package separately into the adapter's Python 3.11 environment, then download its task data and environment images with `mac-bench data download`; the adapter README has pinned install commands. Keep the external checkout, benchmark data, and runtime images outside Git.


## Super Agent adapter

MAC-Bench 使用官方任务环境和评分器，主要指标是 Progress Score（PS）和 Resolution Rate（RR）。本目录不包含任务数据或官方评分实现；请使用独立的 [MAC-Bench 官方仓库](https://github.com/rzchen777-wq/MAC_Bench)，固定其代码及数据版本。

`benchmarks/macbench/adapters/super_agent/adapter.py` 调用 MAC-Bench 官方 `superagent` 基线和评测器。配置文件位于 `methods/super_agent/config/macbench.yaml`，默认会传给官方评测器；它记录模型连接、统一 backbone、共享 episode world 和单次生成上限；任务预算通过命令行设置。

官方要求 Linux、Python 3.11+、Node.js 24+、pnpm，以及 Modal 或 Daytona 环境。请根据官方文档安装 `.[modal,agents,superagent]` 或对应 Daytona 依赖，并在官方 Superagent 目录安装 pnpm 依赖；配置模型和环境凭据并下载数据。示例：

```sh
python benchmarks/macbench/adapters/super_agent/adapter.py \
  --upstream /path/to/MAC_Bench --python /path/to/MAC_Bench/.venv/bin/python \
  --task-id op26_service_administration_007 --output-dir results/macbench-dev \
  --max-seconds 7200 --method-config methods/super_agent/config/macbench.yaml --dry-run
```

去掉 `--dry-run` 才会实际评测。上游目录、环境、数据及凭据目前未配置；本入口尚未完成端到端验证。最终 PS/RR、用量及轨迹以官方输出为准。

MAC-Bench 的官方 `superagent` 实现自带其任务环境、router、worker、subagent 和评分器；本适配器只负责固定上游 checkout、传递 YAML 配置和预算，并保存官方输出。
