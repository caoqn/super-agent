# 依赖快照与离线检查

`python-current-snapshot.txt` 是当前实验 Python 3.11.15 环境中 `pip freeze` 的输出，采集时 `pip check` 通过。GAIA 和 MASBench 的历史启动脚本默认指向该 Python 环境；MACBench 历史执行环境尚未独立确认。该环境可能在实验后发生变化，因此这是**当前快照**，不是历史精确版本，也不是完整可移植锁文件。两个指向本机 Meta-Team、LoCoBench 的 editable 安装已排除；它们不是本次三个适配器的发布依赖。快照含该环境中其他项目的包，未证明每个包都为 Skill-MAS 必需。不同操作系统/架构上版本可能不可用；安装前请自行核对。

外部依赖仍需单独固定：GAIA 原始数据及其附件；MASBench 代码、数据和 Game 运行环境（本地额外安装了 pulp 3.3.2）；MACBench harness、数据、服务及 Docker 镜像。`required-artifacts.json` 列出了尚未填实的来源、授权条款、revision 和哈希材料，不应把空字段当成已验证信息。不能把 MACBench 数据 world revision 当成 harness 代码版本。各上游来源及许可证、镜像可获取性、平台与系统依赖需要在发布前另行核验。

本次离线检查用固定 ZIP 和 SHA-256 重新准备源码，检查三个 train/test 划分；通过 `offline_entry_check.py` 验证真实适配器加载、调用参数、配置和 frozen skill。该脚本使用 stub 代替训练/测试过程，网络与 API 未调用。复查示例（替换实际路径及各参数）：

```bash
SKILLMAS_TEST_UPSTREAM=/path/to/work/Skill_MAS \
SKILLMAS_TEST_BENCHMARK=gaia SKILLMAS_TEST_PHASE=test \
SKILLMAS_TEST_DATA_ROOT=/path/to/gaia \
python methods/skillmas/offline_entry_check.py
```

`SKILLMAS_TEST_BENCHMARK` 可为 `gaia`、`masbench` 或 `macbench`，`SKILLMAS_TEST_PHASE` 为 `train` 或 `test`。运行前需有上述数据及适配器可导入的基准依赖；这不等同完整基准运行或真实费用验证。
