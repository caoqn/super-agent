# Skill-MAS 复现实验适配

本目录提供 [Skill-MAS](https://github.com/linhh29/Skill_MAS) 固定源码引用、本地兼容补丁、GAIA/MASBench/MACBench 适配器、划分清单和历史 frozen skill。上游源码运行时单独准备，不复制进本仓库。`upstream.json` 的 commit 来自原始源码 ZIP 注释，ZIP SHA-256 用于校验归档；commit 尚未通过在线 Git 记录独立核实。`patches/runtime.patch` 面向该归档生成。它不能证明历史实验运行时源码与固定快照完全一致；MACBench 结果目录映射也作过便携化修正。

## 运行环境

建议 Python 3.11。`environment/python-current-snapshot.txt` 是实验机器当前 `pip freeze`，不是历史运行时锁文件，也不是干净环境的最小 requirements。MACBench 的历史 Python 环境未能确认。发布前如需严格复现，应在目标平台新建虚拟环境、按上游和 benchmark 文档安装，并生成带 Python/OS/架构信息的独立锁文件。MASBench Game 额外依赖列于 `requirements/masbench-game.txt`；安装时需确认游戏依赖版本。MACBench 需要其 harness、任务服务和 Docker 环境；本仓库尚未记录历史镜像 digest。

## 数据准备

数据与任务包不包含在本仓库。数据下载、认证和许可遵循各 benchmark 官方项目。

- **GAIA**：`--data-root` 目录须包含 `level_1_train_20.json`、`level_2_train_20.json`、`level_3_train_20.json` 与对应的 `level_{1,2,3}_test_100.json`。冻结清单中的训练 20 题、测试 100 题跨 level 汇总；附件应放在 `val_files/` 下，并保持 GAIA 原始附件布局。仓库没有 GAIA 原始数据文件和附件哈希，使用者需从获准来源准备并核对数据版本。
- **MASBench**：`--data-root` 指向包含 `Reasoning/reasoning.json`、`Scheduling/`、`Game/custom_map/` 的根目录。仓库提供 18 条训练和 90 条测试 ID manifest；reasoning 划分排除测试题族，审计记录见 `splits/masbench/split_audit.json`。本地 MASBench 源目录无法提供干净 Git revision（Git 状态含大量未跟踪文件），需自行留存源文件哈希。运行脚本验证 ID 和顺序，缺失数据会失败。
- **MACBench**：`--data-root` 指向任务 catalogue 数据目录或项目根目录。本仓库固定 48 个测试 ID、各 task world revision 及训练/测试设置。数据准备后需确认 `mac-bench` CLI、相关服务和 Docker 可用。已知数据 revision `425592652fc225292903c055e2c3a595b061a4a7` 不是 harness 代码版本或镜像 digest，后二者的历史记录缺失。

## 准备并运行

在仓库根目录执行。归档须为固定上游版本 ZIP；省略 `--archive` 时准备脚本从固定 Git commit 克隆。目标目录必须是新建的 `Skill_MAS` 目录。

```bash
python3 -m venv .venv-skillmas
source .venv-skillmas/bin/activate
python -m pip install -r requirements/masbench-game.txt
# 依照 Skill-MAS、GAIA、MACBench 官方说明安装其他 Python/系统依赖。
python scripts/prepare_skillmas.py --destination /path/to/work/Skill_MAS --archive /path/to/Skill_MAS-main.zip

export OPENAI_API_KEY='your-key'
export OPENAI_BASE_URL='https://your-openai-compatible-endpoint/v1'

# 检查路径及题目清单，不发 API 请求
python scripts/run_skillmas.py --upstream /path/to/work/Skill_MAS --benchmark gaia --phase train --data-root /path/to/gaia --output /path/to/results --run-id gaia-k3 --validate-only
# 训练后以产出的 SKILL.md 测试
python scripts/run_skillmas.py --upstream /path/to/work/Skill_MAS --benchmark gaia --phase train --data-root /path/to/gaia --output /path/to/results --run-id gaia-k3
python scripts/run_skillmas.py --upstream /path/to/work/Skill_MAS --benchmark gaia --phase test --data-root /path/to/gaia --output /path/to/results --run-id gaia-test --skill /path/to/SKILL.md
```

将 `gaia` 替换为 `masbench` 或 `macbench`，并指定对应数据根目录。确保 API 服务端模型 ID 与配置 `deepseek-flash` 一致。各实验均为一轮进化：GAIA 20 题、每题 k=3；MASBench 18 题、每题 k=3；MACBench 6 个 dev 题、每题 k=2。测试集分别 100、90、48 题。并发和 timeout 参数见 `config/`。训练测试使用不同 run-id；恢复同一 run 时需保持配置不变。结果目录保存有效配置和 skill 哈希。

历史 frozen skill 位于 `artifacts/{gaia,masbench,macbench}/SKILL.md`，SHA-256 见 `experiment_evidence.json`。测试历史结果时可将对应文件作为 `--skill`。MACBench 的另一组 manager/worker 实验不是这里的 Skill-MAS frozen-skill 测试。价格配置只用于估算，不代表服务账单。

## 预算、限制与验证状态

训练的 `max_steps=300` 由 Skill-MAS pipeline 使用；测试工具轮数默认 `max_rounds=12`。timeout 为基础时间加 800 秒墙钟窗口，不是精确暂停 API 等待的双时钟。`max_output_tokens` 是单次响应限制，不是单题累计 token 限制。适配器不强制 MACBench 官方 token/call 预算，MACBench 当前 `admit_call` 也未执行调用计数。因此若研究要求遵守官方预算，应先补齐可审计的预算执行器。

本地检查通过固定 ZIP 校验和准备、六个 train/test 数据划分加载、ID/顺序核对，以及用 stub 替代执行的入口检查。没有发 API 请求，也未在干净机器完成 Docker、外部服务和真实 benchmark 的端到端测试。严格复现仍缺历史完整 Python 锁、MACBench harness commit 和镜像 digest、GAIA 数据与附件哈希，以及 MASBench 干净源码 revision。检查命令见 `environment/README.md`。
