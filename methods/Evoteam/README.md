# 多方法评测代码：整理导出

这是一份从 `meta_team evolution/new code` 整理出的独立代码包。`methods/Evoteam` 存放 Evoteam 方法，代码保留主席选队、成员协作、任务后多层反思及紧急提交；三个数据集采用原适配器。未包含其他方法实现，合并到已有仓库时不要覆盖已有的 `methods/` 或 `benchmarks/` 文件。

```text
methods/Evoteam/{gaia,macbench,masbench}/  适配器与入口
methods/Evoteam/runtime/                  方法运行时、工具、提示词与池子
benchmarks/{gaia,macbench,masbench}/  共用数据读取与评分组件
configs/                              示例运行参数
scripts/                              启动及数据准备脚本
requirements/                         Python 依赖
results/                              本地输出（默认不提交）
```

评分与方法适配仍有耦合：部分 MASBench 环境与 game 评分留在 `methods/Evoteam/masbench/adapter.py`，MAC-Bench 评分依赖原生包。不能把此包宣称为已实现其他方法的统一评分框架；接入其他方法时应核对环境、数据划分和评分口径。

## 安装与 API

建议 Python 3.11，创建虚拟环境并执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/base.txt
cp .env.example .env
# 填写 DEV_API_BASE 和 DEV_API_KEY
```

配置的 `META_TEAM_MODEL`、`TEAM_SELECTOR_MODEL`、`TEAM_REFLECTION_MODEL` 分别控制执行、选队和反思模型；请确认 API 服务商支持该模型名。示例配置显式使用 `deepseek-chat`；改用其他模型时修改对应 JSON。`.env` 和 API 密钥不应提交。使用外部 env 文件可设置 `META_TEAM_ENV_FILE`；此方式会覆盖已经导出的同名变量，包括模型变量，使用前检查其中配置。

## 数据准备

自行取得数据和上游源码，遵循其授权；数据、答案、凭据和实验记录不在此导出包内。

- GAIA：将原实验使用的 `level_{1,2,3}_{train_20,test_100}.json` 等划分文件和 `val_files/` 放到 `data/gaia/`，或设置 `GAIA_DATA_DIR`。这些文件并非通用 GAIA 原始下载格式；划分须与实验一致。
- MAC-Bench：设置 `MAC_BENCH_ROOT` 指向含 `data/`、`src/mac_bench/` 和 `configs/evaluation/task_budgets.json` 的源码目录，在当前虚拟环境中安装上游核心包及本适配器需要的 Agent extras：`pip install -e "$MAC_BENCH_ROOT[meta_team]"`。
- MASBench：设置 `MASBENCH_ROOT` 指向原始 `MASBench-main`，然后运行下列命令基于固定题号生成数据。Game 还需设置 `MASBENCH_SUPPORT_ROOT`，指向一个可写目录。按以下命令把隔离依赖安装到 `.masbench_deps/`；此目录由使用者准备，不要提交。

```bash
export MAC_BENCH_ROOT=/path/to/MAC_Bench-main
export MASBENCH_ROOT=/path/to/MASBench-main
export MASBENCH_SUPPORT_ROOT=/path/to/masbench-support
mkdir -p "$MASBENCH_SUPPORT_ROOT/.masbench_deps"
python -m pip install --target "$MASBENCH_SUPPORT_ROOT/.masbench_deps" -r requirements/masbench-game.txt
python scripts/prepare_masbench.py --source "$MASBENCH_ROOT"
```

`benchmarks/masbench/` 提供原实验固定的 train 18 / test 90 题号，生成目录含 Reasoning 答案，只用于评分和环境初始化，不应提交或暴露给执行代理。

## 运行

从仓库根目录执行；`--dry-run` 只列题，不调用模型。请在实际运行前先检查池子、模型、划分和参数：

```bash
python scripts/run.py --config configs/Evoteam_gaia.json --dry-run --cases 0
python scripts/run.py --config configs/Evoteam_macbench.json --dry-run --max-items 1
python scripts/run.py --config configs/Evoteam_masbench.json --dry-run --max-items 1
python scripts/run.py --config configs/Evoteam_macbench.json --cases 0
```

后附同名 CLI 参数可覆盖示例值。运行时训练加 `--evolve`，如 `--split dev --evolve --team pool_MACBENCH_OURS_INITIAL`；测试使用已冻结的团队。具体参数通过 `python methods/Evoteam/macbench/run.py --help` 查看。示例池：GAIA 为原指定池，MAC-Bench 包含 initial 和 V006 test，MASBench 为 dynamic 6；GAIA 池名称带 test，运行前请确认其训练来源和数据隔离。进化可能将更新持久化到传入的池子，先复制池子以保留初始版本；不要对测试集使用 `--evolve`。

MAC-Bench 的单题限制从上游 `task_budgets.json` 读取，涵盖 token、模型/工具调用、环境动作及时间；预算耗尽或时间临近时触发尽力提交，能否实际完成仍受剩余预算及 API 状态影响。示例配置关闭了 agent 步数上限（`unlimited`），其余预算仍生效。并非所有 GAIA/MASBench 都使用 MAC-Bench 限额，具体设置见各适配器。

输出写入 `results/Evoteam/`，可通过 `EVAL_RESULTS_DIR` 修改。不要将轨迹、题目、答案或敏感信息直接上传。更多来源及许可说明见 [THIRD_PARTY.md](../../THIRD_PARTY.md)。

## 检查

```bash
pip install -r requirements/dev.txt
python -m pytest -q
```

未提供外部数据时，依赖原数据的集成测试会跳过。离线测试不等于实际调用 API 的端到端复现。首次公开前还应检查第三方许可证、各池训练来源与可发布性。
