# EvoTeam 评测适配器

方法源码：[caoqn/EvoTeam 的 individual-benchmark-v1 分支](https://github.com/caoqn/EvoTeam/tree/individual-benchmark-v1)。此分支提供三个适配器使用的单数据集实验接口；源码仓库的 `main` 保持混合数据集实现。本评测仓库只保存适配器、实验配置、初始池和源码版本锁，不包含方法本体、冻结池或演化产物。

## 安装固定源码版本

使用 Python 3.11，在本评测仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python scripts/setup_evoteam.py
cp .env.example .env
# 编辑 .env 中的 DEV_API_BASE / DEV_API_KEY。
export META_TEAM_ENV_FILE="$PWD/.env"
```

安装器从公开 GitHub 下载 `source-lock.json` 指定的确切提交，校验每个源码文件的 SHA-256，并安装运行依赖；不需要 GitHub 登录。默认放入被 Git 忽略的 `external/EvoTeam/`。可加 `--no-install` 只下载，或 `--destination` 指定位置。非默认位置需在 shell 中导出 `EVOTEAM_ROOT`，仅写入 `.env` 不会改变源码解析路径。已有目录不覆盖；不兼容或被修改的源码会明确报错。

模型默认为 `deepseek-chat`，三个 benchmark 适配器位于 `benchmarks/{gaia,macbench,masbench}/evoteam/`；`methods/Evoteam/` 只保留公共源码引导、版本锁和说明。三个训练配置均为 `workers=1`、开启 `--evolve`。显式 `META_TEAM_ENV_FILE` 中的模型变量可能覆盖配置默认值，请核对 `.env` 中的 `META_TEAM_MODEL`、`TEAM_SELECTOR_MODEL` 和 `TEAM_REFLECTION_MODEL`。数据路径需通过 shell `export` 设置。

## 准备数据

数据及参考答案不随代码发布，需自行获取对应上游数据的使用权限。

**GAIA**：获取 [官方 GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) 的 `2023/validation/`（`metadata.jsonl` 和附件），再执行：

```bash
python scripts/prepare_evoteam_gaia.py --source "$GAIA_VALIDATION_DIR"
```

此脚本依据 `benchmarks/gaia/evoteam/splits.json` 中的固定任务 ID 重建 20 题训练集及 100 题测试集。默认输出 `data/gaia`；使用其他位置时导出 `GAIA_DATA_DIR`，其中需包含 `level_{1,2,3}_{train_20,test_100}.json` 和 `val_files/`。不改变其他方法的 GAIA 划分。

**MAC-Bench**：获取包含 `data/`、`src/mac_bench/`、`configs/evaluation/task_budgets.json` 的 [MAC-Bench](https://github.com/rzchen777-wq/MAC_Bench) 源码及数据，并设置：

```bash
export MAC_BENCH_ROOT="$PWD/external/MAC_Bench"
python -m pip install -e "${MAC_BENCH_ROOT}[meta_team]"
```

训练使用 `dev`，测试使用 `test`。保留上游任务的 token、工具调用和时间预算以及主席紧急提交；agent 步数上限关闭。原生工具环境还须满足 MAC-Bench 官方说明（例如对应任务的外部命令）。

**MASBench**：获取 [官方 MASBench](https://github.com/BUPT-GAMMA/MASBench)，包含 Reasoning、Scheduling、Game 数据：

```bash
export MASBENCH_ROOT="$PWD/external/MASBench"
python scripts/prepare_masbench.py --source "$MASBENCH_ROOT"
export MASBENCH_SUPPORT_ROOT="$PWD/external/masbench-support"
python -m pip install --target "$MASBENCH_SUPPORT_ROOT/.masbench_deps" -r requirements/masbench-game.txt
```

固定训练 18 题、测试 90 题。自定义划分输出位置需导出 `MASBENCH_SPLIT_DIR`。Game 子进程使用独立支持目录及依赖。

## 从初始池训练

```bash
python scripts/run.py --config configs/Evoteam_gaia_train.json
python scripts/run.py --config configs/Evoteam_macbench_train.json
python scripts/run.py --config configs/Evoteam_masbench_train.json
```

启动前可追加 `--dry-run --max-items 1`，检查数据选择且不调用模型。追加 CLI 参数覆盖默认参数，例如 `--run-id my_mac_train` 或 `--cases 0`。初始池位于 `benchmarks/{gaia,macbench,masbench}/evoteam/pools/initial`；来源及文件校验和见 `initial_pools.json`。

训练复制初始池到运行目录的 `team/v000`，再保存演化版本；后续题目使用最新版本，初始快照不会被修改。默认输出 `results/Evoteam/`，可导出 `EVAL_RESULTS_DIR` 重定向。通过 `--run-id <已有运行名> --resume` 续跑；具体参数可用对应入口的 `--help` 查看。

`configs/Evoteam_{gaia,macbench,masbench}.json` 保留原实验测试参数，引用的冻结池不提供。训练后用 `--team <训练产出的完整池目录>` 指定测试池。测试不加 `--evolve`；未经对应训练的初始池不能视为原实验冻结池。

## 验证范围

```bash
python -m pip install -r requirements/dev.txt
python -m pytest -q tests/test_evoteam_source.py tests/test_evoteam_training.py tests/test_evoteam_budget.py
python -m pytest -q external/EvoTeam/tests
```

三套初始池的离线训练测试使用真实选队、协作运行、反思与版本保存逻辑，替换模型回答和评测环境，验证第一题生成版本、第二题复用、初始池不变。预算回归覆盖 token 超限和紧急提交。源码已安装时自动启用集成测试；真实数据测试还需设置上述数据环境。离线测试不调用付费模型，也不代表完整基准得分复现。
