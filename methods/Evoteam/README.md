# Evoteam 评测适配器

方法源码：[caoqn/EvoTeam](https://github.com/caoqn/EvoTeam)。本目录只保留 GAIA、MAC-Bench、MASBench 的适配器、启动入口和外部源码连接代码；方法核心和工具库在外部源码中维护；本仓库另外保存三个数据集的初始 agent 池（角色配置、提示词和初始协作规则），不保存冻结测试池或演化产物。实验配置位于 `configs/Evoteam_{gaia,macbench,masbench}.json`。

## 源码版本与安装

这些适配器对应原来的单数据集实验接口。已检查的公开源码提交 `a4b6920d02dd29c682e0b2bc4359c72c57ee0995` 是混合评测版本，缺少本实验所需的 `core/task_budget.py` 等接口，**不能直接用于复现本次实验**。目前没有确认过可用的公开提交，因此 `upstream.yaml` 的 `revision` 留空，不能将公开 `main` 视为已验证版本。

设置 `EVOTEAM_ROOT` 指向原实验的兼容源码目录，其中需要包含 `core/`、`benchmarks/adapter.py`、`main.py`、`tools/` 。启动时会检查必需接口，缺失时明确报错，不静默切换方法版本。

建议 Python 3.11，从评测仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/base.txt
# 将兼容源码放到 external/EvoTeam，或将此变量设为现有源码目录。
export EVOTEAM_ROOT="$PWD/external/EvoTeam"
cp .env.example .env
# 填写 DEV_API_BASE、DEV_API_KEY，并按实际位置填写 EVOTEAM_ROOT。
export META_TEAM_ENV_FILE="$PWD/.env"
```

`EVOTEAM_ROOT` 必须导出到进程环境；仅写入 `.env` 不会自动被连接代码读取。外部源码的附加依赖和工具需按其说明安装。`META_TEAM_ENV_FILE` 由历史方法运行时加载，文件中的同名模型变量可能覆盖 JSON 配置，运行前应核对。

## 数据及实验配置

- GAIA：设置 `GAIA_DATA_DIR`，目录包含原实验的 `level_{1,2,3}_{train_20,test_100}.json` 和 `val_files/`。配置使用 `pool_GAIA_pool_chairman_profile_change_r1_v020_test`、`test_100`。
- MAC-Bench：设置 `MAC_BENCH_ROOT`，包含 `data/`、`src/mac_bench/`、`configs/evaluation/task_budgets.json`；安装 `pip install -e "$MAC_BENCH_ROOT[meta_team]"`。配置使用 `pool_MACBENCH_OURS_V006_TEST`、`test`，关闭 agent 步数限制，保留数据集预算。
- MASBench：设置 `MASBENCH_ROOT`，执行 `python scripts/prepare_masbench.py --source "$MASBENCH_ROOT"` 生成固定 train 18 / test 90 划分。Game 还需设置可写的 `MASBENCH_SUPPORT_ROOT`，执行 `python -m pip install --target "$MASBENCH_SUPPORT_ROOT/.masbench_deps" -r requirements/masbench-game.txt`。配置使用 `pool_MASBENCH_DYNAMIC_6`、`test`。

初始池位于 `methods/Evoteam/{gaia,macbench,masbench}/pools/initial/`。来源池名和每个文件的 SHA-256 记录在 `initial_pools.json`。新增的 `configs/Evoteam_*_train.json` 分别使用 GAIA `train_20`、MAC-Bench `dev`、MASBench `train`，均开启 `--evolve`；启动器会将配置的仓库相对池路径解析为绝对路径再传入外部方法，不需要将池安装到源码目录。

原 `configs/Evoteam_{gaia,macbench,masbench}.json` 保留测试参数，其中引用的冻结池不随仓库提供，需自行放入 `$EVOTEAM_ROOT/agents/` 或用 `--team` 指定外部完整池路径。不要用初始池替代冻结池并声称复现相同测试成绩。

三个配置均使用 `deepseek-chat`，`workers=1`。MAC-Bench 预算从上游文件读取；token 预算和紧急提交需要兼容的外部方法实现，不能仅凭适配器加载成功认定已生效。

## 运行

```bash
python scripts/run.py --config configs/Evoteam_gaia.json --dry-run --cases 0
python scripts/run.py --config configs/Evoteam_macbench.json --dry-run --max-items 1
python scripts/run.py --config configs/Evoteam_masbench.json --dry-run --max-items 1
python scripts/run.py --config configs/Evoteam_macbench.json --cases 0
```

从初始池开始训练：

```bash
python scripts/run.py --config configs/Evoteam_gaia_train.json
python scripts/run.py --config configs/Evoteam_macbench_train.json
python scripts/run.py --config configs/Evoteam_masbench_train.json
```

先加 `--dry-run --max-items 1` 检查选题（GAIA 使用 `--cases 0`）。后附 CLI 参数覆盖配置默认值，包括 `--team`。外部运行时将初始池复制到 run 版本目录后演化；仓库内初始快照保持不变。测试不加 `--evolve`。输出、断点和池的持久化由外部方法运行时管理；原实验接口默认写入源码目录的 `runs/`、`sessions/` 和基准结果目录。

## 检查

```bash
pip install -r requirements/dev.txt
python -m pytest -q
```

未设置 `EVOTEAM_ROOT` 时不收集外部方法集成测试；设置后仍需相应数据集，否则数据测试跳过。离线测试和 `--dry-run` 不调用模型，不等于完整评测复现。
