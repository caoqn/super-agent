# Skill-MAS × MASBench

First [prepare Skill-MAS](../../../../methods/skill_mas/README.md) with `--dataset masbench`. Install from this repository root:

```sh
python -m pip install -r benchmarks/masbench/adapters/skill_mas/requirements.txt
git clone https://github.com/BUPT-GAMMA/MASBench.git /path/to/MASBench
export MASBENCH_ROOT=/path/to/MASBench
```

Keep the source/data checkout separate. It must contain `Reasoning/reasoning.json`, `Reasoning/src/data_loader_v4_mas.py`, `Reasoning/src/metrics.py`, `Scheduling/meeting/minutes_10/5days/`, and `Game/masbench/` with `Game/custom_map/`. Record the exact MASBench revision used in your experiment; a matching public revision for the supplied local source has not been established yet. Game imports the external source through a subprocess; PuLP and other direct dependencies are installed in the environment, not a private `.masbench_deps` folder.

The checked-in [ID manifests](../../../../data/manifests/skill_mas/masbench/README.md) specify train18/test90, seed42, disjoint IDs. Task IDs are prefixed with `reasoning__`, `scheduling__` or `game__`.

```sh
python methods/skill_mas/run_native_eval.py \
  --runtime skill_mas_masbench_run --backend masbench --model YOUR_MODEL_ID \
  --skill-file /path/to/experiment/SKILL.md \
  --task-ids reasoning__sample_50_8fc73a48 --split test
```

Use `--manifest-dir /path/to/manifests` to provide a different train/test ID selection. Evaluation preserves the supplied adapter: Reasoning uses upstream F1, Scheduling checks an integer-valued 20-minute slot against `task_plans`, Game measures delivered/target resources within 50 simultaneous rounds. Private observations are exposed by runtime-bound actor identity and explicit messages. Generated workflows retain Skill-MAS planning; they do not reproduce MASBench's fixed dialogue protocol or turn limits. The shared tool interface limits private reads, but generated Python still executes in the local process.
