# EvoTeam

Method source: [DEEP-JLU/EvoTeam](https://github.com/DEEP-JLU/EvoTeam), pinned to [`0c104c2235d428473726087596b855a057ee1564`](https://github.com/DEEP-JLU/EvoTeam/tree/0c104c2235d428473726087596b855a057ee1564). The selected URL and adapter provenance are recorded in [upstream.json](upstream.json). No EvoTeam method implementation is vendored in this directory or elsewhere in this change.

## Files

- `benchmarks/{gaia,macbench,masbench}/adapters/evoteam/`: uploaded benchmark integrations, dataset loaders, scoring, and environment bridges.
- `configs/evoteam/{gaia,macbench,masbench}.json`: uploaded example run settings and compatibility status.
- `configs/evoteam/pools/`: four frozen experiment pools, including role YAML, role prompts, skills, templates, and collaboration rules. These are experiment inputs, not the method's framework implementation.
- `data/manifests/evoteam/masbench/`: train/test IDs only. Benchmark task data must be obtained separately.
- `scripts/run_evoteam.py`: source compatibility and data-only checks.
- `scripts/prepare_evoteam_masbench.py`: rebuild selected Reasoning files from official local data.

See the [file-by-file map](../../docs/evoteam-file-map.csv) and [remaining reproducibility gaps](../../docs/evoteam-reproducibility.md).

## Current compatibility

The uploaded adapters were built for a different runtime than the selected official source. The selected source lacks its expected `tools/` directory; MAC-Bench also requires `core.task_budget` and runner integration; MASBench requires native actor-tool, messaging, capacity, and reflection-filter hooks absent from the official base adapter. Copying the old runtime into this repository would violate the source boundary. Substituting the official URL alone would not reproduce the experiments.

The preserved `adapter.py` files therefore reject execution with a clear compatibility error. Their data/scoring helpers and the data-only CLI work independently of EvoTeam. No supported model execution command is published yet. Removing the gate requires porting the integrations and validating them end to end.

## Source installation

Use a separate checkout and environment, outside this repository:

```bash
git clone https://github.com/DEEP-JLU/EvoTeam.git /path/to/EvoTeam
git -C /path/to/EvoTeam checkout 0c104c2235d428473726087596b855a057ee1564
export EVOTEAM_ROOT=/path/to/EvoTeam
python3.11 -m venv /path/to/evoteam-env
source /path/to/evoteam-env/bin/activate
python -m pip install -e "$EVOTEAM_ROOT"
```

The official source's `pyproject.toml` declares its method dependencies; installing them does not restore the missing runtime modules or adapter hooks. This is a source-install command, not a guarantee of experiment compatibility. Transitive dependencies still need a tested lockfile.

Official EvoTeam reads `EVOTEAM_ENV_FILE`, `LLM_PROVIDER`, `DEV_API_BASE`, and `DEV_API_KEY` for its endpoint, and agent model configuration uses `EVOTEAM_MODEL`. Use an external, untracked environment file. The uploaded JSON examples instead used `META_TEAM_MODEL` and `META_TEAM_MAX_STEPS`. These original values are preserved under `env_original`, with an explanation, and are not applied by the check/data CLI. Confirm the exact model configuration during porting instead of relying on fallback defaults.

## Checks without model calls

The following commands require only Python 3.11 and local data; `--check` requires an external Git checkout. Run from this repository root:

```bash
python scripts/run_evoteam.py --config configs/evoteam/gaia.json --check
python scripts/run_evoteam.py --config configs/evoteam/gaia.json --dry-run --data-dir /path/to/gaia
python scripts/run_evoteam.py --config configs/evoteam/macbench.json --dry-run --macbench-root /path/to/MAC_Bench
python scripts/prepare_evoteam_masbench.py --source /path/to/MASBench-main
python scripts/run_evoteam.py --config configs/evoteam/masbench.json --dry-run --masbench-root /path/to/MASBench-main
```

`--check` exits 2 while compatibility is unresolved. `--dry-run` only validates data/IDs; it neither loads EvoTeam nor applies model environment settings. It exits nonzero on missing or inconsistent data. It does not validate the full runtime, budgets, environments, or scoring execution.

Dataset-specific dependencies, data layout, and scoring notes are beside the [GAIA](../../benchmarks/gaia/adapters/evoteam/README.md), [MAC-Bench](../../benchmarks/macbench/adapters/evoteam/README.md), and [MASBench](../../benchmarks/masbench/adapters/evoteam/README.md) adapters.
