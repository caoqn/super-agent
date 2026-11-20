# Skill-MAS native integration

This directory keeps the reusable native evaluation loop, tool runtime and small core patches used by the three adapters. The upstream framework is installed separately from [linhh29/Skill_MAS](https://github.com/linhh29/Skill_MAS), pinned to `f6cf17ab3ddbd3893d12b75f08c4b06a23ead873` (Apache-2.0; see [upstream license](LICENSE.upstream)).

## Prepare

Use Python 3.11 in a dedicated environment. Run from this repository root:

```sh
python3.11 -m venv .venv-skill-mas
source .venv-skill-mas/bin/activate
python -m pip install -r methods/skill_mas/requirements.txt
git clone https://github.com/linhh29/Skill_MAS.git /path/to/Skill_MAS
git -C /path/to/Skill_MAS checkout f6cf17ab3ddbd3893d12b75f08c4b06a23ead873
python scripts/prepare_skill_mas.py /path/to/Skill_MAS --dataset gaia
```

Repeat the last command with `macbench` or `masbench` for that dataset. The generated `skill_mas_<dataset>_run/` directory is ignored by Git. The assembler copies only the upstream core packages, applies the recorded patches and installs one dataset adapter under the upstream-compatible `Skill_MAS.bench_adapters` namespace. It does not modify the upstream checkout. Keep generated directories outside version control; remove a generated directory before rebuilding it. Use `--output-dir /path/to/new/runtime` to build outside the repository instead.

The build, client and subagent patches add `native_tool` routing for GAIA/MAC-Bench/MASBench, OpenAI-compatible function calling and endpoint/runtime controls. `native_eval.py` preserves the supplied local three-stage workflow evaluation with build/repair usage accounting, checkpoints and per-task traces. `native_runtime.py` is the shared tool loop; dataset loaders, task prompts and scorers live under `benchmarks/`.

## Configure and evaluate

In the generated `Skill_MAS/skill_mas/model_config.json`, add your exact API model ID with `api_key: "${OPENAI_API_KEY}"`, numeric `input_per_1m`/`output_per_1m`, and appropriate generation parameters. Supply credentials via the environment or this repository's ignored `.env`. Set `DEV_API_BASE` for an OpenAI-compatible endpoint override. Never enter a literal secret into a tracked config. The local machine's model config is deliberately not copied.

Choose the exact `SKILL.md` snapshot used by the experiment and pass it with `--skill-file`. The upstream `init_skill/SKILL.md` can be used for an initial smoke run; it is not an evolved experiment snapshot. Run:

```sh
python methods/skill_mas/run_native_eval.py --help
```

Then follow the [GAIA](../../benchmarks/gaia/adapters/skill_mas/README.md), [MAC-Bench](../../benchmarks/macbench/adapters/skill_mas/README.md) or [MASBench](../../benchmarks/masbench/adapters/skill_mas/README.md) instructions. This entry point evaluates a skill snapshot; the native evolution CLI and experiment-specific trained skill snapshots are not released here yet. Results are written under ignored `runs/`. The evaluation loop defaults to six generation attempts, four execution attempts, twelve tool turns per subagent and one concurrent task in the public CLI. Historical source defaults used two concurrent tasks; set this explicitly when reproducing a run.

## Evaluation boundaries

- GAIA retains the supplied ASCII alphanumeric-normalized match metric; it is not the official GAIA scorer.
- MAC-Bench uses native task tools and final-state evaluation, but its admission callback is a no-op: official shared token/call/action budgets are not enforced by this adapter.
- MASBench retains asymmetric evidence, private calendars, local game views, explicit messaging and simultaneous game actions; Skill-MAS generates the collaboration workflow instead of reproducing the official fixed dialogue protocol.
- Generated Python and GAIA shell tools execute locally. The workspace helper is not an OS security sandbox.
- Timeout traces retain the source policy: timeout counts as zero in the current aggregate, has unavailable score in its trace, and is not automatically retried. Missing API usage on failure/timeout is not recovered. Treat cost totals as incomplete when such failures occur.

Before claiming full reproduction, publish the trained skill snapshot, exact model/parameters, dataset revisions, per-environment dependency lock and a successful end-to-end run. See the [migration map](../../docs/skill-mas-migration.md) for the supplied files and exclusions.

## Offline verification

With this method's requirements installed, run from the repository root:

```sh
SKILL_MAS_CHECKOUT=/path/to/Skill_MAS python -m unittest discover -s tests -v
```

The optional checkout enables assembly of all three dataset runtimes in temporary directories, import checks, synchronous/asynchronous role payload checks, and SDK client construction without API requests. Without it, the adapter, manifest and revision-rejection checks still run. These checks do not execute model calls, external task services or complete benchmark episodes.
