# Skill-MAS file migration

Source: the supplied local `Skill_MAS-main/bench_adapters` implementation. Reusable method integration is separated from dataset-specific adapters; external frameworks/data are not vendored.

| Supplied file | Public location | Treatment |
| --- | --- | --- |
| `bench_adapters/gaia.py` | `benchmarks/gaia/adapters/skill_mas/adapter.py` | Task loading, prompts, attachment handling, supplied scorer; portable data path |
| `bench_adapters/macbench.py` | `benchmarks/macbench/adapters/skill_mas/adapter.py` | Native tool and evaluator bridge; external checkout configured by environment |
| `bench_adapters/masbench.py` | `benchmarks/masbench/adapters/skill_mas/adapter.py` | Manifest loader, private role tools, three category scorers |
| `bench_adapters/masbench_support/game_worker.py` | `benchmarks/masbench/adapters/skill_mas/masbench_support/game_worker.py` | Native Game RPC; private dependency path removed |
| `bench_adapters/masbench_support/README.zh-CN.md` | `benchmarks/masbench/adapters/skill_mas/masbench_support/README.md` | Rewritten for portable setup |
| `bench_adapters/base.py`, `__init__.py` | `benchmarks/common/skill_mas/base.py`, `__init__.py` | Shared native task interface and registry; unrelated adapters removed |
| `bench_adapters/native_runtime.py` | `methods/skill_mas/native_runtime.py` | Shared model/tool loop with actor binding; unrelated tools removed |
| `evolution/native_eval.py` | `methods/skill_mas/native_eval.py` | Evaluation of an explicit skill snapshot; no evolution CLI claim |
| local edits to `skill_mas/build.py` | `methods/skill_mas/patches/native-tool-build.patch` | Small patch against the pinned external upstream; three datasets only |
| local edits to `skill_mas/openai_async_client.py` | `methods/skill_mas/patches/native-tool-client.patch` | Native function calling and runtime environment controls |
| local edits to `template/sub_agent.py` | `methods/skill_mas/patches/native-tool-subagent.patch` | Native role/task payload bridge; callbacks invoked once for sync and async execution |
| remaining `template/*`, `utils/llm_cost.py`, `utils/secrets_resolve.py` | External pinned Skill-MAS checkout | Reused without local modifications |
| `data/masbench_split18_test90/{train_manifest,test_manifest,split_audit}.json` | `data/manifests/skill_mas/masbench/` | Small selection/audit metadata only |
| local model config and `.env*` | Ignored runtime configuration | Supply model ID/pricing and credential environment separately |
| LOCA, LoCoBench, BeyondSWE and mixed adapters | Not included | Outside this three-dataset release |
| datasets, run traces, logs, caches, archives, `.masbench_deps` | External source/environment or ignored output | Not uploaded |

The assembler builds an ignored upstream-compatible import tree. Source-of-truth files remain in the locations above. Do not edit generated run directories and then expect those edits to be published.

Remaining release gaps: EvoTeam adapters for all three datasets; AOrchestra adapters for GAIA/MAC-Bench; exact experiment model configs/skill snapshots and native evolution entry point; GAIA selection exports; a portable MASBench split-generation script; validated MASBench source revision; dependency locks and end-to-end runs. GAIA scoring, MAC-Bench budget admission and MASBench workflow protocol differences are documented in each adapter README.
