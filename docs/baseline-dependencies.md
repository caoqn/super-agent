# Baseline dependencies and adapter status

The selected baselines are **EvoTeam, Skill-MAS, AgentNet, and AOrchestra**. Keep each baseline's integration code in this repository and install its upstream framework and Python packages in a separate environment. An upstream installation alone does not create an adapter for a different benchmark.

| Baseline | Upstream installation | Adapter status for this repository's datasets |
| --- | --- | --- |
| EvoTeam | [DEEP-JLU/EvoTeam](https://github.com/DEEP-JLU/EvoTeam) at `0c104c2235d428473726087596b855a057ee1564`; Python 3.10+; in its checkout, run `python3 -m pip install -e .`. BeyondSWE additionally needs Docker. | No local adapter yet. The upstream repository documents GAIA and other tracks, but not MAC-Bench or MASBench. |
| Skill-MAS | [linhh29/Skill_MAS](https://github.com/linhh29/Skill_MAS) at `f6cf17ab3ddbd3893d12b75f08c4b06a23ead873`; Python 3.11; assemble the checked-in native-tool integration with `python scripts/prepare_skill_mas.py /path/to/Skill_MAS --dataset gaia`. | Local adapters exist for GAIA, MAC-Bench, and MASBench. |
| AgentNet | [zoe-yyx/AgentNet](https://github.com/zoe-yyx/AgentNet) at `325d39f2a940be5fa903d28c411bd3426b8007f5`; run `python3 scripts/prepare_agentnet.py /path/to/AgentNet` and install only the selected adapter's requirements. | Local adapters exist for GAIA, MAC-Bench, and MASBench. |
| AOrchestra | [FoundationAgents/AOrchestra](https://github.com/FoundationAgents/AOrchestra) at `14a1a2051d6b03c479b706f8f555a60b8419e3b5`; Python 3.13 per upstream setup; in its checkout, run `python3 -m pip install -r requirements.txt`. | Local adapter exists for MASBench only. |

Use a separate virtual environment for frameworks with incompatible Python or package requirements. Do not install the full AgentNet or AOrchestra environment into one shared environment just because their adapters live in the same repository. Keep datasets, upstream checkouts, and run outputs outside tracked source.

The local `requirements*.txt` files describe direct dependencies and currently do not freeze every transitive version. For reported experiments, record the Python version and freeze each tested adapter environment separately after a successful smoke run.

## Existing adapter setup

- **Super Agent:** present in the merged repository for GAIA, MAC-Bench, and MASBench, but not part of the four selected baselines. Its DSH runtime is external and requires a benchmark-service bridge; see `methods/super_agent/README.md`. `requirements/masbench.txt` is the shared Python dependency list for this adapter family, not a baseline-wide or repository-wide requirements file.
- **AgentNet × GAIA:** use `benchmarks/gaia/adapters/agentnet/requirements_bench.txt` inside the generated `agentnet_gaia_run/` directory. GAIA data and credentials are separate prerequisites.
- **AgentNet × MAC-Bench:** use that adapter's `requirements_bench.txt`, then install MAC-Bench from its separate checkout with the `[modal,agents]` extras. The exact steps are in `benchmarks/macbench/adapters/agentnet/README.md`.
- **AgentNet × MASBench:** use `benchmarks/masbench/adapters/agentnet/requirements.txt`; download MASBench source/data separately.
- **Skill-MAS:** use `methods/skill_mas/requirements.txt`, prepare the pinned upstream checkout, then follow each dataset-specific README under `benchmarks/<dataset>/adapters/skill_mas/`. MASBench and GAIA inputs remain separate. MAC-Bench execution needs its separate package and task bundle.
- **AOrchestra × MASBench:** install the official AOrchestra `requirements.txt` in its own Python 3.13 environment; the adapter and MASBench task data remain separate.
- **ReAct and Vanilla:** retained adapters, though they are not in the selected baseline set above. Their HTTP-compatible model clients use the Python standard library; no additional Python package install is needed for those clients. MASBench source/data are still required.

EvoTeam can be added to `configs/matrix.yaml` after its adapters for the intended datasets are implemented and its environment-specific requirements are recorded beside those adapters. ReAct, Vanilla, and Super Agent are existing repository methods outside this selected four-baseline comparison.
