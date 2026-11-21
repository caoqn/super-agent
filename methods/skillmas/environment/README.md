# Dependency Snapshot and Offline Checks

`python-current-snapshot.txt` is the output of `pip freeze` from the current Python 3.11.15 experiment environment; `pip check` passed when it was captured. Historical GAIA and MASBench launch scripts default to this Python environment; the historical MACBench execution environment has not been independently confirmed. The environment may have changed since the experiments, so this is a **current snapshot**, not an exact historical version or a complete portable lockfile. Two editable installs pointing to local Meta-Team and LoCoBench checkouts were excluded because they are not release dependencies for these three adapters. The snapshot includes packages from other projects in the environment, and not every package has been proven necessary for Skill-MAS. Some versions may not be available on other operating systems or architectures; verify compatibility before installing.

External dependencies still need to be pinned separately: raw GAIA data and attachments; MASBench code, data, and Game runtime (pulp 3.3.2 was also installed locally); and the MACBench harness, data, services, and Docker image. `required-artifacts.json` lists sources, authorization terms, revisions, and hashes that have not yet been verified; empty fields must not be treated as verified information. Do not use a MACBench data world revision as the harness code version. Verify all upstream sources and licenses, image availability, platform requirements, and system dependencies before release.

The offline checks prepared the source again from the pinned ZIP and SHA-256, checked all three train/test splits, and used `offline_entry_check.py` to verify real adapter loading, call parameters, configuration, and the frozen skill. The script stubs the training/testing process; it does not access the network or call an API. Example (replace paths and settings with actual values):

```bash
SKILLMAS_TEST_UPSTREAM=/path/to/work/Skill_MAS \
SKILLMAS_TEST_BENCHMARK=gaia SKILLMAS_TEST_PHASE=test \
SKILLMAS_TEST_DATA_ROOT=/path/to/gaia \
python methods/skillmas/offline_entry_check.py
```

`SKILLMAS_TEST_BENCHMARK` may be `gaia`, `masbench`, or `macbench`; `SKILLMAS_TEST_PHASE` may be `train` or `test`. The data described above and benchmark dependencies importable by the adapter are required. This check is not a complete benchmark run or a validation of real-world costs.
