# Dependency Snapshot and Offline Checks

`python-current-snapshot.txt` is a `pip freeze` record from the Python 3.11 validation environment; `pip check` passed when it was captured. Use it as a dependency reference when preparing a run environment. Two editable installs pointing to local Meta-Team and LoCoBench checkouts were excluded because they are not release dependencies for these adapters. The snapshot includes packages from other projects, and not every package is required by Skill-MAS. Verify compatibility when installing on another platform.

External dependencies are installed separately: raw GAIA data and attachments; MASBench code, data, and Game runtime; and the MACBench harness, data, services, and Docker image. `required-artifacts.json` records the relevant sources, authorization terms, revisions, and hashes. Verify upstream licenses, image availability, platform requirements, and system dependencies before running an evaluation.

The offline checks prepared the source again from the pinned ZIP and SHA-256, checked all three train/test splits, and used `offline_entry_check.py` to verify real adapter loading, call parameters, configuration, and the frozen skill. The script stubs the training/testing process; it does not access the network or call an API. Example (replace paths and settings with actual values):

```bash
SKILLMAS_TEST_UPSTREAM=/path/to/work/Skill_MAS \
SKILLMAS_TEST_BENCHMARK=gaia SKILLMAS_TEST_PHASE=test \
SKILLMAS_TEST_DATA_ROOT=/path/to/gaia \
python methods/skillmas/offline_entry_check.py
```

`SKILLMAS_TEST_BENCHMARK` may be `gaia`, `masbench`, or `macbench`; `SKILLMAS_TEST_PHASE` may be `train` or `test`. The data described above and benchmark dependencies importable by the adapter are required. This check is not a complete benchmark run or a validation of real-world costs.
