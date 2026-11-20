# EvoTeam reproducibility status

The official source is external and pinned in `methods/evoteam/upstream.json`. The uploaded adapters, example settings, four pools, and fixed MASBench manifests are preserved here. [The migration map](evoteam-file-map.csv) lists the original uploaded paths, new paths, and original SHA-256 values; relocated adapters/helpers have path and dependency-installation changes, while pool files and manifests remain byte-for-byte unchanged.

| Dataset | Config | Selected pool | Selection | Status |
| --- | --- | --- | --- | --- |
| GAIA | `configs/evoteam/gaia.json` | `pool_GAIA_pool_chairman_profile_change_r1_v020_test` | test_100, public manifest | Source mismatch; data-only checks available |
| MAC-Bench | `configs/evoteam/macbench.json` | `pool_MACBENCH_OURS_V006_TEST` | upstream test catalogue | Source mismatch; data-only checks available |
| MASBench | `configs/evoteam/masbench.json` | `pool_MASBENCH_DYNAMIC_6` | seed 42; 18 train / 90 test | Source mismatch; data-only checks available |

`pool_MACBENCH_OURS_INITIAL` is preserved as uploaded, but is not selected by the supplied JSON. Original model environment values are retained as provenance, not asserted to configure the official runtime. The old `methods/Evoteam/runtime/{core,tools,prompts,...}` implementation is not copied. `runtime/tools/macbench.py` is retained only as the benchmark-specific `tool_bridge.py`; framework primitives, method orchestration, and evolution code remain external. The original runtime wrappers and launcher are replaced by the data/check CLI.

## Missing before paper results can be reproduced

1. **Compatible official source integration.** Port the three adapters to the pinned source API. The official tree lacks `tools/primitives.py`; GAIA also expects file/bash/web tools. MAC-Bench expects `core.task_budget`, native callback integration, and runtime budget enforcement. MASBench's native tools, runner setup, actor-capacity, message checks, and reflection filters have no corresponding base hooks. Keep the native scoring and information boundaries intact during porting.
2. **Exact experiment settings.** Confirm model/provider, decoding, timeouts, cost and token budgets, evolution switches, seeds, repetitions, and concurrency used for the paper. The three uploaded configs are examples; they do not establish these values. Preserve the original pool settings and explicitly record every override when reproducing them.
3. **Environment and data versions.** Supply tested dependency locks, dataset hashes/revisions, and GAIA's exact training-20 selection. MAC-Bench is currently referenced at `c70d962ee8e428a6d9bae42af8d92e086581ee0d`; MASBench data/environment reference is `c0301a650db95958924bb080192c09fef39389a5`. These pins are setup references, not verified original experiment revisions.
4. **Pool provenance.** Document how the evolved GAIA/MAC-Bench pools were trained and frozen. The GAIA pool contains 19 distinct UUIDs and none match the public 100-task GAIA test manifest. This check alone cannot establish training-only provenance or rule out other task-specific information. MASBench manifests have 6 train and 30 test IDs per category with no exact-ID overlap; this does not prove semantic independence of Reasoning questions.
5. **Execution evidence.** After porting, run a model-backed smoke case for every dataset, verify native scoring, and archive actual configuration, dependency versions, model IDs, per-task outputs, and failures. No paid/API experiment or end-to-end runtime test was performed by this reorganization.
6. **License/provenance.** The uploaded `THIRD_PARTY.md` says the original export had no project license to inherit. No new license is assigned to EvoTeam or these pools; source and benchmark attributions are retained. Confirm redistribution terms with the rights holder for the final public artifact.

The data-only checks validate the local selection and basic file availability. They do not certify that the method can execute against the selected upstream. All three entries in `configs/matrix.yaml` represent preserved integrations, with execution status documented here and in each config.
