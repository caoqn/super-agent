# EvoTeam on MASBench

See [external EvoTeam source and compatibility status](../../../../methods/evoteam/README.md). This integration preserves role-private Reasoning/Scheduling observations, one-to-one actor bindings, explicit messaging, Game actions, and native environment scoring. Those hooks are absent from the selected official runtime, so model execution remains gated.

Install benchmark data/environment separately:

```bash
git clone https://github.com/BUPT-GAMMA/MASBench.git /path/to/MASBench-main
git -C /path/to/MASBench-main checkout c0301a650db95958924bb080192c09fef39389a5
python -m pip install -r benchmarks/masbench/adapters/evoteam/requirements-game.txt
export MASBENCH_ROOT=/path/to/MASBench-main
python scripts/prepare_evoteam_masbench.py --source "$MASBENCH_ROOT"
python scripts/run_evoteam.py --config configs/evoteam/masbench.json --dry-run
```

The reference revision is a setup pin; the original experiment revision is not established. Python packages install normally into the selected environment; no `.masbench_deps` or `MASBENCH_SUPPORT_ROOT` directory is needed. The Game worker uses the same Python interpreter and imports the simulator from the external source's `Game/` directory. Dependencies are listed in `requirements-game.txt`; they still need a tested lockfile.

The manifests contain 6 train and 30 test IDs per category (18/90 total). The preparation script rebuilds Reasoning rows from `Reasoning/reasoning.json` into ignored `.artifacts/evoteam/masbench_split18_test90/`; it refuses to overwrite an existing output directory. The loader reads Scheduling calendars/solutions and Game maps from the external benchmark. Use `MASBENCH_SPLIT_DIR` or `--data-dir` to select another prepared directory. Generated task rows contain answers and must remain outside Git.

Reasoning uses the uploaded SQuAD-style token F1; Scheduling checks strict JSON with a 20-minute slot on a 10-minute grid against native solutions; Game wraps upstream physics and measures delivered resources over up to 50 simultaneous rounds. These reproduce the supplied adapter's semantics, not a verified equivalence to every official scoring path. Compare them to the paper protocol before publishing numbers.
