# EvoTeam on GAIA

See [external source setup and compatibility status](../../../../methods/evoteam/README.md). This adapter preserves the uploaded execution policy, task prompts, attachment handling, and historical scoring. Model execution awaits upstream porting.

Obtain authorized [GAIA data](https://huggingface.co/datasets/gaia-benchmark/GAIA) separately. Set `GAIA_DATA_DIR` or use `--data-dir`; the directory must contain `level_{1,2,3}_test_100.json` and `val_files/` for the supplied config. The loader also supports `level_{1,2,3}_{val_new,train_20}.json`. Attachments may be under `val_files/<file_name>` or `val_files/<task_id>/<file_name>`. Task data and answers are not stored here.

```bash
python scripts/run_evoteam.py --config configs/evoteam/gaia.json --dry-run --data-dir /path/to/gaia
```

The full `test_100` task-ID set must match `benchmarks/gaia/manifest.json`, including when `--max-items` requests a small subset. Exact training-20 IDs and source file hashes still need to be recorded. The frozen GAIA pool is preserved in `configs/evoteam/pools/`.

The historical scorer lowercases and strips all characters outside ASCII letters/digits, then uses exact equality after extracting `FINAL ANSWER:`. This is the uploaded experiment metric, not a claim of equivalence to the official GAIA evaluator. Do not change metrics silently when comparing reported results. Evolution feedback includes references; only authorized training splits may use reference-feedback evolution.

Data-only checks use the Python standard library. Once runtime compatibility is resolved, install EvoTeam externally and its attachment parsers (`pandas`, `openpyxl`, `pdfplumber`, `python-docx`, `python-pptx`, as applicable). Tool implementations, search services, network access, and file/bash isolation still require runtime validation.
