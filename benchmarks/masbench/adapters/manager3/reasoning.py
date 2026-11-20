"""Load actor-specific evidence and score answers using upstream MASBench."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def private_context(root: Path, task_id: str) -> dict:
    loader_module = load_module("_manager3_mas_loader", root / "Reasoning/src/data_loader_v4_mas.py")
    loader = loader_module.V4MasDataLoader(str(root / "Reasoning/reasoning.json"), mode="context")
    sample = loader.sample(next(i for i, row in enumerate(loader._items) if row["id"] == task_id))
    return {"Alice": sample.context_first, "Bob": sample.context_second}


def evaluate(root: Path, reference, answer: str) -> tuple[float, dict]:
    metrics = load_module("_manager3_mas_metrics", root / "Reasoning/src/metrics.py")
    return metrics.cal_f1_score(str(reference), answer), {"prediction": answer, "reference": reference}
