"""Skill-MAS native benchmark adapters installed by prepare_skill_mas.py."""

from .base import NativeEvalResult, NativeTask


def get_adapter(backend: str):
    key = str(backend).strip().lower()
    if key == "gaia":
        from benchmarks.gaia.adapters.skill_mas.adapter import GaiaAdapter
        return GaiaAdapter()
    if key == "macbench":
        from benchmarks.macbench.adapters.skill_mas.adapter import MacBenchAdapter
        return MacBenchAdapter()
    if key == "masbench":
        from benchmarks.masbench.adapters.skill_mas.adapter import MASBenchAdapter
        return MASBenchAdapter()
    raise ValueError(f"Unsupported Skill-MAS benchmark: {backend}")


__all__ = ["NativeEvalResult", "NativeTask", "get_adapter"]
