"""Guard external source resolution without requiring a model or dataset."""
from pathlib import Path

import pytest
from methods.Evoteam.bootstrap import source_root


def test_source_is_required(monkeypatch):
    monkeypatch.delenv("EVOTEAM_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="Set EVOTEAM_ROOT"):
        source_root()


def test_incompatible_source_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("EVOTEAM_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="core/task_budget.py"):
        source_root()


def test_compatible_layout_resolves(tmp_path, monkeypatch):
    for name in ("benchmarks/adapter.py", "core/runner.py", "core/task_budget.py",
                 "core/team_selector.py", "main.py", "tools/macbench.py"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    monkeypatch.setenv("EVOTEAM_ROOT", str(tmp_path))
    assert source_root() == tmp_path.resolve()
