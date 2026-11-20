"""Validate source discovery and the exact public source pin without model calls."""
import hashlib
import json
import pytest
from methods.Evoteam import bootstrap

REQUIRED = ("benchmarks/adapter.py", "core/runner.py", "core/task_budget.py",
            "core/team_selector.py", "main.py", "tools/macbench.py")


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    repo = tmp_path / "evaluation"
    root = repo / "external/EvoTeam"
    root.mkdir(parents=True)
    lock_path = repo / "methods/Evoteam/source-lock.json"
    lock_path.parent.mkdir(parents=True)
    files = {}
    for name in REQUIRED:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# pinned source\n")
        files[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    lock_path.write_text(json.dumps({"revision": "test-pin", "files": files}))
    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo)
    monkeypatch.delenv("EVOTEAM_ROOT", raising=False)
    return root


def test_default_downloaded_source_resolves(checkout):
    assert bootstrap.source_root() == checkout.resolve()


def test_missing_source_explains_setup(checkout):
    import shutil
    shutil.rmtree(checkout)
    with pytest.raises(RuntimeError, match="setup_evoteam.py"):
        bootstrap.source_root()


def test_explicit_incompatible_source_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("EVOTEAM_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="core/task_budget.py"):
        bootstrap.source_root()


def test_modified_source_rejected(checkout):
    (checkout / "core/runner.py").write_text("# incompatible version\n")
    with pytest.raises(RuntimeError, match="differs from pinned test-pin"):
        bootstrap.source_root()
