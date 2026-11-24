"""Validate source discovery and the exact public source pin without model calls."""
import hashlib
import io
import json
import tarfile
import pytest
from methods.Evoteam import bootstrap
from scripts import setup_evoteam

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


@pytest.mark.parametrize("source_args, expected_repository", [
    ([], "example/runtime"),
    (["--source-repository", "example/override"], "example/override"),
])
def test_setup_downloads_configured_pinned_source(tmp_path, monkeypatch, source_args,
                                                expected_repository):
    repo = tmp_path / "evaluation"
    lock_path = repo / "methods/Evoteam/source-lock.json"
    lock_path.parent.mkdir(parents=True)
    payload = b"runtime contents\n"
    lock_path.write_text(json.dumps({"revision": "revision123", "files": {
        "core/runner.py": hashlib.sha256(payload).hexdigest()}}))
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        member = tarfile.TarInfo("source/core/runner.py")
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))
    urls = []

    def open_archive(url, timeout):
        urls.append(url)
        return io.BytesIO(archive.getvalue())

    monkeypatch.setattr(setup_evoteam, "ROOT", repo)
    monkeypatch.setattr(setup_evoteam.urllib.request, "urlopen", open_archive)
    monkeypatch.setenv("EVOTEAM_SOURCE_REPOSITORY", "example/runtime")
    monkeypatch.setattr("sys.argv", ["setup_evoteam.py", "--no-install", *source_args])
    setup_evoteam.main()
    assert urls == [f"https://codeload.github.com/{expected_repository}/tar.gz/revision123"]
    assert (repo / "external/EvoTeam/core/runner.py").read_bytes() == payload


def test_setup_requires_runtime_source_for_fresh_download(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "evaluation"
    lock_path = repo / "methods/Evoteam/source-lock.json"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(json.dumps({"revision": "test-pin", "files": {}}))
    monkeypatch.setattr(setup_evoteam, "ROOT", repo)
    monkeypatch.delenv("EVOTEAM_SOURCE_REPOSITORY", raising=False)
    monkeypatch.setattr("sys.argv", ["setup_evoteam.py", "--no-install"])
    with pytest.raises(SystemExit, match="2"):
        setup_evoteam.main()
    assert "EVOTEAM_SOURCE_REPOSITORY" in capsys.readouterr().err


def test_setup_verifies_existing_checkout_without_source_setting(checkout, monkeypatch):
    monkeypatch.setattr(setup_evoteam, "ROOT", bootstrap.REPO_ROOT)
    monkeypatch.delenv("EVOTEAM_SOURCE_REPOSITORY", raising=False)
    monkeypatch.setattr("sys.argv", ["setup_evoteam.py", "--no-install"])
    setup_evoteam.main()
