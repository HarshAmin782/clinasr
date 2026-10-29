import json

import pytest

from clinasr.run_manifest import make_run_id, new_run


def test_make_run_id_uses_git_commit_when_given():
    run_id = make_run_id(git_commit="abc123def456")
    assert run_id.endswith("_abc123def456")


def test_make_run_id_falls_back_to_nogit():
    run_id = make_run_id(git_commit=None)
    assert run_id.endswith("_nogit")


def test_new_run_creates_isolated_directory(tmp_path):
    handle = new_run(tmp_path, git_commit="deadbeef0000")
    assert handle.run_dir.exists()
    assert handle.run_dir.parent == tmp_path


def test_new_run_never_overwrites_existing_run(tmp_path, monkeypatch):
    import clinasr.run_manifest as run_manifest_module
    from datetime import datetime, timezone

    fixed_now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(run_manifest_module, "make_run_id", lambda git_commit=None: "fixed-run-id")

    first = new_run(tmp_path, git_commit="deadbeef0000")
    assert first.run_id == "fixed-run-id"
    with pytest.raises(FileExistsError):
        new_run(tmp_path, git_commit="deadbeef0000")


def test_write_manifest_includes_extra_fields(tmp_path):
    handle = new_run(tmp_path, git_commit="deadbeef0000")
    handle.write_manifest({"git_commit": "deadbeef0000", "note": "hello"})
    data = json.loads(handle.manifest_path.read_text(encoding="utf-8"))
    assert data["run_id"] == handle.run_id
    assert data["note"] == "hello"


def test_get_logger_writes_to_log_file(tmp_path):
    handle = new_run(tmp_path, git_commit="deadbeef0000")
    logger = handle.get_logger()
    logger.info("hello from test")
    for h in logger.handlers:
        h.flush()
    assert "hello from test" in handle.log_path.read_text(encoding="utf-8")
