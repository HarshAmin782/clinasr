"""Tests for the local Kaggle CLI wrapper's own logic (argument validation,
opt-in gating, budget logging, non-overwriting fetch), never the real
`kaggle` CLI or network. The real `kaggle()` subprocess call is monkeypatched
out in every test.
"""

import json

import pytest

import kaggle_workflow


@pytest.fixture
def isolated_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle_workflow, "ROOT", tmp_path)
    monkeypatch.setattr(kaggle_workflow, "KAGGLE_DIR", tmp_path / "kaggle")
    monkeypatch.setattr(kaggle_workflow, "META", tmp_path / "kaggle" / "kernel-metadata.json")
    monkeypatch.setattr(kaggle_workflow, "SCRIPT", tmp_path / "kaggle" / "run_clinasr.py")
    monkeypatch.setattr(kaggle_workflow, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(kaggle_workflow, "BUDGET_LOG", tmp_path / "results" / "gpu_budget_log.json")
    calls = []
    monkeypatch.setattr(kaggle_workflow, "kaggle", lambda args: calls.append(args) or 0)
    monkeypatch.setattr(kaggle_workflow, "_get_git_commit", lambda: "deadbeef0000")
    return tmp_path, calls


def test_init_creates_private_gpu_internet_metadata(isolated_repo):
    _run(["init", "--username", "someuser"])
    data = json.loads(kaggle_workflow.META.read_text(encoding="utf-8"))
    assert data["is_private"] == "true"
    assert data["enable_gpu"] == "true"
    assert data["id"] == "someuser/clinasr-medical-asr"


def _run(argv):
    import sys

    old_argv = sys.argv
    sys.argv = ["kaggle_workflow.py", *argv]
    try:
        return kaggle_workflow.main()
    finally:
        sys.argv = old_argv


def test_push_refuses_without_run_clinasr_script(isolated_repo):
    _run(["init", "--username", "someuser"])
    with pytest.raises(RuntimeError, match="does not exist yet"):
        _run(["push", "--confirm", "--budget-minutes", "10"])


def test_push_refuses_without_confirm(isolated_repo):
    tmp_path, _ = isolated_repo
    _run(["init", "--username", "someuser"])
    kaggle_workflow.SCRIPT.parent.mkdir(parents=True, exist_ok=True)
    kaggle_workflow.SCRIPT.write_text("# placeholder\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Refusing to push without --confirm"):
        _run(["push", "--budget-minutes", "10"])


def test_push_refuses_without_budget(isolated_repo):
    tmp_path, _ = isolated_repo
    _run(["init", "--username", "someuser"])
    kaggle_workflow.SCRIPT.parent.mkdir(parents=True, exist_ok=True)
    kaggle_workflow.SCRIPT.write_text("# placeholder\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Pass --budget-minutes"):
        _run(["push", "--confirm"])


def test_push_with_confirm_and_budget_logs_and_calls_kaggle(isolated_repo):
    tmp_path, calls = isolated_repo
    _run(["init", "--username", "someuser"])
    kaggle_workflow.SCRIPT.parent.mkdir(parents=True, exist_ok=True)
    kaggle_workflow.SCRIPT.write_text("# placeholder\n", encoding="utf-8")

    rc = _run(["push", "--confirm", "--budget-minutes", "15"])
    assert rc == 0
    assert calls and calls[0][0] == "push"

    log = json.loads(kaggle_workflow.BUDGET_LOG.read_text(encoding="utf-8"))
    assert len(log) == 1
    assert log[0]["requested_budget_minutes"] == 15.0
    assert log[0]["git_commit"] == "deadbeef0000"


def test_budget_log_accumulates_across_pushes(isolated_repo):
    tmp_path, _ = isolated_repo
    _run(["init", "--username", "someuser"])
    kaggle_workflow.SCRIPT.parent.mkdir(parents=True, exist_ok=True)
    kaggle_workflow.SCRIPT.write_text("# placeholder\n", encoding="utf-8")

    _run(["push", "--confirm", "--budget-minutes", "10"])
    _run(["push", "--confirm", "--budget-minutes", "20"])
    log = json.loads(kaggle_workflow.BUDGET_LOG.read_text(encoding="utf-8"))
    assert len(log) == 2
    assert sum(e["requested_budget_minutes"] for e in log) == 30.0


def test_kaggle_subprocess_env_leaves_existing_token_alone(monkeypatch):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "already-set-token")
    env = kaggle_workflow._kaggle_subprocess_env()
    assert env["KAGGLE_API_TOKEN"] == "already-set-token"


def test_kaggle_subprocess_env_derives_token_from_credentials_file(tmp_path, monkeypatch):
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    creds_file = tmp_path / "kaggle.json"
    creds_file.write_text('{"username":"someuser","key":"fake-test-key-not-real"}', encoding="utf-8")
    monkeypatch.setattr(kaggle_workflow, "KAGGLE_CREDENTIALS_FILE", creds_file)

    env = kaggle_workflow._kaggle_subprocess_env()
    assert env["KAGGLE_API_TOKEN"] == "fake-test-key-not-real"


def test_kaggle_subprocess_env_handles_missing_credentials_file(tmp_path, monkeypatch):
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.setattr(kaggle_workflow, "KAGGLE_CREDENTIALS_FILE", tmp_path / "does-not-exist.json")

    env = kaggle_workflow._kaggle_subprocess_env()
    assert "KAGGLE_API_TOKEN" not in env


def test_fetch_creates_new_timestamped_dir_each_time(isolated_repo):
    tmp_path, calls = isolated_repo
    _run(["init", "--username", "someuser"])
    _run(["fetch"])
    _run(["fetch"])
    fetched_dirs = sorted(p.name for p in kaggle_workflow.RESULTS.iterdir() if p.name.startswith("fetched_"))
    assert len(fetched_dirs) == 2
    assert fetched_dirs[0] != fetched_dirs[1]
