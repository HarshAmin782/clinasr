"""Unit tests for the parts of clinasr.smoke_test that don't need
torch/transformers/network: row shaping, CSV writing, config building, and
the package-bootstrap helper (with subprocess/import mocked out). The
orchestration function `run_smoke_test` itself requires a real GPU/network
run and is deliberately NOT covered here -- see docs/WORKLOG.md for what
remains to be validated on an actual Kaggle run.
"""

import csv
import json

from clinasr.smoke_test import (
    build_experiment_config,
    build_prediction_rows,
    ensure_installed,
    write_predictions_csv,
    write_summary_csv,
)
from clinasr.metrics import compute_corpus_metric, compute_metrics


def _sample_records():
    return [
        {"session_id": "s1", "text": "hello world", "recording_context": "clinic"},
        {"session_id": "s2", "text": "take 2.5 mg", "recording_context": "pharmacy"},
    ]


def test_build_prediction_rows_shapes_fields_correctly():
    records = _sample_records()
    rows = build_prediction_rows(records, ["hello world", "take 2.5mg"], model_id="openai/whisper-tiny.en")
    assert rows[0] == {
        "sample_id": "s1",
        "reference": "hello world",
        "hypothesis": "hello world",
        "model": "openai/whisper-tiny.en",
        "condition": "clean",
        "session_id": "s1",
        "recording_context": "clinic",
    }
    assert rows[1]["sample_id"] == "s2"


def test_build_prediction_rows_requires_equal_length():
    import pytest

    with pytest.raises(ValueError):
        build_prediction_rows(_sample_records(), ["only one"], model_id="m")


def test_write_predictions_csv_round_trips(tmp_path):
    records = _sample_records()
    rows = build_prediction_rows(records, ["hello world", "take 2.5 mg"], model_id="m")
    path = tmp_path / "predictions.csv"
    write_predictions_csv(rows, path)

    with open(path, newline="", encoding="utf-8") as f:
        read_rows = list(csv.DictReader(f))
    assert len(read_rows) == 2
    assert read_rows[0]["sample_id"] == "s1"
    assert read_rows[0]["hypothesis"] == "hello world"


def test_write_summary_csv_contains_corpus_metrics(tmp_path):
    sample_metrics = compute_metrics(["s1", "s2"], ["hello world", "take 2.5 mg"], ["hello world", "take 2.5 mg"])
    corpus = compute_corpus_metric(sample_metrics)
    path = tmp_path / "summary_metrics.csv"
    write_summary_csv("openai/whisper-tiny.en", "clean", corpus, path)

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["model"] == "openai/whisper-tiny.en"
    assert rows[0]["wer"] == "0.0"


def test_build_experiment_config_round_trips_through_json():
    config = build_experiment_config("openai/whisper-tiny.en", resolved_revision="abc123", n=5, sample_ids=["s1", "s2"])
    data = json.loads(config.to_json())
    assert data["models"][0]["model_id"] == "openai/whisper-tiny.en"
    assert data["models"][0]["resolved_revision"] == "abc123"
    assert data["sample_selection"]["method"] == "smoke_test_first_n"
    assert data["sample_selection"]["sample_ids"] == ["s1", "s2"]
    # software_versions must be populated from the real environment, not guessed.
    assert "jiwer" in data["software_versions"]


def test_ensure_installed_returns_true_when_already_importable():
    assert ensure_installed("pytest") is True


def test_ensure_installed_installs_and_reports_failure(monkeypatch):
    import subprocess

    import clinasr.smoke_test as smoke_test_module

    def fake_run(cmd, capture_output, text):
        class Result:
            returncode = 1
            stderr = "simulated: no matching distribution found"

        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    ok = ensure_installed("a-package-that-does-not-exist-xyz")
    assert ok is False
