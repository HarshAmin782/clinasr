import json
from pathlib import Path

import pytest

from clinasr import stage2
from clinasr.sampling import selection_fingerprint

SELECTION_FILE = Path(__file__).resolve().parents[1] / "kaggle" / "stage2_selection.json"


def _records():
    return [
        {"file_name": "/a/1.wav", "recording_context": "narration_entity", "type_concept": "drugs", "session_id": "s1",
         "duration": 2.0, "text": "Take azithromycin 500 mg.", "medical_entities": '[["azithromycin", "drugs", [[5, 17]]], ["500 mg", "drugs", [[18, 24]]]]'},
        {"file_name": "/a/2.wav", "recording_context": "conversation", "type_concept": "misc_medical", "session_id": "s2",
         "duration": 3.0, "text": "fever since two days", "medical_entities": "[]"},
    ]


def test_pinned_fingerprint_matches_committed_selection():
    data = json.loads(SELECTION_FILE.read_text(encoding="utf-8"))
    assert selection_fingerprint(data["sample_ids"]) == data["fingerprint"] == stage2.STAGE2_EXPECTED_FINGERPRINT
    assert len(data["sample_ids"]) == stage2.STAGE2_TOTAL
    assert data["dataset_revision"] == stage2.DATASET_REVISION
    assert min(data["allocation"].values()) >= stage2.STAGE2_MIN_PER_STRATUM


def test_revisions_are_pinned_not_placeholders():
    assert len(stage2.DATASET_REVISION) == 40
    assert set(stage2.MODELS) == {"openai/whisper-tiny.en", "openai/whisper-base.en"}
    assert all(len(rev) == 40 for rev in stage2.MODELS.values())


def test_prediction_rows_keyed_by_file_name():
    rows = stage2.prediction_rows(_records(), ["take azithromycin 500 mg", "fever since 2 days"], "m")
    assert [r["sample_id"] for r in rows] == ["/a/1.wav", "/a/2.wav"]
    assert rows[0]["type_concept"] == "drugs"
    with pytest.raises(ValueError):
        stage2.prediction_rows(_records(), ["only one"], "m")


def test_summary_rows_overall_and_per_stratum():
    rows = stage2.prediction_rows(_records(), ["Take azithromycin 500 mg.", "fever since two days"], "m")
    summaries = {s["stratum"]: s for s in stage2.summary_rows(rows, "m")}
    assert summaries["ALL"]["n_samples"] == 2
    assert summaries["ALL"]["wer"] == 0.0
    assert summaries["narration_entity | drugs"]["n_samples"] == 1
    assert summaries["conversation | misc_medical"]["n_samples"] == 1


def test_entity_rows_are_marked_unvalidated():
    rows = stage2.entity_rows(_records(), ["take azithromycin five hundred mg", "fever"], "m")
    assert len(rows) == 2  # second sample has no annotated entities
    assert all(r["matching_validated"] is False for r in rows)
    status = {r["entity_text"]: r["match_status"] for r in rows}
    assert status == {"azithromycin": "exact", "500 mg": "omitted"}
    assert rows[1]["dosage_like"] is True
    assert "hypothesis" not in rows[0]
