"""Tests for the EkaCare medical_entities adapter.

Annotation strings below follow the exact structure observed on real
records ('[[text, category, [[start, end], ...]], ...]'). Reference texts
are short synthetic sentences written for the tests, not copied
transcripts.
"""

import json

import pytest

from clinasr.entities import EntitySpan
from clinasr.medical_entities import (
    STATUS_EMPTY,
    STATUS_MALFORMED,
    STATUS_MISSING,
    STATUS_OK,
    STATUS_PARTIAL,
    STATUS_UNUSABLE,
    analyze_coverage,
    assert_unique_ids,
    evaluate_sample_entities,
    is_dosage_like,
    offsets_align,
    parse_medical_entities,
    summarize_strata,
)

REF = "Take azithromycin 500 mg once in a day for fever."
REAL_SHAPE = json.dumps(
    [
        ["azithromycin", "drugs", [[5, 17]]],
        ["500 mg", "drugs", [[18, 24]]],
        ["once in a day", "advices", [[25, 38]]],
        ["fever", "clinical_findings", [[43, 48]]],
    ]
)


def test_parses_real_shape():
    parsed = parse_medical_entities(REAL_SHAPE)
    assert parsed.status == STATUS_OK
    assert parsed.raw == REAL_SHAPE
    assert [e.text for e in parsed.entities] == ["azithromycin", "500 mg", "once in a day", "fever"]
    assert parsed.entities[0].entity_type == "drugs"
    assert parsed.entities[0].offsets == ((5, 17),)
    assert parsed.problems == ()


def test_real_shape_offsets_align_with_reference():
    parsed = parse_medical_entities(REAL_SHAPE)
    assert all(offsets_align(e, REF) for e in parsed.entities)


def test_misaligned_offsets_detected():
    entity = EntitySpan(text="fever", entity_type="clinical_findings", offsets=((0, 5),))
    assert offsets_align(entity, REF) is False


def test_offsets_align_returns_none_without_offsets_or_text():
    assert offsets_align(EntitySpan(text="fever"), REF) is None
    assert offsets_align(EntitySpan(text="fever", offsets=((0, 5),)), None) is None


def test_multiple_spans_per_entity_supported():
    parsed = parse_medical_entities('[["pain", "clinical_findings", [[0, 4], [10, 14]]]]')
    assert parsed.status == STATUS_OK
    assert parsed.entities[0].offsets == ((0, 4), (10, 14))


def test_empty_list_is_empty_not_missing():
    parsed = parse_medical_entities("[]")
    assert parsed.status == STATUS_EMPTY
    assert parsed.entities == ()


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_missing_values(raw):
    assert parse_medical_entities(raw).status == STATUS_MISSING


@pytest.mark.parametrize(
    "raw",
    ['[["unterminated", "drugs"', '{"text": "x"}', "42", 3.5, {"a": 1}],
)
def test_malformed_values_never_raise(raw):
    parsed = parse_medical_entities(raw)
    assert parsed.status == STATUS_MALFORMED
    assert parsed.raw == raw
    assert parsed.problems


def test_already_decoded_list_is_accepted():
    parsed = parse_medical_entities([["fever", "clinical_findings", [[0, 5]]]])
    assert parsed.status == STATUS_OK


def test_bad_entries_are_dropped_but_good_ones_kept():
    raw = json.dumps(
        [
            ["fever", "clinical_findings", [[0, 5]]],
            ["no category"],
            ["", "drugs", [[0, 1]]],
            ["cough", "", [[0, 5]]],
        ]
    )
    parsed = parse_medical_entities(raw)
    assert parsed.status == STATUS_PARTIAL
    assert [e.text for e in parsed.entities] == ["fever"]
    assert len(parsed.problems) == 3


def test_bad_offsets_keep_entity_without_offsets():
    for bad in ([], [[5, 2]], [[-1, 3]], [[1]], "0-5", [[True, 3]], [["0", "5"]]):
        parsed = parse_medical_entities(json.dumps([["fever", "clinical_findings", bad]]))
        assert parsed.status == STATUS_PARTIAL, bad
        assert parsed.entities[0].offsets is None
        assert parsed.problems == ("entry_0: bad_offsets",)


def test_all_entries_bad_is_unusable():
    parsed = parse_medical_entities('[["only two", "fields"], 7]')
    assert parsed.status == STATUS_UNUSABLE
    assert parsed.entities == ()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("500 mg", True),
        ("thyroxine 25 mcg", True),
        ("2.5 ml", True),
        ("10 units", True),
        ("Dolo 650", False),  # strength implied but no unit: not detectable
        ("Pantop DSR 40", False),
        ("6 days", False),
        ("azithromycin", False),
        ("5mg", True),
    ],
)
def test_dosage_heuristic(text, expected):
    assert is_dosage_like(text) is expected


def test_assert_unique_ids_passes_and_fails():
    assert_unique_ids(["/a/1.m4a", "/a/2.m4a"])
    with pytest.raises(ValueError, match="duplicate"):
        assert_unique_ids(["/a/1.m4a", "/a/1.m4a", "/a/2.m4a"])
    with pytest.raises(ValueError, match="no file_name"):
        assert_unique_ids(["/a/1.m4a", None])
    with pytest.raises(ValueError, match="no file_name"):
        assert_unique_ids(["/a/1.m4a", "  "])


def _records():
    return [
        {"file_name": "/a/1.m4a", "text": REF, "medical_entities": REAL_SHAPE,
         "recording_context": "conversation", "type_concept": "drugs", "duration": 4.0},
        {"file_name": "/a/2.m4a", "text": "fever and cough", "medical_entities": "[]",
         "recording_context": "dictation", "type_concept": "misc_medical", "duration": 2.0},
        {"file_name": "/a/3.m4a", "text": "x", "medical_entities": None,
         "recording_context": "conversation", "type_concept": "drugs", "duration": 1.5},
        {"file_name": "/a/4.m4a", "text": "fever", "medical_entities": "not json",
         "recording_context": "conversation", "type_concept": "drugs", "duration": 1.0},
        {"file_name": "/a/5.m4a", "text": "cough now", "medical_entities": '[["cough", "clinical_findings", [[3, 8]]]]',
         "recording_context": "dictation", "type_concept": "clinical_findings", "duration": 3.0},
    ]


def test_analyze_coverage_counts():
    report = analyze_coverage(_records())
    assert report.n_records == 5
    assert report.status_counts == {"ok": 2, "empty": 1, "missing": 1, "malformed": 1}
    assert report.n_samples_with_entities == 2
    assert report.n_entities == 5
    assert report.category_entity_counts == {"drugs": 2, "clinical_findings": 2, "advices": 1}
    assert report.category_sample_counts["clinical_findings"] == 2
    assert report.category_sample_counts["drugs"] == 1
    assert report.n_samples_with_drugs == 1
    assert report.n_dosage_like_drug_entities == 1
    assert report.n_samples_with_dosage_like_drug_entity == 1
    assert report.n_entities_with_offsets == 5
    assert report.n_offsets_aligned == 4
    assert report.n_offsets_misaligned == 1
    assert report.misaligned_examples[0]["sample_id"] == "/a/5.m4a"
    assert report.misaligned_examples[0]["reference_slices"] == ["gh no"]
    assert sorted(report.unusable_sample_ids) == ["/a/3.m4a", "/a/4.m4a"]
    assert report.problem_counts == {"invalid_json": 1}


def test_summarize_strata():
    strata = summarize_strata(_records(), ["recording_context", "type_concept"])
    assert strata["n_records"] == 5
    assert strata["per_key"]["recording_context"] == {"conversation": 3, "dictation": 2}
    top = strata["per_combination"][0]
    assert top["values"] == {"recording_context": "conversation", "type_concept": "drugs"}
    assert top["count"] == 3
    assert top["duration_seconds"] == pytest.approx(6.5)
    assert strata["total_duration_seconds"] == pytest.approx(11.5)


def test_evaluate_sample_entities_is_unvalidated_and_preserves_raw():
    parsed = parse_medical_entities(REAL_SHAPE)
    result = evaluate_sample_entities("/a/1.m4a", parsed, "take azithromycin five hundred mg once in a day for fever")
    assert result["annotation_raw"] == REAL_SHAPE
    assert result["summary"]["validated"] is False
    assert "recall" not in result["summary"]
    statuses = {row["entity_text"]: row["match_status"] for row in result["rows"]}
    assert statuses["azithromycin"] == "exact"
    assert statuses["500 mg"] == "omitted"  # spelled-out number is NOT reconciled
    assert statuses["fever"] == "exact"
    dosage_rows = [row for row in result["rows"] if row["dosage_like"]]
    assert [row["entity_text"] for row in dosage_rows] == ["500 mg"]
    assert result["summary_by_type"]["drugs"]["n_entities"] == 2


def test_evaluate_sample_entities_handles_missing_hypothesis():
    parsed = parse_medical_entities(REAL_SHAPE)
    result = evaluate_sample_entities("/a/1.m4a", parsed, None)
    assert all(row["match_status"] == "omitted" for row in result["rows"])
