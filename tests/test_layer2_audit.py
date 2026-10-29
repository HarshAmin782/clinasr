import pytest

from clinasr.layer2_audit import RAW_FIELDS, summarize_audited, validate_audited_against_original

BASE = {"sample_id": "s1", "model": "m", "condition": "clean", "op": "substitute", "overlaps_annotated_entity": "False",
        "ref_context": "x", "hyp_context": "y", "normalization": "v3", "detector_version": "d", "flagged_by": "f"}


def _rows():
    return [
        dict(BASE, candidate_id="a", ref_span="without", hyp_span="with", categories="negation",
             review_label="significant", reviewer_type="human", span_label_text_only="significant",
             span_label_reviewer_type="ai_assisted", context_label_text_only="significant",
             context_audit_reviewer_type="ai_assisted", audit_needs_human_confirmation="false"),
        dict(BASE, candidate_id="b", ref_span="mg dl", hyp_span="milligram per deciliter", categories="number;dose_unit",
             review_label="significant", reviewer_type="ai_assisted", span_label_text_only="not_significant",
             span_label_reviewer_type="ai_assisted", context_label_text_only="significant",
             context_audit_reviewer_type="ai_assisted", audit_needs_human_confirmation="true"),
        dict(BASE, candidate_id="c", sample_id="s2", ref_span="tumor", hyp_span="tumour", categories="medication_context",
             review_label="not_significant", reviewer_type="ai_assisted", span_label_text_only="not_significant",
             span_label_reviewer_type="ai_assisted", context_label_text_only="not_significant",
             context_audit_reviewer_type="ai_assisted", audit_needs_human_confirmation="false"),
    ]


def _original(rows):
    return [{k: r[k] for k in RAW_FIELDS} for r in rows]


def test_validation_accepts_matching_file():
    validate_audited_against_original(_original(_rows()), _rows())


def test_validation_rejects_modified_raw_field():
    rows = _rows()
    tampered = [dict(r) for r in rows]
    tampered[0]["hyp_span"] = "with out"
    with pytest.raises(ValueError, match="raw field"):
        validate_audited_against_original(_original(rows), tampered)


def test_validation_rejects_missing_rows_and_bad_labels():
    rows = _rows()
    with pytest.raises(ValueError, match="candidate ids differ"):
        validate_audited_against_original(_original(rows), rows[:2])
    bad = [dict(r) for r in rows]
    bad[2]["span_label_text_only"] = "clinically_significant"
    with pytest.raises(ValueError, match="not in"):
        validate_audited_against_original(_original(rows), bad)
    bad2 = [dict(r) for r in rows]
    bad2[0]["reviewer_type"] = "doctor"
    with pytest.raises(ValueError, match="reviewer_type"):
        validate_audited_against_original(_original(rows), bad2)


def test_summary_separates_span_and_context_and_counts_pairs():
    s = summarize_audited(_rows())
    span = s["span_level_detector_precision"]
    assert span["overall"] == {"significant": 1, "n_decided": 3, "rate": pytest.approx(1 / 3), "unclear": 0}
    assert span["by_category"]["negation"]["rate"] == 1.0
    assert span["by_category"]["dose_unit"]["rate"] == 0.0
    assert s["context_label_counts"] == {"not_significant": 1, "significant": 2}
    assert s["span_vs_context_disagreements"] == 1
    assert s["n_unique_sample_model_pairs"] == 2
    assert s["pairs_with_any_significant_context_label"] == 1
    assert s["review_label_reviewer_types"] == {"ai_assisted": 2, "human": 1}
    assert "not clinical significance" in s["interpretation"]
