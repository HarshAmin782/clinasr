import pytest

from clinasr.critical_errors import (
    FLAG_PROVENANCE,
    extract_candidates,
    select_for_review,
    validate_review_rows,
)


def _cats(cands):
    return {c.ref_span + " -> " + c.hyp_span: c.categories for c in cands}


def test_zincovit_example_flags_inserted_medical_term_and_medication_context():
    cands = extract_candidates("s", "m", "Take Zincovit once in a day.", "takes in COVID once in a day")
    cats = [cat for c in cands for cat in c.categories]
    assert "inserted_medical_term" in cats
    assert all(c.flagged_by == FLAG_PROVENANCE for c in cands)


def test_negation_reversal_row_29_pattern():
    cands = extract_candidates("s", "m", "managed without surgery", "managed with surgery")
    assert len(cands) == 1
    assert cands[0].op == "substitute"
    assert "negation" in cands[0].categories


def test_dose_change_flags_number_and_unit():
    cands = extract_candidates("s", "m", "Glisen PM 500 mg tablet", "Glisen PM 5 g tablet")
    cats = set(cat for c in cands for cat in c.categories)
    assert {"number", "dose_unit", "medication_context"} <= cats


def test_unannotated_drug_name_near_tablet_is_flagged():
    cands = extract_candidates("s", "m", "Losartar 50 Tablet is a medicine", "Losator 50 tablet is a medicine",
                               entity_texts=["kidney function"])
    assert len(cands) == 1
    assert cands[0].ref_span == "losartar" and cands[0].hyp_span == "losator"
    assert "medication_context" in cands[0].categories
    assert cands[0].overlaps_annotated_entity is False


def test_overlap_with_annotated_entity_is_reported():
    cands = extract_candidates("s", "m", "Losartar 50 Tablet", "Losator 50 tablet", entity_texts=["Losartar 50 Tablet"])
    assert cands[0].overlaps_annotated_entity is True


def test_drug_suffix_heuristic():
    cands = extract_candidates("s", "m", "start amlodipine today", "start amlodopine today")
    assert "drug_like_name" in cands[0].categories


def test_benign_differences_are_not_flagged():
    assert extract_candidates("s", "m", "the patient was seen today", "a patient was seen today") == []
    assert extract_candidates("s", "m", "fever since monday", "fever since monday") == []


def test_hyphenated_repetition_loop_is_flagged_once_and_truncated():
    hyp = "the result " + "-".join(["no"] * 40)
    cands = extract_candidates("s", "m", "the result is normal", hyp)
    # Aligned as a substitution ("is normal" -> "no no") plus one insertion
    # chunk for the rest of the loop -- not one flag per repeated word.
    assert sorted(c.op for c in cands) == ["insert", "substitute"]
    assert all("negation" in c.categories for c in cands)
    assert any("…(+" in c.hyp_span for c in cands)


def test_empty_reference_or_hypothesis():
    assert extract_candidates("s", "m", "", "anything") == []
    cands = extract_candidates("s", "m", "take 5 mg", None)
    assert cands and cands[0].op == "delete"


def test_candidate_ids_are_stable_and_unique():
    a = extract_candidates("s", "m", "take 500 mg without food", "take 5 mg with food")
    b = extract_candidates("s", "m", "take 500 mg without food", "take 5 mg with food")
    assert [c.candidate_id for c in a] == [c.candidate_id for c in b]
    assert len({c.candidate_id for c in a}) == len(a)


def test_select_for_review_is_deterministic_and_capped():
    cands = []
    for i in range(20):
        cands += extract_candidates(f"s{i}", "m", "take 500 mg tablet", "take 50 mg tablet")
    one = select_for_review(cands, per_category=5, seed=1)
    two = select_for_review(list(reversed(cands)), per_category=5, seed=1)
    assert [c.candidate_id for c in one] == [c.candidate_id for c in two]
    assert len(one) == 5


def test_review_validation_requires_provenance():
    ok = [{"candidate_id": "1", "review_label": "unclear", "reviewer_type": "ai_assisted"},
          {"candidate_id": "2", "review_label": "", "reviewer_type": ""}]
    assert validate_review_rows(ok) == {"ai_assisted:unclear": 1}
    with pytest.raises(ValueError, match="reviewer_type"):
        validate_review_rows([{"candidate_id": "3", "review_label": "clinically_significant", "reviewer_type": ""}])
    with pytest.raises(ValueError, match="review_label"):
        validate_review_rows([{"candidate_id": "4", "review_label": "bad", "reviewer_type": "human"}])
