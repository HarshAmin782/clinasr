import pytest

from clinasr.entity_review import agreement, rematch, select_review_rows
from clinasr.normalization import NORMALIZATION_V1


def _entity_rows():
    return [
        {"sample_id": "s1", "model": "m", "entity_id": "0", "entity_text": "400mg", "entity_type": "drugs", "match_status": "x"},
        {"sample_id": "s1", "model": "m", "entity_id": "1", "entity_text": "Zentel", "entity_type": "drugs", "match_status": "x"},
    ]


def test_rematch_v2_vs_v1():
    hyps = {("s1", "m"): "Z until 400 mg overall suspension."}
    v2 = rematch(_entity_rows(), hyps)
    v1 = rematch(_entity_rows(), hyps, NORMALIZATION_V1)
    assert [r["match_status"] for r in v2] == ["normalized", "omitted"]
    assert [r["match_status"] for r in v1] == ["omitted", "omitted"]
    assert v2[0]["normalization"] == "v2" and v2[0]["dosage_like"] is True


def test_select_review_rows_caps_each_cell_and_is_deterministic():
    rows = [
        {"sample_id": f"s{i}", "model": m, "entity_id": "0", "match_status": st}
        for i in range(30) for m in ("a", "b") for st in ("exact", "omitted")
    ]
    one = select_review_rows(rows, per_cell=5, seed=1)
    two = select_review_rows(list(reversed(rows)), per_cell=5, seed=1)
    assert len(one) == 20
    assert one == two


def test_agreement_and_kappa():
    rows = (
        [{"match_status": "exact", "human_label": "preserved"}] * 8
        + [{"match_status": "omitted", "human_label": "not_preserved"}] * 8
        + [{"match_status": "omitted", "human_label": "preserved"}] * 2
        + [{"match_status": "exact", "human_label": "unclear"}]
        + [{"match_status": "exact", "human_label": ""}]
    )
    a = agreement(rows)
    assert a["n_labelled"] == 18
    assert a["n_unclear"] == 1 and a["n_blank"] == 1
    assert a["observed_agreement"] == pytest.approx(16 / 18)
    assert 0 < a["cohen_kappa"] < 1
    assert a["confusion_auto_vs_human"]["not_preserved->preserved"] == 2
    assert a["agreement_by_auto_status"]["omitted"] == pytest.approx(0.8)


def test_agreement_rejects_unknown_labels_and_handles_none():
    with pytest.raises(ValueError):
        agreement([{"match_status": "exact", "human_label": "yes"}])
    assert agreement([{"match_status": "exact", "human_label": ""}])["n_labelled"] == 0
