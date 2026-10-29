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


def test_agreement_per_class_precision_recall_f1():
    rows = (
        [{"match_status": "exact", "human_label": "preserved"}] * 99
        + [{"match_status": "normalized", "human_label": "not_preserved"}] * 1
        + [{"match_status": "omitted", "human_label": "preserved"}] * 13
        + [{"match_status": "omitted", "human_label": "not_preserved"}] * 37
    )
    a = agreement(rows)
    pc = a["per_class_auto_vs_labels"]
    assert a["observed_agreement"] == pytest.approx(136 / 150)
    assert pc["preserved"]["precision"] == pytest.approx(99 / 100)
    assert pc["preserved"]["recall"] == pytest.approx(99 / 112)
    assert pc["not_preserved"]["precision"] == pytest.approx(37 / 50)
    assert pc["not_preserved"]["recall"] == pytest.approx(37 / 38)
    assert pc["preserved"]["support_in_labels"] == 112
    assert a["cohen_kappa"] == pytest.approx(0.7766, abs=1e-3)
    assert "NOT representative" in a["note"] or "not representative" in a["note"]


def test_agreement_cli_records_provenance_flags_and_keeps_labels(tmp_path):
    import csv
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sheet = tmp_path / "sheet.csv"
    rows = [
        {"review_id": "0", "entity_text": "kidney function", "hypothesis": "preserving kidney function", "auto_label": "exact", "human_label": "preserved"},
        {"review_id": "1", "entity_text": "surgery", "hypothesis": "managed with surgery", "auto_label": "normalized", "human_label": "not_preserved"},
        {"review_id": "2", "entity_text": "statin-induced", "hypothesis": "statin induced", "auto_label": "omitted", "human_label": "preserved"},
    ]
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    before = sheet.read_bytes()
    flags = tmp_path / "flags.csv"
    flags.write_text("review_id,flag,alternative_label_for_sensitivity,reason\n1,contextual_negation_error,preserved,x\n", encoding="utf-8")
    out = tmp_path / "agreement.json"
    subprocess.run(
        [sys.executable, str(root / "scripts" / "entity_review.py"), "agreement", str(sheet),
         "--label-source", "AI-assisted (OpenAI GPT-6)", "--flags", str(flags), "--rematch-with", "v3", "--out", str(out)],
        check=True, capture_output=True,
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert sheet.read_bytes() == before  # labelled sheet never modified
    assert report["annotation_source"] == "AI-assisted (OpenAI GPT-6)"
    assert report["annotations_are_independent_human_labels"] is False
    assert "NOT representative" in report["sample_design"]
    assert report["original_labels"]["n_labelled"] == 3
    assert report["flagged_rows"][0]["flag"] == "contextual_negation_error"
    assert report["sensitivity_excluding_flagged"]["n_labelled"] == 2
    assert report["sensitivity_with_alternative_labels"]["observed_agreement"] == pytest.approx(2 / 3)
    # v3 rematch turns the hyphen row into a match -> agreement with labels improves
    assert report["automatic_labels_recomputed_v3"]["observed_agreement"] > report["original_labels"]["observed_agreement"]


def test_agreement_cli_rejects_flags_for_unknown_rows(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sheet = tmp_path / "sheet.csv"
    sheet.write_text("review_id,entity_text,hypothesis,auto_label,human_label\n0,a,a,exact,preserved\n", encoding="utf-8")
    flags = tmp_path / "flags.csv"
    flags.write_text("review_id,flag,alternative_label_for_sensitivity,reason\n99,x,,y\n", encoding="utf-8")
    r = subprocess.run([sys.executable, str(root / "scripts" / "entity_review.py"), "agreement", str(sheet),
                        "--label-source", "human", "--flags", str(flags)], capture_output=True, text=True)
    assert r.returncode != 0 and "unknown review_id" in r.stderr
