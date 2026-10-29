"""Recompute every headline number for the final report from the saved
experiment outputs and write docs/final_results.json.

Nothing here runs a model. Metrics are recomputed from predictions.csv and
then CROSS-CHECKED against the summary files written at the time; any
mismatch aborts. The report, README and figures read only this JSON, so
every published number traces back to a source file listed in it.

Usage:
    python scripts/collect_final_results.py
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinasr.analysis import paired_bootstrap  # noqa: E402
from clinasr.layer2_audit import summarize_audited  # noqa: E402
from clinasr.normalization import NORMALIZATION_VERSIONS  # noqa: E402
from clinasr.stage2 import summary_rows  # noqa: E402

RESULTS = ROOT / "results"
STAGE1 = RESULTS / "fetched_20261029T143136Z"
STAGE2 = RESULTS / "fetched_20261029T163039Z"
STAGE4 = RESULTS / "fetched_20261029T164654Z"
COVERAGE = RESULTS / "20261029T150959Z_f96c5f60bc95" / "entity_coverage_report.json"
OUT = ROOT / "docs" / "final_results.json"

TINY, BASE = "openai/whisper-tiny.en", "openai/whisper-base.en"
MODELS = [TINY, BASE]
CONDITIONS = ["clean", "snr_20db", "snr_10db", "snr_0db"]
TOL = 1e-12

sources = {}


def src(path: Path) -> Path:
    sources[str(path.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_csv(path: Path):
    with open(src(path), newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_json(path: Path):
    return json.loads(src(path).read_text(encoding="utf-8"))


def scores(rows, version):
    """{condition: {model: {stratum: {wer, cer, n}}}} recomputed from predictions."""
    cfg = NORMALIZATION_VERSIONS[version]
    out = defaultdict(dict)
    for cond in sorted({r.get("condition", "clean") for r in rows}):
        for model in MODELS:
            sub = [r for r in rows if r["model"] == model and r.get("condition", "clean") == cond]
            out[cond][model] = {s["stratum"]: {"wer": s["wer"], "cer": s["cer"], "n": s["n_valid"]} for s in summary_rows(sub, model, cfg)}
    return out


def cross_check(recomputed, stored_csv: Path, version):
    for row in read_csv(stored_csv):
        cond = row.get("condition", "clean")
        mine = recomputed[cond][row["model"]][row["stratum"]]
        for metric in ("wer", "cer"):
            if abs(float(row[metric]) - mine[metric]) > TOL:
                raise SystemExit(f"MISMATCH {stored_csv.name} {version} {cond} {row['model']} {row['stratum']} {metric}")


def paired(rows, cond, version):
    sub = [r for r in rows if r.get("condition", "clean") == cond]
    return dataclasses.asdict(paired_bootstrap(sub, TINY, BASE, n_bootstrap=2000, seed=0, config=NORMALIZATION_VERSIONS[version]))


def degradation(rows, model, cond, version):
    sub = [dict(r, model="clean") for r in rows if r["model"] == model and r["condition"] == "clean"]
    sub += [dict(r, model="noisy") for r in rows if r["model"] == model and r["condition"] == cond]
    d = paired_bootstrap(sub, "clean", "noisy", n_bootstrap=2000, seed=0, config=NORMALIZATION_VERSIONS[version])
    return {"wer_increase": d.difference, "ci_low": d.ci_low, "ci_high": d.ci_high, "n": d.n_samples}


def entity_rates(path: Path):
    rows = read_csv(path)
    out = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        keys = ["ALL", r["entity_type"]]
        for k in keys:
            out[r["model"]][k][0] += r["match_status"] != "omitted"
            out[r["model"]][k][1] += 1
        if r["dose_preserved"] in ("True", "False"):
            out[r["model"]]["dose_preserved"][0] += r["dose_preserved"] == "True"
            out[r["model"]]["dose_preserved"][1] += 1
            out[r["model"]]["dose_entity_whole_match"][0] += r["match_status"] != "omitted"
            out[r["model"]]["dose_entity_whole_match"][1] += 1
    return {m: {k: {"matched": a, "n": n, "rate": a / n} for k, (a, n) in v.items()} for m, v in out.items()}


def main() -> int:
    final = {"generated_by": "scripts/collect_final_results.py", "validated_entity_labels": False}

    # Smoke test (historical, v1-era normalization at run time).
    smoke = read_csv(STAGE1 / "summary_metrics.csv")[0]
    final["stage1_smoke"] = {"model": smoke["model"], "n": int(smoke["n_samples_total"]), "wer": float(smoke["wer"]), "cer": float(smoke["cer"]),
                             "note": "5 samples, first-N selection, pre-v2 normalization; historical only"}

    # Population coverage (full test split, no audio).
    cov = read_json(COVERAGE)
    final["dataset"] = {
        "n_records": cov["n_records"], "n_sessions": cov["n_unique_sessions"], "duration_seconds": cov["duration_seconds"],
        "annotation_status": cov["coverage"]["status_counts"], "n_entities": cov["coverage"]["n_entities"],
        "category_entity_counts": cov["coverage"]["category_entity_counts"],
        "category_sample_counts": cov["coverage"]["category_sample_counts"],
        "n_samples_with_drugs": cov["coverage"]["n_samples_with_drugs"],
        "n_dosage_like_drug_entities": cov["coverage"]["n_dosage_like_drug_entities"],
        "n_samples_with_dosage_like_drug_entity": cov["coverage"]["n_samples_with_dosage_like_drug_entity"],
        "offsets_checked": cov["coverage"]["n_entities_with_offsets"], "offsets_aligned": cov["coverage"]["n_offsets_aligned"],
        "strata": cov["strata"]["per_combination"],
    }

    # Stage 2 clean comparison, v1/v2/v3, recomputed + cross-checked.
    preds2 = read_csv(STAGE2 / "predictions.csv")
    sel = read_json(STAGE2 / "selection.json")
    final["stage2"] = {"n_samples": len(sel["sample_ids"]), "fingerprint": sel["fingerprint"], "allocation": sel["allocation"],
                       "environment": read_json(STAGE2 / "environment.json"), "by_normalization": {}}
    for v in ("v1", "v2", "v3"):
        rec = scores(preds2, v)
        cross_check(rec, STAGE2 / f"summary_metrics_{v}.csv", v)
        if v == "v1":
            cross_check(rec, STAGE2 / "summary_metrics.csv", "v1-original")
        final["stage2"]["by_normalization"][v] = {"metrics": rec["clean"], "paired_base_minus_tiny": paired(preds2, "clean", v)}

    # Stage 4 noise, v2/v3.
    preds4 = read_csv(STAGE4 / "predictions.csv")
    final["stage4"] = {"environment": read_json(STAGE4 / "environment.json"), "by_normalization": {}}
    for v in ("v2", "v3"):
        rec = scores(preds4, v)
        cross_check(rec, STAGE4 / f"summary_metrics_{v}.csv", v)
        if v == "v2":
            cross_check(rec, STAGE4 / "summary_metrics.csv", "v2-original")
        final["stage4"]["by_normalization"][v] = {
            "metrics": {c: rec[c] for c in CONDITIONS},
            "paired_base_minus_tiny": {c: paired(preds4, c, v) for c in CONDITIONS},
            "degradation_vs_clean": {m: {c: degradation(preds4, m, c, v) for c in CONDITIONS[1:]} for m in MODELS},
        }
    clean2 = final["stage2"]["by_normalization"]["v2"]["metrics"]
    clean4 = final["stage4"]["by_normalization"]["v2"]["metrics"]["clean"]
    final["stage4"]["clean_reproduces_stage2_v2"] = all(
        abs(clean2[m]["ALL"][k] - clean4[m]["ALL"][k]) < TOL for m in MODELS for k in ("wer", "cer"))
    noise = Counter((r["condition"], r["status"], r["was_clipped"]) for r in read_csv(STAGE4 / "noise_log.csv"))
    final["stage4"]["noise_log"] = {f"{c}|{s}|clipped={k}": n for (c, s, k), n in sorted(noise.items())}

    # Entity matching (automatic, unvalidated) v2/v3.
    final["entities"] = {v: entity_rates(STAGE2 / f"entity_matches_{v}.csv") for v in ("v2", "v3")}

    # Entity agreement vs AI-assisted (GPT-6) labels.
    final["entity_agreement"] = read_json(STAGE2 / "entity_review_agreement_v2.json")

    # Layer 2 candidates.
    l2 = read_json(STAGE2 / "layer2_summary_v3.json")
    cands = read_csv(STAGE2 / "layer2_candidates_v3.csv")
    per_cat = Counter()
    for c in cands:
        for cat in c["categories"].split(";"):
            per_cat[(c["model"], cat)] += 1
    final["layer2"] = {
        "n_candidates_clean": l2["n_candidates"], "by_op": l2["by_op"],
        "not_overlapping_annotated_entities": l2["not_overlapping_annotated_entities"],
        "by_model_category": {f"{m} | {cat}": n for (m, cat), n in sorted(per_cat.items())},
        "stage4_by_condition_model": read_json(STAGE4 / "layer2_summary_v3.json")["by_condition_model"],
    }

    # Layer 2 annotations (audited version) + human vs prior AI on the same rows.
    audited = read_csv(STAGE2 / "clinasr_layer2_104_consistency_audited.csv")
    final["layer2_annotations"] = summarize_audited(audited)
    final["layer2_annotations"]["provenance"] = read_json(STAGE2 / "layer2_audit_summary_v2.json")["provenance_note"]
    human = {r["candidate_id"]: r for r in read_csv(STAGE2 / "clinasr_human_review_final_30.csv")}
    prior_ai = {r["candidate_id"]: r for r in read_csv(STAGE2 / "layer2_review_sheet_v3_labeled.csv")}
    pairs = Counter((prior_ai[i]["review_label"], h["review_label"]) for i, h in human.items())
    final["layer2_annotations"]["human_vs_prior_ai_same_30_rows"] = {
        "n": len(human), "agree": sum(n for (a, b), n in pairs.items() if a == b),
        "pairs_prior_ai_to_human": {f"{a}->{b}": n for (a, b), n in sorted(pairs.items())},
        "note": "Same 30 rows. The human reviewer may have seen the AI labels; not an independent double-annotation study.",
    }

    final["source_files_sha256"] = dict(sorted(sources.items()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(final, indent=2, default=str), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}; all recomputed metrics matched the stored summaries; "
          f"stage4 clean reproduces stage2 v2: {final['stage4']['clean_reproduces_stage2_v2']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
