"""Layer 2 candidate critical-error extraction for a fetched run.

Usage:
    python scripts/layer2_candidates.py results/fetched_<ts> \
        [--entities results/fetched_<stage2>/entity_matches_unvalidated.csv] \
        [--normalization v3] [--per-category 10] [--seed 20261029]

Writes (never overwriting other versions):
- layer2_candidates_<v>.csv    every flagged span, with context + provenance
- layer2_review_sheet_<v>.csv  seeded stratified sample with blank
                               review_label / reviewer_type / reviewer_id / notes
- layer2_summary_<v>.json      counts by condition, model, category, op

Flags are automatic heuristics, NOT clinical judgements. review_label is
blank until a reviewer fills it in and records reviewer_type
(ai_assisted / human / clinician).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinasr.critical_errors import (  # noqa: E402
    DETECTOR_VERSION,
    FLAG_PROVENANCE,
    REVIEW_LABELS,
    REVIEWER_TYPES,
    extract_candidates,
    select_for_review,
)
from clinasr.normalization import NORMALIZATION_VERSIONS  # noqa: E402

REVIEW_FIELDS = ["review_label", "reviewer_type", "reviewer_id", "notes"]


def _row(c):
    d = dataclasses.asdict(c)
    d["categories"] = ";".join(c.categories)
    return d


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fetched_dir", type=Path)
    p.add_argument("--entities", type=Path, default=None, help="entity_matches_unvalidated.csv (defaults to the one in fetched_dir if present)")
    p.add_argument("--normalization", choices=sorted(NORMALIZATION_VERSIONS), default="v3")
    p.add_argument("--per-category", type=int, default=10)
    p.add_argument("--seed", type=int, default=20261029)
    args = p.parse_args(argv)
    config = NORMALIZATION_VERSIONS[args.normalization]

    with open(args.fetched_dir / "predictions.csv", newline="", encoding="utf-8") as f:
        preds = list(csv.DictReader(f))
    entity_path = args.entities or (args.fetched_dir / "entity_matches_unvalidated.csv")
    entities = defaultdict(set)
    if entity_path.exists():
        with open(entity_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                entities[r["sample_id"]].add(r["entity_text"])

    cands = []
    for r in preds:
        cands += extract_candidates(
            r["sample_id"], r["model"], r["reference"], r["hypothesis"],
            condition=r.get("condition", "clean"), entity_texts=entities.get(r["sample_id"], ()), config=config,
        )
    if not cands:
        print("no candidates flagged")
        return 0

    v = config.version
    rows = [_row(c) for c in cands]
    fields = list(rows[0].keys())
    with open(args.fetched_dir / f"layer2_candidates_{v}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Review sample drawn from the clean condition only (the noisy conditions
    # are dominated by hallucination loops; review those separately if needed).
    clean = [c for c in cands if c.condition == "clean"] or cands
    picked = select_for_review(clean, per_category=args.per_category, seed=args.seed)
    with open(args.fetched_dir / f"layer2_review_sheet_{v}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields + REVIEW_FIELDS)
        w.writeheader()
        for c in picked:
            w.writerow({**_row(c), **{k: "" for k in REVIEW_FIELDS}})

    def count(key):
        return dict(sorted(Counter(key(c) for c in cands).items()))

    per_cat = Counter()
    for c in cands:
        for cat in c.categories:
            per_cat[(c.condition, c.model, cat)] += 1
    summary = {
        "detector_version": DETECTOR_VERSION,
        "flagged_by": FLAG_PROVENANCE,
        "normalization": v,
        "n_candidates": len(cands),
        "n_predictions_scanned": len(preds),
        "by_condition_model": {f"{k[0]} | {k[1]}": n for k, n in sorted(Counter((c.condition, c.model) for c in cands).items())},
        "by_category": {f"{k[0]} | {k[1]} | {k[2]}": n for k, n in sorted(per_cat.items())},
        "by_op": count(lambda c: c.op),
        "not_overlapping_annotated_entities": sum(not c.overlaps_annotated_entity for c in cands),
        "review_sheet": {"rows": len(picked), "design": f"clean condition only; up to {args.per_category} per (model, first category); seed {args.seed}; stratified, NOT representative",
                         "allowed_review_labels": list(REVIEW_LABELS), "allowed_reviewer_types": list(REVIEWER_TYPES)},
        "interpretation": "Counts are heuristic flags for review, not error rates and not clinical findings.",
    }
    (args.fetched_dir / f"layer2_summary_{v}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n_candidates", "by_condition_model", "by_op", "not_overlapping_annotated_entities")}, indent=1))
    print(f"review sheet: {len(picked)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
