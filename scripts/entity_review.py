"""Stage 3 entity-label review.

  sheet:     python scripts/entity_review.py sheet results/fetched_<ts> [--per-cell 25]
             -> entity_matches_v2.csv (all rows, rematched under v2)
             -> entity_match_rates_v2_UNVALIDATED.json
             -> entity_review_sheet.csv (stratified sample; fill `human_label`)
  agreement: python scripts/entity_review.py agreement <labeled sheet> --label-source "AI-assisted (OpenAI GPT-6)" [--flags flags.csv] [--rematch-with v3]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinasr.entity_review import agreement, rematch, select_review_rows  # noqa: E402

SHEET_FIELDS = ["review_id", "model", "sample_id", "entity_type", "entity_text", "reference", "hypothesis",
                "auto_label", "dosage_like", "human_label", "notes"]


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def cmd_sheet(args):
    d = args.fetched_dir
    preds = _read(d / "predictions.csv")
    hyps = {(p["sample_id"], p["model"]): p["hypothesis"] for p in preds}
    refs = {p["sample_id"]: p["reference"] for p in preds}
    rows = rematch(_read(d / "entity_matches_unvalidated.csv"), hyps)
    _write(d / "entity_matches_v2.csv", rows, list(rows[0].keys()))

    rates = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        for key in ("ALL", r["entity_type"], "dosage_like" if r["dosage_like"] else None):
            if key:
                cell = rates[r["model"]][key]
                cell[0] += r["match_status"] != "omitted"
                cell[1] += 1
    summary = {
        "validated": False,
        "normalization": "v2",
        "definition": "fraction of reference entities labelled exact or normalized by the automatic matcher",
        "match_rate": {m: {k: {"matched": a, "n": n, "rate": a / n} for k, (a, n) in sorted(v.items())} for m, v in rates.items()},
    }
    (d / "entity_match_rates_v2_UNVALIDATED.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    picked = select_review_rows(rows, per_cell=args.per_cell, seed=args.seed)
    sheet = [
        {"review_id": i, "model": r["model"], "sample_id": r["sample_id"], "entity_type": r["entity_type"],
         "entity_text": r["entity_text"], "reference": refs[r["sample_id"]], "hypothesis": hyps[(r["sample_id"], r["model"])],
         "auto_label": r["match_status"], "dosage_like": r["dosage_like"], "human_label": "", "notes": ""}
        for i, r in enumerate(picked)
    ]
    _write(d / "entity_review_sheet.csv", sheet, SHEET_FIELDS)
    for m, v in summary["match_rate"].items():
        print(m, {k: round(x["rate"], 3) for k, x in v.items()})
    print(f"review sheet: {len(sheet)} rows -> {d / 'entity_review_sheet.csv'}")
    print("Fill human_label with preserved / not_preserved / unclear, then run the agreement command.")
    return 0


def _read_any(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def cmd_agreement(args):
    rows = _read_any(args.sheet)
    for r in rows:
        r["match_status"] = r["auto_label"]
    flags = {f["review_id"]: f for f in _read_any(args.flags)} if args.flags else {}
    unknown = sorted(set(flags) - {r["review_id"] for r in rows})
    if unknown:
        raise SystemExit(f"flags reference unknown review_id(s): {unknown}")

    report = {
        "annotation_source": args.label_source,
        "annotations_are_independent_human_labels": bool(args.independent_human),
        "sheet": str(args.sheet),
        "automatic_labels": "auto_label column (matcher output stored in the sheet)",
        "sample_design": "stratified by model x automatic label (25 per cell); NOT representative of the dataset",
        "original_labels": agreement(rows),
        "flagged_rows": [{k: f.get(k, "") for k in ("review_id", "flag", "reason")} for f in flags.values()],
    }
    if flags:
        report["sensitivity_excluding_flagged"] = agreement([r for r in rows if r["review_id"] not in flags])
        alt = []
        for r in rows:
            f = flags.get(r["review_id"])
            if f and f.get("alternative_label_for_sensitivity") in ("preserved", "not_preserved", "unclear"):
                r = dict(r, human_label=f["alternative_label_for_sensitivity"])
            alt.append(r)
        report["sensitivity_with_alternative_labels"] = agreement(alt)
    if args.rematch_with:
        from clinasr.normalization import NORMALIZATION_VERSIONS
        from clinasr.entities import EntitySpan, match_entity

        cfg = NORMALIZATION_VERSIONS[args.rematch_with]
        rem = [dict(r, match_status=match_entity(EntitySpan(text=r["entity_text"]), r["hypothesis"], cfg).status) for r in rows]
        report[f"automatic_labels_recomputed_{cfg.version}"] = agreement(rem)

    out = args.out or args.sheet.with_name(args.sheet.stem + "_agreement.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("fetched_dir", type=Path)
    s.add_argument("--per-cell", type=int, default=25)
    s.add_argument("--seed", type=int, default=20261029)
    a = sub.add_parser("agreement")
    a.add_argument("sheet", type=Path)
    a.add_argument("--label-source", required=True, help='e.g. "AI-assisted (OpenAI GPT-6)" or "human"')
    a.add_argument("--independent-human", action="store_true",
                   help="Set ONLY if a person labelled every row without model assistance.")
    a.add_argument("--flags", type=Path, default=None, help="CSV: review_id, flag, alternative_label_for_sensitivity, reason")
    a.add_argument("--rematch-with", choices=["v1", "v2", "v3"], default=None)
    a.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    return cmd_sheet(args) if args.cmd == "sheet" else cmd_agreement(args)


if __name__ == "__main__":
    sys.exit(main())
