"""Re-score a fetched run's predictions.csv under a named normalization
version (no GPU, no re-transcription). Hypotheses are never changed; only
the declared normalization applied to BOTH reference and hypothesis.

Usage:
    python scripts/rescore.py results/fetched_<ts> --normalization v2

Writes summary_metrics_<version>.csv, paired_comparison_<version>.json and,
if entity_matches_unvalidated.csv exists, entity_matches_<version>.csv +
entity_match_rates_<version>_UNVALIDATED.json into the same folder. Existing
files for other versions are never touched.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinasr.analysis import paired_bootstrap  # noqa: E402
from clinasr.entity_review import rematch  # noqa: E402
from clinasr.normalization import NORMALIZATION_VERSIONS  # noqa: E402
from clinasr.stage2 import summary_rows  # noqa: E402

VERSIONS = NORMALIZATION_VERSIONS
MODEL_A = "openai/whisper-tiny.en"
MODEL_B = "openai/whisper-base.en"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fetched_dir", type=Path)
    parser.add_argument("--normalization", choices=sorted(VERSIONS), default="v2")
    args = parser.parse_args(argv)
    config = VERSIONS[args.normalization]

    with open(args.fetched_dir / "predictions.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Score each (condition, model) separately; Stage 4 files hold several
    # noise conditions for the same sample ids.
    conditions = sorted({r.get("condition", "clean") for r in rows})
    summaries, comparisons = [], {}
    for cond in conditions:
        cond_rows = [r for r in rows if r.get("condition", "clean") == cond]
        for model in sorted({r["model"] for r in cond_rows}):
            for s in summary_rows([r for r in cond_rows if r["model"] == model], model, config):
                s["condition"] = cond
                summaries.append(s)
        comparisons[cond] = paired_bootstrap(cond_rows, MODEL_A, MODEL_B, config=config)
    out_csv = args.fetched_dir / f"summary_metrics_{config.version}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    comparison = comparisons.get("clean") or next(iter(comparisons.values()))
    out_json = args.fetched_dir / f"paired_comparison_{config.version}.json"
    payload = {cond: dataclasses.asdict(c) for cond, c in comparisons.items()} if len(comparisons) > 1 else dataclasses.asdict(comparison)
    if len(comparisons) > 1:
        payload = {"by_condition": payload}
    payload["normalization"] = dataclasses.asdict(config)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    entity_path = args.fetched_dir / "entity_matches_unvalidated.csv"
    if entity_path.exists():
        with open(entity_path, newline="", encoding="utf-8") as f:
            entity_rows = list(csv.DictReader(f))
        hyps = {(r["sample_id"], r["model"]): r["hypothesis"] for r in rows if r.get("condition", "clean") == "clean"}
        matched = rematch(entity_rows, hyps, config)
        with open(args.fetched_dir / f"entity_matches_{config.version}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(matched[0].keys()))
            writer.writeheader()
            writer.writerows(matched)
        rates = {}
        for r in matched:
            for key in ("ALL", r["entity_type"]):
                cell = rates.setdefault(r["model"], {}).setdefault(key, [0, 0])
                cell[0] += r["match_status"] != "omitted"
                cell[1] += 1
            if r["dose_preserved"] is not None:
                cell = rates[r["model"]].setdefault("dose_preserved", [0, 0])
                cell[0] += bool(r["dose_preserved"])
                cell[1] += 1
        out = {"validated": False, "normalization": config.version,
               "match_rate": {m: {k: {"matched": a, "n": n, "rate": a / n} for k, (a, n) in v.items()} for m, v in rates.items()}}
        (args.fetched_dir / f"entity_match_rates_{config.version}_UNVALIDATED.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        for m, v in out["match_rate"].items():
            print(f"[{config.version}] {m} entity match_rate (unvalidated): {v['ALL']['rate']:.4f}; dose_preserved: {v.get('dose_preserved', {}).get('rate')}")

    for s in summaries:
        if s["stratum"] == "ALL":
            print(f"[{config.version}] {s['condition']:9s} {s['model']}: WER={s['wer']:.4f} CER={s['cer']:.4f}")
    for cond, c in comparisons.items():
        print(f"[{config.version}] {cond:9s} base - tiny = {c.difference:+.4f} [{c.ci_low:+.4f}, {c.ci_high:+.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
