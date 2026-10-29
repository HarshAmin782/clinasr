"""Re-score a fetched run's predictions.csv under a named normalization
version (no GPU, no re-transcription). Hypotheses are never changed; only
the declared normalization applied to BOTH reference and hypothesis.

Usage:
    python scripts/rescore.py results/fetched_<ts> --normalization v2

Writes summary_metrics_<version>.csv and paired_comparison_<version>.json
into the same folder.
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
from clinasr.normalization import DEFAULT_NORMALIZATION, NORMALIZATION_V1  # noqa: E402
from clinasr.stage2 import summary_rows  # noqa: E402

VERSIONS = {"v1": NORMALIZATION_V1, "v2": DEFAULT_NORMALIZATION}
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

    summaries = []
    for model in sorted({r["model"] for r in rows}):
        summaries.extend(summary_rows([r for r in rows if r["model"] == model], model, config))
    out_csv = args.fetched_dir / f"summary_metrics_{config.version}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    comparison = paired_bootstrap(rows, MODEL_A, MODEL_B, config=config)
    out_json = args.fetched_dir / f"paired_comparison_{config.version}.json"
    payload = dataclasses.asdict(comparison)
    payload["normalization"] = dataclasses.asdict(config)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for s in summaries:
        if s["stratum"] == "ALL":
            print(f"[{config.version}] {s['model']}: WER={s['wer']:.4f} CER={s['cer']:.4f}")
    print(f"[{config.version}] base - tiny = {comparison.difference:+.4f} [{comparison.ci_low:+.4f}, {comparison.ci_high:+.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
