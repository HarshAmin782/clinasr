"""Paired tiny.en vs base.en comparison on a fetched Stage 2 output folder.

Usage:
    python scripts/analyze_stage2.py results/fetched_<timestamp>

Reads predictions.csv, writes paired_comparison.json next to it.
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

MODEL_A = "openai/whisper-tiny.en"
MODEL_B = "openai/whisper-base.en"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fetched_dir", type=Path)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    with open(args.fetched_dir / "predictions.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    result = paired_bootstrap(rows, MODEL_A, MODEL_B, n_bootstrap=args.n_bootstrap, seed=args.seed)
    out = args.fetched_dir / "paired_comparison.json"
    out.write_text(json.dumps(dataclasses.asdict(result), indent=2), encoding="utf-8")

    print(f"n={result.n_samples}  WER {MODEL_A}={result.wer_a:.4f}  {MODEL_B}={result.wer_b:.4f}")
    print(f"difference (b - a) = {result.difference:+.4f}, 95% paired bootstrap CI [{result.ci_low:+.4f}, {result.ci_high:+.4f}]")
    print(f"per-sample: b better {result.n_b_better}, a better {result.n_a_better}, tied {result.n_tied}")
    print("Subset-level result only; strata are over-sampled (not a population estimate).")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
