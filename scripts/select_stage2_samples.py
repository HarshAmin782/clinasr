"""Compute the Stage 2 stratified selection locally from a sample_index.csv
produced by scripts/analyze_entity_coverage.py (no network, no GPU).

Prints the allocation per stratum and the selection fingerprint; writes
the selection to kaggle/stage2_selection.json for review. The fingerprint
must then be copied into clinasr.stage2.STAGE2_EXPECTED_FINGERPRINT so the
Kaggle run aborts if it would evaluate a different set.

Usage:
    python scripts/select_stage2_samples.py results/<run_id>/sample_index.csv
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

from clinasr import stage2  # noqa: E402

OUTPUT = ROOT / "kaggle" / "stage2_selection.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_index", type=Path)
    args = parser.parse_args(argv)

    with open(args.sample_index, newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    selection = stage2.select(records)

    payload = dataclasses.asdict(selection)
    payload["source_index"] = str(args.sample_index)
    payload["dataset_revision"] = stage2.DATASET_REVISION
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"records: {selection.n_records_seen}, eligible: {selection.n_eligible}, selected: {len(selection.sample_ids)}")
    for stratum, k in selection.allocation.items():
        print(f"  {stratum:45s} {k:4d} / {selection.stratum_population[stratum]}")
    print(f"fingerprint: {selection.fingerprint}")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
