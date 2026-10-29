"""Validate an audited Layer 2 annotation file against the original review
sheet (read-only) and write a span-level vs context-level summary.

Usage:
    python scripts/layer2_audit_summary.py <original_review_sheet.csv> <audited.csv> \
        --annotation-version layer2-annotations-v2-audited --out <summary.json> [--provenance-note "..."]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinasr.layer2_audit import summarize_audited, validate_audited_against_original  # noqa: E402


def _read(path: Path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("original", type=Path)
    p.add_argument("audited", type=Path)
    p.add_argument("--annotation-version", required=True)
    p.add_argument("--provenance-note", default="")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    original, audited = _read(args.original), _read(args.audited)
    validate_audited_against_original(original, audited)
    summary = summarize_audited(audited)
    summary["annotation_version"] = args.annotation_version
    summary["source_file"] = str(args.audited)
    summary["source_sha256"] = hashlib.sha256(args.audited.read_bytes()).hexdigest()
    summary["original_review_sheet"] = str(args.original)
    summary["provenance_note"] = args.provenance_note
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
