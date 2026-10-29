"""Opt-in CLI: run clinasr.schema_audit.audit_hf_dataset against the real,
live EkaCare dataset and persist the actual result into an isolated run
directory under results/, per AGENTS.md's run-isolation and
"write real outputs, never fabricate" requirements.

Requires network access and the `datasets` package (see
requirements-dev.txt's optional extras note). Not run automatically by any
other script in this repository -- this is the explicit, documented first
step of Phase 1 (docs/ISSUES.md #4).

Usage:
    python scripts/audit_dataset_schema.py
    python scripts/audit_dataset_schema.py --n 20
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinasr.run_manifest import get_git_commit, new_run  # noqa: E402
from clinasr.schema_audit import audit_hf_dataset  # noqa: E402

DEFAULT_DATASET_ID = "ekacare/eka-medical-asr-evaluation-dataset"
DEFAULT_CONFIG = "en"
DEFAULT_SPLIT = "test"

# PROJECT_SPEC.md's originally expected fields, kept here (not re-derived
# from the spec file) so a change to the spec doesn't silently change what
# this script reports as "expected" without a deliberate edit.
EXPECTED_FIELDS = [
    "audio",
    "text",
    "session_id",
    "speaker",
    "recording_context",
    "type_concept",
    "medical_entities",
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--n", type=int, default=20, help="Number of records to stream and inspect.")
    args = parser.parse_args(argv)

    handle = new_run(ROOT / "results", git_commit=get_git_commit(ROOT))
    logger = handle.get_logger()
    logger.info("Starting schema audit: dataset=%s config=%s split=%s n=%d", args.dataset_id, args.config, args.split, args.n)

    report = audit_hf_dataset(args.dataset_id, args.config, args.split, n=args.n)
    missing_expected = report.missing_expected(EXPECTED_FIELDS)

    report_dict = dataclasses.asdict(report)
    output_path = handle.run_dir / "schema_audit_report.json"
    output_path.write_text(json.dumps(report_dict, indent=2, default=repr), encoding="utf-8")

    handle.write_manifest(
        {
            "git_commit": get_git_commit(ROOT),
            "command": "scripts/audit_dataset_schema.py",
            "dataset_id": args.dataset_id,
            "dataset_config": args.config,
            "dataset_split": args.split,
            "n_records_requested": args.n,
            "n_records_inspected": report.n_records_inspected,
            "expected_fields": EXPECTED_FIELDS,
            "expected_fields_missing": missing_expected,
            "output_file": str(output_path.relative_to(ROOT)),
        }
    )

    print(f"Run directory: {handle.run_dir.relative_to(ROOT)}")
    print(f"Inspected {report.n_records_inspected} real records from {args.dataset_id} ({args.config}/{args.split}).")
    print("Fields found:")
    for name, field_report in report.fields.items():
        print(
            f"  {name}: types={field_report.observed_types} "
            f"present={field_report.n_present} missing={field_report.n_missing}"
        )
    if missing_expected:
        print(f"WARNING: expected fields not found in the real schema: {missing_expected}")
    else:
        print("All PROJECT_SPEC.md expected fields are present (type/representation may still differ -- see the report).")

    logger.info("Schema audit complete. Report written to %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
