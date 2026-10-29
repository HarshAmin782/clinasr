"""Opt-in CLI: medical-entity annotation coverage + stratification summary
over the real EkaCare test split (no GPU, no audio decoding).

Writes into an isolated results/<run_id>/ directory:
- entity_coverage_report.json: annotation status counts, per-category
  entity/sample counts, drug and dosage-like coverage, offset alignment,
  stratification counts by recording_context x type_concept.
- sample_index.csv: one row per record (file_name, session_id, strata,
  duration, annotation status, entity counts). No audio, no transcript
  text. This is the input Stage 2's stratified sampler draws from.
- manifest.json / run.log via clinasr.run_manifest.

Fails explicitly (non-zero exit, nothing silently deduplicated) if any
file_name is missing or repeated.

Usage:
    python scripts/analyze_entity_coverage.py            # whole split
    python scripts/analyze_entity_coverage.py --n 300    # first 300 only
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

from clinasr.medical_entities import (  # noqa: E402
    DRUG_CATEGORY,
    analyze_coverage,
    assert_unique_ids,
    is_dosage_like,
    parse_medical_entities,
    summarize_strata,
)
from clinasr.run_manifest import get_git_commit, new_run  # noqa: E402

DATASET_ID = "ekacare/eka-medical-asr-evaluation-dataset"
CONFIG = "en"
SPLIT = "test"
STRATA_KEYS = ["recording_context", "type_concept"]
INDEX_FIELDS = [
    "file_name",
    "session_id",
    "recording_context",
    "type_concept",
    "duration",
    "annotation_status",
    "n_entities",
    "n_drug_entities",
    "n_dosage_like_drug_entities",
    "categories",
]


def stream_records(n=None):
    import datasets

    stream = datasets.load_dataset(DATASET_ID, CONFIG, split=SPLIT, streaming=True)
    audio_columns = [c for c, f in stream.features.items() if isinstance(f, datasets.Audio)]
    # remove_columns alone still decodes audio on a streaming dataset (needs
    # torchcodec); disable decoding first, then drop the column.
    for column in audio_columns:
        stream = stream.cast_column(column, datasets.Audio(decode=False))
    if audio_columns:
        stream = stream.remove_columns(audio_columns)
    for i, record in enumerate(stream):
        if n is not None and i >= n:
            break
        yield dict(record)


def index_row(record):
    parsed = parse_medical_entities(record.get("medical_entities"))
    drugs = [e for e in parsed.entities if e.entity_type == DRUG_CATEGORY]
    return {
        "file_name": record.get("file_name"),
        "session_id": record.get("session_id"),
        "recording_context": record.get("recording_context"),
        "type_concept": record.get("type_concept"),
        "duration": record.get("duration"),
        "annotation_status": parsed.status,
        "n_entities": len(parsed.entities),
        "n_drug_entities": len(drugs),
        "n_dosage_like_drug_entities": sum(1 for e in drugs if is_dosage_like(e.text)),
        "categories": ";".join(sorted({e.entity_type for e in parsed.entities})),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=None, help="Only the first N records (default: whole split).")
    args = parser.parse_args(argv)

    handle = new_run(ROOT / "results", git_commit=get_git_commit(ROOT))
    logger = handle.get_logger()
    logger.info("Entity coverage: dataset=%s/%s/%s n=%s", DATASET_ID, CONFIG, SPLIT, args.n)

    records = list(stream_records(args.n))
    logger.info("Streamed %d records", len(records))

    assert_unique_ids([r.get("file_name") for r in records])  # raises on duplicates/missing

    coverage = analyze_coverage(records)
    strata = summarize_strata(records, STRATA_KEYS)
    sessions = {r.get("session_id") for r in records}
    durations = sorted(float(r["duration"]) for r in records if isinstance(r.get("duration"), (int, float)))

    report = {
        "dataset": {"id": DATASET_ID, "config": CONFIG, "split": SPLIT, "n_requested": args.n},
        "n_records": len(records),
        "n_unique_file_names": len(records),
        "n_unique_sessions": len(sessions),
        "duration_seconds": {
            "total": sum(durations),
            "min": durations[0] if durations else None,
            "median": durations[len(durations) // 2] if durations else None,
            "max": durations[-1] if durations else None,
        },
        "coverage": dataclasses.asdict(coverage),
        "strata": strata,
        "notes": [
            "Dosage counts use a regex heuristic (number + dose unit) inside entity text; "
            "there is no dosage category in the annotations.",
            "No entity precision/recall is reported here or anywhere until matching rules are validated.",
        ],
    }
    report_path = handle.run_dir / "entity_coverage_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    index_path = handle.run_dir / "sample_index.csv"
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(index_row(record))

    handle.write_manifest(
        {
            "git_commit": get_git_commit(ROOT),
            "command": "scripts/analyze_entity_coverage.py",
            "n_requested": args.n,
            "n_records": len(records),
            "outputs": [str(report_path.relative_to(ROOT)), str(index_path.relative_to(ROOT))],
        }
    )

    c = coverage
    print(f"Run directory: {handle.run_dir.relative_to(ROOT)}")
    print(f"Records: {len(records)} (unique file_name: yes; sessions: {len(sessions)})")
    print(f"Annotation status: {c.status_counts}")
    print(f"Samples with >=1 entity: {c.n_samples_with_entities}; entities: {c.n_entities}")
    print(f"Entities per category: {c.category_entity_counts}")
    print(f"Samples per category: {c.category_sample_counts}")
    print(f"Samples with drugs: {c.n_samples_with_drugs}")
    print(
        f"Dosage-like drug entities (heuristic): {c.n_dosage_like_drug_entities} "
        f"in {c.n_samples_with_dosage_like_drug_entity} samples; "
        f"dosage-like in other categories: {c.n_dosage_like_entities_other_categories}"
    )
    print(
        f"Offsets: {c.n_entities_with_offsets} checked, {c.n_offsets_aligned} aligned, "
        f"{c.n_offsets_misaligned} misaligned"
    )
    print(f"Problems: {c.problem_counts}; unusable samples: {len(c.unusable_sample_ids)}")
    print(f"Strata per key: {strata['per_key']}")
    logger.info("Done: %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
