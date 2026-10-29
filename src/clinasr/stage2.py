"""Stage 2: Whisper-tiny.en vs Whisper-base.en on one fixed stratified subset.

Design (docs/WORKLOG.md has the rationale and the real population counts):
- Dataset pinned to a Hub revision; models pinned to Hub revisions.
- Strata: recording_context x type_concept. 300 samples, at least 20 per
  stratum, seed STAGE2_SEED, hash-ranked selection (clinasr.sampling).
- The selection is recomputed on Kaggle from streamed metadata and must
  match STAGE2_EXPECTED_FINGERPRINT (computed locally by
  scripts/select_stage2_samples.py and reviewed before any GPU run). A
  mismatch aborts the run before inference.
- Both models decode the identical audio with identical greedy settings.
- Entity matching is written out per entity but labelled unvalidated.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from clinasr.config import ExperimentConfig, ModelSpec, NoiseConfig, SampleSelection, resolve_installed_versions
from clinasr.medical_entities import evaluate_sample_entities, parse_medical_entities
from clinasr.metrics import compute_corpus_metric, compute_metrics
from clinasr.normalization import DEFAULT_NORMALIZATION
from clinasr.run_manifest import get_git_commit, new_run
from clinasr.sampling import Selection, stratified_select
from clinasr.smoke_test import bootstrap_all

DATASET_ID = "ekacare/eka-medical-asr-evaluation-dataset"
DATASET_CONFIG = "en"
DATASET_SPLIT = "test"
DATASET_REVISION = "433e66037c8aebb8f396b938624a5ab6afc4ca36"
MODELS = {
    "openai/whisper-tiny.en": "87c7102498dcde7456f24cfd30239ca606ed9063",
    "openai/whisper-base.en": "911407f4214e0e1d82085af863093ec0b66f9cd6",
}
STRATA_KEYS = ["recording_context", "type_concept"]
STAGE2_TOTAL = 300
STAGE2_MIN_PER_STRATUM = 20
STAGE2_SEED = 20261029
# Set from the local, reviewed selection (scripts/select_stage2_samples.py).
STAGE2_EXPECTED_FINGERPRINT = "8b345d92bc204dedebe465065dac1993ac1e48c6cd55f408a12e47117d84a6c3"
BATCH_SIZE = 16
GENERATE_KWARGS = {"num_beams": 1, "do_sample": False}


def select(records: Sequence[Dict[str, Any]]) -> Selection:
    return stratified_select(
        records,
        STRATA_KEYS,
        total=STAGE2_TOTAL,
        seed=STAGE2_SEED,
        min_per_stratum=STAGE2_MIN_PER_STRATUM,
    )


def _stream(decode_audio: bool):
    import datasets

    stream = datasets.load_dataset(
        DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT, streaming=True, revision=DATASET_REVISION
    )
    if not decode_audio:
        stream = stream.cast_column("audio", datasets.Audio(decode=False)).remove_columns(["audio"])
    return stream


def load_metadata() -> List[Dict[str, Any]]:
    return [dict(r) for r in _stream(decode_audio=False)]


def load_selected_audio(sample_ids: Sequence[str]) -> List[Dict[str, Any]]:
    wanted = set(sample_ids)
    found: Dict[str, Dict[str, Any]] = {}
    for record in _stream(decode_audio=True):
        if record["file_name"] in wanted:
            found[record["file_name"]] = dict(record)
            if len(found) == len(wanted):
                break
    missing = wanted - set(found)
    if missing:
        raise RuntimeError(f"{len(missing)} selected samples not found in the stream: {sorted(missing)[:5]}")
    return [found[sid] for sid in sorted(found)]


def transcribe(records: Sequence[Dict[str, Any]], model_id: str, revision: str) -> Dict[str, Any]:
    import torch
    from transformers import pipeline

    use_cuda = torch.cuda.is_available()
    asr = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        revision=revision,
        device=0 if use_cuda else -1,
        dtype=torch.float16 if use_cuda else torch.float32,
    )
    inputs = [{"array": r["audio"]["array"], "sampling_rate": r["audio"]["sampling_rate"]} for r in records]
    start = time.perf_counter()
    outputs = asr(inputs, batch_size=BATCH_SIZE, generate_kwargs=dict(GENERATE_KWARGS))
    elapsed = time.perf_counter() - start
    return {
        "hypotheses": [o["text"] for o in outputs],
        "seconds": elapsed,
        "device": "cuda" if use_cuda else "cpu",
        "dtype": "float16" if use_cuda else "float32",
    }


def prediction_rows(records: Sequence[Dict[str, Any]], hypotheses: Sequence[str], model_id: str) -> List[Dict[str, Any]]:
    if len(records) != len(hypotheses):
        raise ValueError("records and hypotheses must be the same length")
    return [
        {
            "sample_id": r["file_name"],
            "model": model_id,
            "condition": "clean",
            "recording_context": r.get("recording_context"),
            "type_concept": r.get("type_concept"),
            "session_id": r.get("session_id"),
            "duration": r.get("duration"),
            "reference": r.get("text"),
            "hypothesis": h,
        }
        for r, h in zip(records, hypotheses)
    ]


def summary_rows(rows: Sequence[Dict[str, Any]], model_id: str, config=DEFAULT_NORMALIZATION) -> List[Dict[str, Any]]:
    """Corpus WER/CER overall and per stratum for one model's rows."""
    groups: Dict[str, List[Dict[str, Any]]] = {"ALL": list(rows)}
    for row in rows:
        groups.setdefault(f"{row['recording_context']} | {row['type_concept']}", []).append(row)
    out = []
    for stratum, group in groups.items():
        metrics = compute_metrics([g["sample_id"] for g in group], [g["reference"] for g in group], [g["hypothesis"] for g in group], config)
        corpus = compute_corpus_metric(metrics)
        out.append(
            {
                "model": model_id,
                "condition": "clean",
                "normalization": config.version,
                "stratum": stratum,
                "n_samples": corpus.n_samples_total,
                "n_valid": corpus.n_samples_valid,
                "wer": corpus.wer,
                "cer": corpus.cer,
            }
        )
    return out


def entity_rows(records: Sequence[Dict[str, Any]], hypotheses: Sequence[str], model_id: str) -> List[Dict[str, Any]]:
    rows = []
    for record, hypothesis in zip(records, hypotheses):
        parsed = parse_medical_entities(record.get("medical_entities"))
        result = evaluate_sample_entities(record["file_name"], parsed, hypothesis)
        for row in result["rows"]:
            row = dict(row)
            row.pop("hypothesis", None)  # already in predictions.csv
            row["model"] = model_id
            row["annotation_status"] = parsed.status
            row["matching_validated"] = False
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_stage2(output_dir: Path, expected_fingerprint: Optional[str] = None) -> int:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = expected_fingerprint or STAGE2_EXPECTED_FINGERPRINT

    bootstrap = bootstrap_all()
    print(f"bootstrap: {bootstrap}")
    if not all(bootstrap.values()):
        print("ERROR: package bootstrap failed; aborting before inference.")
        return 1

    handle = new_run(output_dir / "run", git_commit=get_git_commit())
    logger = handle.get_logger()

    metadata = load_metadata()
    selection = select(metadata)
    (output_dir / "selection.json").write_text(json.dumps(dataclasses.asdict(selection), indent=2), encoding="utf-8")
    print(f"selected {len(selection.sample_ids)} samples, fingerprint {selection.fingerprint}")
    if selection.fingerprint != expected:
        print(f"ERROR: selection fingerprint {selection.fingerprint} != expected {expected}; aborting before inference.")
        return 2

    records = load_selected_audio(selection.sample_ids)
    all_predictions, all_summaries, all_entities, timings = [], [], [], {}
    for model_id, revision in MODELS.items():
        print(f"transcribing with {model_id}@{revision[:10]} ...")
        result = transcribe(records, model_id, revision)
        timings[model_id] = {k: result[k] for k in ("seconds", "device", "dtype")}
        preds = prediction_rows(records, result["hypotheses"], model_id)
        all_predictions.extend(preds)
        all_summaries.extend(summary_rows(preds, model_id))
        all_entities.extend(entity_rows(records, result["hypotheses"], model_id))
        logger.info("%s done in %.1fs on %s", model_id, result["seconds"], result["device"])

    _write_csv(output_dir / "predictions.csv", all_predictions)
    _write_csv(output_dir / "summary_metrics.csv", all_summaries)
    _write_csv(output_dir / "entity_matches_unvalidated.csv", all_entities)

    config = ExperimentConfig(
        dataset_id=DATASET_ID,
        dataset_config=DATASET_CONFIG,
        dataset_split=DATASET_SPLIT,
        seed=STAGE2_SEED,
        sample_selection=SampleSelection(
            method="stratified_hash_ranked",
            size=len(selection.sample_ids),
            stratify_by=list(STRATA_KEYS),
            sample_ids=list(selection.sample_ids),
        ),
        models=[ModelSpec(model_id=m, resolved_revision=r) for m, r in MODELS.items()],
        normalization=DEFAULT_NORMALIZATION,
        noise=NoiseConfig(snrs_db=[None]),
        software_versions=resolve_installed_versions(["jiwer", "numpy", "torch", "transformers", "datasets", "torchcodec"]),
        dataset_revision=DATASET_REVISION,
    )
    (output_dir / "experiment_config.json").write_text(config.to_json(), encoding="utf-8")
    handle.write_manifest(
        {
            "stage": 2,
            "git_commit": get_git_commit(),
            "selection_fingerprint": selection.fingerprint,
            "timings": timings,
            "generate_kwargs": GENERATE_KWARGS,
            "batch_size": BATCH_SIZE,
        }
    )
    for row in all_summaries:
        if row["stratum"] == "ALL":
            print(f"{row['model']}: WER={row['wer']:.4f} CER={row['cer']:.4f} (n={row['n_valid']})")
    return 0
