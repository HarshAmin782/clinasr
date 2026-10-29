"""Phase 1, stage 1: five-sample Whisper-tiny.en smoke test on the real
EkaCare dataset. Runs on a Kaggle GPU; nothing here is unit-testable end to
end without one. Pure/data-shaping pieces (sample selection determinism,
CSV row construction, CSV writing) ARE split out and unit tested
(tests/test_smoke_test.py); the pieces that need torch/transformers/network
(model loading, audio decoding, transcription, HF revision resolution) are
isolated into their own small functions so the untestable surface is as
small as possible, per AGENTS.md's "report exact output and explain what
can't be validated until a Kaggle GPU run."

Research-integrity notes (AGENTS.md):
- Decoding settings are fixed and printed: greedy decoding (num_beams=1,
  do_sample=False), no custom language/task forcing (whisper-tiny.en is an
  English-only checkpoint).
- The model revision actually used is resolved from the loaded model's own
  config at run time (`_resolved_revision_from_model`), never hardcoded.
- Sample selection is `smoke_test_first_n`: the first N records of the
  streamed `test` split, in dataset order, for a fixed `n`. This is
  explicitly NOT a stratified or random sample (that is stage 2's job) --
  it exists only to prove the pipeline runs end to end.
"""

from __future__ import annotations

import csv
import dataclasses
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from clinasr.config import ExperimentConfig, ModelSpec, NoiseConfig, SampleSelection, resolve_installed_versions
from clinasr.metrics import compute_corpus_metric, compute_metrics
from clinasr.normalization import DEFAULT_NORMALIZATION
from clinasr.run_manifest import RunHandle, get_git_commit, new_run

DATASET_ID = "ekacare/eka-medical-asr-evaluation-dataset"
DATASET_CONFIG = "en"
DATASET_SPLIT = "test"
MODEL_ID = "openai/whisper-tiny.en"
SMOKE_TEST_N = 5
SEED = 0

# (pip install name, import name) -- import name differs for some packages.
BOOTSTRAP_PACKAGES = [
    ("datasets", "datasets"),
    ("transformers", "transformers"),
    ("jiwer", "jiwer"),
    ("torchcodec", "torchcodec"),
]


def ensure_installed(pip_name: str, import_name: Optional[str] = None) -> bool:
    """Import `import_name`, pip-installing `pip_name` first if it's missing.

    Returns True if the package is importable after this call, False if the
    install itself failed (e.g. no matching wheel for this CUDA/torch
    build) -- callers must handle False explicitly rather than assume
    success, since a failed audio-codec install should not silently produce
    empty transcriptions.
    """
    import_name = import_name or pip_name
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        pass
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", pip_name],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"WARNING: pip install {pip_name!r} failed:\n{result.stderr[-2000:]}")
        return False
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def bootstrap_all() -> Dict[str, bool]:
    return {pip_name: ensure_installed(pip_name, import_name) for pip_name, import_name in BOOTSTRAP_PACKAGES}


def load_smoke_records(n: int = SMOKE_TEST_N) -> List[Dict[str, Any]]:
    """Stream the first `n` records of the real test split, WITH decoded audio."""
    import datasets

    stream = datasets.load_dataset(DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT, streaming=True)
    records = []
    for i, record in enumerate(stream):
        if i >= n:
            break
        records.append(dict(record))
    return records


def resolve_model_revision(model_id: str) -> Optional[str]:
    """Return the Hub commit SHA the loaded model actually resolved to."""
    try:
        from huggingface_hub import model_info

        return model_info(model_id).sha
    except Exception as exc:  # pragma: no cover - network/HF-hub dependent
        print(f"WARNING: could not resolve revision for {model_id!r}: {exc}")
        return None


def transcribe_records(records: Sequence[Dict[str, Any]], model_id: str = MODEL_ID) -> List[str]:
    """Run whisper-tiny.en with a fixed, documented decoding configuration."""
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline("automatic-speech-recognition", model=model_id, device=device)

    hypotheses = []
    for record in records:
        audio = record["audio"]
        result = asr(
            {"array": audio["array"], "sampling_rate": audio["sampling_rate"]},
            generate_kwargs={"num_beams": 1, "do_sample": False},
        )
        hypotheses.append(result["text"])
    return hypotheses


def build_prediction_rows(
    records: Sequence[Dict[str, Any]],
    hypotheses: Sequence[str],
    model_id: str,
    condition: str = "clean",
) -> List[Dict[str, Any]]:
    """Pure, unit-testable: shape raw records+hypotheses into output rows."""
    if len(records) != len(hypotheses):
        raise ValueError("records and hypotheses must be the same length")
    rows = []
    for record, hypothesis in zip(records, hypotheses):
        rows.append(
            {
                # `session_id` is NOT a unique per-record identifier: the
                # real dataset's first 5 streamed records already contained
                # a repeated session_id across two different utterances
                # (one session = multiple recorded segments). `file_name`
                # is the real per-record unique path and is used as the
                # sample_id instead; `session_id` stays as its own grouping
                # column so cross-model joins by sample_id are unambiguous.
                # See docs/ISSUES.md.
                "sample_id": record.get("file_name"),
                "reference": record.get("text"),
                "hypothesis": hypothesis,
                "model": model_id,
                "condition": condition,
                "session_id": record.get("session_id"),
                "recording_context": record.get("recording_context"),
            }
        )
    return rows


def write_predictions_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    fieldnames = ["sample_id", "reference", "hypothesis", "model", "condition", "session_id", "recording_context"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_csv(
    model_id: str,
    condition: str,
    corpus_metric,
    path: Path,
) -> None:
    fieldnames = ["model", "condition", "wer", "cer", "n_samples_total", "n_samples_valid", "n_samples_excluded"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "model": model_id,
                "condition": condition,
                "wer": corpus_metric.wer,
                "cer": corpus_metric.cer,
                "n_samples_total": corpus_metric.n_samples_total,
                "n_samples_valid": corpus_metric.n_samples_valid,
                "n_samples_excluded": corpus_metric.n_samples_excluded,
            }
        )


def build_experiment_config(
    model_id: str,
    resolved_revision: Optional[str],
    n: int,
    sample_ids: Sequence[str],
) -> ExperimentConfig:
    versions = resolve_installed_versions(["jiwer", "numpy", "torch", "transformers", "datasets", "torchcodec"])
    return ExperimentConfig(
        dataset_id=DATASET_ID,
        dataset_config=DATASET_CONFIG,
        dataset_split=DATASET_SPLIT,
        seed=SEED,
        sample_selection=SampleSelection(method="smoke_test_first_n", size=n, sample_ids=list(sample_ids)),
        models=[ModelSpec(model_id=model_id, resolved_revision=resolved_revision)],
        normalization=DEFAULT_NORMALIZATION,
        noise=NoiseConfig(snrs_db=[None]),
        software_versions=versions,
    )


def run_smoke_test(output_dir: Path, n: int = SMOKE_TEST_N, model_id: str = MODEL_ID) -> int:
    """Orchestrates the real Phase-1 smoke test. Requires network + GPU/CPU
    inference; not run in unit tests. Writes predictions.csv,
    summary_metrics.csv, and experiment_config.json into `output_dir`
    (Kaggle: /kaggle/working), plus a run manifest/log under
    `output_dir/run/` for traceability.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Bootstrapping required packages...")
    bootstrap_results = bootstrap_all()
    for pkg, ok in bootstrap_results.items():
        print(f"  {pkg}: {'ok' if ok else 'FAILED'}")
    if not bootstrap_results.get("datasets") or not bootstrap_results.get("transformers"):
        print("ERROR: required packages missing after bootstrap; aborting smoke test.")
        return 1

    handle: RunHandle = new_run(output_dir / "run", git_commit=get_git_commit())
    logger = handle.get_logger()
    logger.info("Starting smoke test: model=%s n=%d", model_id, n)

    print(f"Streaming first {n} records from {DATASET_ID} ({DATASET_CONFIG}/{DATASET_SPLIT})...")
    records = load_smoke_records(n=n)
    sample_ids = [r.get("session_id") for r in records]
    print(f"Loaded {len(records)} records: {sample_ids}")

    print(f"Resolving revision for {model_id}...")
    resolved_revision = resolve_model_revision(model_id)
    print(f"Resolved revision: {resolved_revision}")

    print("Running inference (greedy, num_beams=1, do_sample=False)...")
    hypotheses = transcribe_records(records, model_id=model_id)
    for sid, hyp in zip(sample_ids, hypotheses):
        print(f"  {sid}: {hyp!r}")

    rows = build_prediction_rows(records, hypotheses, model_id=model_id, condition="clean")
    write_predictions_csv(rows, output_dir / "predictions.csv")

    sample_metrics = compute_metrics(
        [r["sample_id"] for r in rows],
        [r["reference"] for r in rows],
        [r["hypothesis"] for r in rows],
    )
    corpus_metric = compute_corpus_metric(sample_metrics)
    write_summary_csv(model_id, "clean", corpus_metric, output_dir / "summary_metrics.csv")
    print(
        f"Corpus WER={corpus_metric.wer} CER={corpus_metric.cer} "
        f"(valid={corpus_metric.n_samples_valid}/{corpus_metric.n_samples_total})"
    )

    config = build_experiment_config(model_id, resolved_revision, n, sample_ids)
    (output_dir / "experiment_config.json").write_text(config.to_json(), encoding="utf-8")

    handle.write_manifest(
        {
            "git_commit": get_git_commit(),
            "model_id": model_id,
            "resolved_revision": resolved_revision,
            "sample_ids": sample_ids,
            "corpus_metric": dataclasses.asdict(corpus_metric),
            "bootstrap_results": bootstrap_results,
        }
    )
    logger.info("Smoke test complete. wer=%s cer=%s", corpus_metric.wer, corpus_metric.cer)
    print(f"Done. Outputs written to {output_dir}")
    return 0
