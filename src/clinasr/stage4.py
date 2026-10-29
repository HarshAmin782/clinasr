"""Stage 4: noise robustness on the Stage 2 sample set.

Same 300 samples (same fingerprint check as Stage 2), same two pinned
models, same greedy decoding. Conditions: clean, 20, 10, 0 dB SNR additive
white Gaussian noise from clinasr.noise (seeded per sample and condition,
so every model hears the identical noisy waveform). Every per-sample noise
outcome (status, achieved SNR, whether peak renormalization was needed) is
written to noise_log.csv. Scored with the declared default normalization
(v2), whose version is recorded in every summary row.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from clinasr.config import ExperimentConfig, ModelSpec, NoiseConfig, SampleSelection, resolve_installed_versions
from clinasr.noise import add_noise_at_snr
from clinasr.normalization import DEFAULT_NORMALIZATION
from clinasr.run_manifest import get_git_commit, new_run
from clinasr.smoke_test import bootstrap_all
from clinasr import stage2

SNRS_DB: List[Optional[float]] = [None, 20.0, 10.0, 0.0]
NOISE_SEED = 20261029


def condition_name(snr_db: Optional[float]) -> str:
    return "clean" if snr_db is None else f"snr_{int(snr_db)}db"


def noisy_records(records: Sequence[Dict[str, Any]], snr_db: Optional[float], seed: int = NOISE_SEED):
    """Return (records with replaced audio arrays, per-sample noise log rows)."""
    out, log = [], []
    for record in records:
        audio = record["audio"]
        signal = np.asarray(audio["array"], dtype=np.float64)
        result = add_noise_at_snr(signal, snr_db, base_seed=seed, sample_id=record["file_name"])
        new = dict(record)
        new["audio"] = {"array": result.signal.astype(np.float32), "sampling_rate": audio["sampling_rate"]}
        out.append(new)
        log.append(
            {
                "sample_id": record["file_name"],
                "condition": condition_name(snr_db),
                "status": result.status,
                "requested_snr_db": result.requested_snr_db,
                "achieved_snr_db": result.achieved_snr_db,
                "was_clipped": result.was_clipped,
            }
        )
    return out, log


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_stage4(output_dir: Path) -> int:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bootstrap = bootstrap_all()
    print(f"bootstrap: {bootstrap}")
    if not all(bootstrap.values()):
        print("ERROR: package bootstrap failed; aborting before inference.")
        return 1

    handle = new_run(output_dir / "run", git_commit=get_git_commit())
    logger = handle.get_logger()

    selection = stage2.select(stage2.load_metadata())
    (output_dir / "selection.json").write_text(json.dumps(dataclasses.asdict(selection), indent=2), encoding="utf-8")
    if selection.fingerprint != stage2.STAGE2_EXPECTED_FINGERPRINT:
        print(f"ERROR: selection fingerprint {selection.fingerprint} != expected; aborting before inference.")
        return 2
    clean_records = stage2.load_selected_audio(selection.sample_ids)

    predictions, summaries, noise_log, timings = [], [], [], {}
    for snr in SNRS_DB:
        cond = condition_name(snr)
        records, log = noisy_records(clean_records, snr)
        noise_log.extend(log)
        for model_id, revision in stage2.MODELS.items():
            print(f"[{cond}] {model_id} ...")
            result = stage2.transcribe(records, model_id, revision)
            timings[f"{cond}/{model_id}"] = {k: result[k] for k in ("seconds", "device", "dtype")}
            rows = stage2.prediction_rows(records, result["hypotheses"], model_id)
            for row in rows:
                row["condition"] = cond
            predictions.extend(rows)
            for s in stage2.summary_rows(rows, model_id, DEFAULT_NORMALIZATION):
                s["condition"] = cond
                summaries.append(s)
            logger.info("%s %s done in %.1fs", cond, model_id, result["seconds"])

    _write_csv(output_dir / "predictions.csv", predictions)
    _write_csv(output_dir / "summary_metrics.csv", summaries)
    _write_csv(output_dir / "noise_log.csv", noise_log)

    config = ExperimentConfig(
        dataset_id=stage2.DATASET_ID,
        dataset_config=stage2.DATASET_CONFIG,
        dataset_split=stage2.DATASET_SPLIT,
        seed=stage2.STAGE2_SEED,
        sample_selection=SampleSelection(
            method="stratified_hash_ranked",
            size=len(selection.sample_ids),
            stratify_by=list(stage2.STRATA_KEYS),
            sample_ids=list(selection.sample_ids),
        ),
        models=[ModelSpec(model_id=m, resolved_revision=r) for m, r in stage2.MODELS.items()],
        normalization=DEFAULT_NORMALIZATION,
        noise=NoiseConfig(snrs_db=list(SNRS_DB), seed=NOISE_SEED),
        metrics=["wer", "cer"],
        software_versions=resolve_installed_versions(["jiwer", "numpy", "torch", "transformers", "datasets", "torchcodec"]),
        dataset_revision=stage2.DATASET_REVISION,
    )
    (output_dir / "experiment_config.json").write_text(config.to_json(), encoding="utf-8")
    statuses: Dict[str, int] = {}
    for row in noise_log:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    handle.write_manifest(
        {
            "stage": 4,
            "git_commit": get_git_commit(),
            "selection_fingerprint": selection.fingerprint,
            "noise_type": "additive white Gaussian",
            "noise_status_counts": statuses,
            "timings": timings,
        }
    )
    for s in summaries:
        if s["stratum"] == "ALL":
            print(f"{s['condition']:10s} {s['model']}: WER={s['wer']:.4f} CER={s['cer']:.4f}")
    return 0
