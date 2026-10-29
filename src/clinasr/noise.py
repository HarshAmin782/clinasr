"""Seeded additive white-noise augmentation at a target SNR.

Safety policy for edge cases (required by the repository-hardening task;
see prompts/04-robustness.md):
- Signal with (near-)zero power (silence, or shorter than `min_samples`):
  noise cannot be scaled to a meaningful target SNR (division by ~0 power).
  The original signal is returned UNCHANGED with status="skipped_silent" (or
  "skipped_short"), never a divide-by-zero crash and never a fabricated
  noisy signal.
- `snr_db=None` is the explicit "clean" condition: a deliberate no-op,
  returned with status="clean", not an error.
- NaN/inf SNR values, or a `snr_db` not drawn from the configured condition
  set, raise ValueError: those indicate a configuration bug, not a
  recoverable per-sample audio condition, and silently clamping them would
  hide the bug.
- The noisy signal is peak-normalized back into [-1, 1] ONLY if adding noise
  pushed it outside that range (clipping would otherwise distort the
  waveform); otherwise the signal is left exactly as computed. This is
  recorded on the result as `was_clipped`.
- Noise is deterministic for a given (seed, sample_id, snr_db) triple via a
  dedicated `numpy.random.default_rng` seeded from a stable hash, so re-runs
  and different SNR conditions for the same sample use independent but
  reproducible noise draws.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

MIN_SAMPLES = 16  # shorter clips are not considered meaningful for SNR scaling
SILENCE_POWER_EPS = 1e-9


@dataclass(frozen=True)
class NoiseResult:
    signal: np.ndarray
    status: str  # "clean" | "ok" | "skipped_silent" | "skipped_short"
    requested_snr_db: Optional[float]
    achieved_snr_db: Optional[float]
    was_clipped: bool


def _seed_for(base_seed: int, sample_id: str, snr_db: Optional[float]) -> int:
    key = f"{base_seed}|{sample_id}|{snr_db}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _rms_power(signal: np.ndarray) -> float:
    if signal.size == 0:
        return 0.0
    return float(np.mean(np.square(signal, dtype=np.float64)))


def add_noise_at_snr(
    signal: np.ndarray,
    snr_db: Optional[float],
    base_seed: int,
    sample_id: str,
) -> NoiseResult:
    if snr_db is None:
        return NoiseResult(
            signal=np.asarray(signal),
            status="clean",
            requested_snr_db=None,
            achieved_snr_db=None,
            was_clipped=False,
        )

    if not math.isfinite(snr_db):
        raise ValueError(f"snr_db must be finite, got {snr_db!r}")

    signal = np.asarray(signal, dtype=np.float64)

    if signal.shape[0] < MIN_SAMPLES:
        return NoiseResult(
            signal=signal,
            status="skipped_short",
            requested_snr_db=snr_db,
            achieved_snr_db=None,
            was_clipped=False,
        )

    signal_power = _rms_power(signal)
    if signal_power < SILENCE_POWER_EPS:
        return NoiseResult(
            signal=signal,
            status="skipped_silent",
            requested_snr_db=snr_db,
            achieved_snr_db=None,
            was_clipped=False,
        )

    rng = np.random.default_rng(_seed_for(base_seed, sample_id, snr_db))
    noise = rng.standard_normal(signal.shape[0])
    noise_power = _rms_power(noise)

    target_noise_power = signal_power / (10 ** (snr_db / 10))
    scale = math.sqrt(target_noise_power / noise_power) if noise_power > 0 else 0.0
    scaled_noise = noise * scale

    noisy = signal + scaled_noise
    achieved_noise_power = _rms_power(scaled_noise)
    achieved_snr_db = (
        10 * math.log10(signal_power / achieved_noise_power)
        if achieved_noise_power > 0
        else float("inf")
    )

    peak = float(np.max(np.abs(noisy))) if noisy.size else 0.0
    was_clipped = peak > 1.0
    if was_clipped:
        noisy = noisy / peak

    return NoiseResult(
        signal=noisy,
        status="ok",
        requested_snr_db=snr_db,
        achieved_snr_db=achieved_snr_db,
        was_clipped=was_clipped,
    )
