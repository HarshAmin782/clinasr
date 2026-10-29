"""Paired comparison of two models evaluated on the same samples.

Corpus WER is a ratio of sums (edits / reference words), so the bootstrap
resamples SAMPLES (with replacement, same draw for both models) and
recomputes each model's corpus WER from summed counts; the statistic is
WER(model_b) - WER(model_a). Uses numpy's seeded Generator so the interval
is reproducible. A 95% interval that excludes 0 is evidence of a
difference on THIS subset; it says nothing about the population unless
strata are reweighted (see docs/STAGE2_PROPOSAL.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from clinasr.normalization import DEFAULT_NORMALIZATION, NormalizationConfig, normalize_text


def word_edit_counts(reference: Optional[str], hypothesis: Optional[str], config: NormalizationConfig = DEFAULT_NORMALIZATION) -> Tuple[int, int]:
    """(word edits, reference word count) for one sample after normalization."""
    import jiwer

    ref = normalize_text(reference, config)
    hyp = normalize_text(hypothesis or "", config)
    n_ref = len(ref.split())
    if n_ref == 0:
        return 0, 0
    out = jiwer.process_words([ref], [hyp])
    return out.substitutions + out.deletions + out.insertions, n_ref


@dataclass
class PairedComparison:
    model_a: str
    model_b: str
    n_samples: int
    wer_a: float
    wer_b: float
    difference: float  # wer_b - wer_a
    ci_low: float
    ci_high: float
    n_bootstrap: int
    seed: int
    n_b_better: int
    n_a_better: int
    n_tied: int


def paired_bootstrap(
    rows: Sequence[Dict[str, str]],
    model_a: str,
    model_b: str,
    n_bootstrap: int = 2000,
    seed: int = 0,
    config: NormalizationConfig = DEFAULT_NORMALIZATION,
) -> PairedComparison:
    """`rows` are predictions.csv rows (sample_id, model, reference, hypothesis)."""
    by_model: Dict[str, Dict[str, Dict[str, str]]] = {model_a: {}, model_b: {}}
    for row in rows:
        if row["model"] in by_model:
            if row["sample_id"] in by_model[row["model"]]:
                raise ValueError(f"duplicate sample_id {row['sample_id']} for {row['model']}")
            by_model[row["model"]][row["sample_id"]] = row
    ids_a, ids_b = set(by_model[model_a]), set(by_model[model_b])
    if ids_a != ids_b:
        raise ValueError(f"models were not evaluated on the same samples ({len(ids_a ^ ids_b)} differ)")
    ids: List[str] = sorted(ids_a)
    if not ids:
        raise ValueError("no samples to compare")

    counts = np.array(
        [
            [*word_edit_counts(by_model[model_a][s]["reference"], by_model[model_a][s]["hypothesis"], config),
             word_edit_counts(by_model[model_b][s]["reference"], by_model[model_b][s]["hypothesis"], config)[0]]
            for s in ids
        ],
        dtype=float,
    )  # columns: edits_a, n_ref, edits_b
    valid = counts[:, 1] > 0
    counts = counts[valid]
    if not len(counts):
        raise ValueError("no samples with a non-empty reference")

    def wers(c):
        return c[:, 0].sum() / c[:, 1].sum(), c[:, 2].sum() / c[:, 1].sum()

    wer_a, wer_b = wers(counts)
    rng = np.random.default_rng(seed)
    n = len(counts)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = counts[rng.integers(0, n, n)]
        a, b = wers(sample)
        diffs[i] = b - a
    low, high = np.percentile(diffs, [2.5, 97.5])

    per_a = counts[:, 0] / counts[:, 1]
    per_b = counts[:, 2] / counts[:, 1]
    return PairedComparison(
        model_a=model_a,
        model_b=model_b,
        n_samples=n,
        wer_a=float(wer_a),
        wer_b=float(wer_b),
        difference=float(wer_b - wer_a),
        ci_low=float(low),
        ci_high=float(high),
        n_bootstrap=n_bootstrap,
        seed=seed,
        n_b_better=int((per_b < per_a).sum()),
        n_a_better=int((per_a < per_b).sum()),
        n_tied=int((per_a == per_b).sum()),
    )
