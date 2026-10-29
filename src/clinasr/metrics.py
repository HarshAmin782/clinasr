"""WER/CER computation shared by local tests and the bundled Kaggle script.

Metric denominator policy (PROJECT_SPEC.md "Measurements"):
- Sample-level WER/CER: jiwer's standard ratio (edits / reference length) for
  that single sample. If the reference is empty, the sample is excluded from
  sample-level aggregation (undefined denominator) and flagged in the
  returned record via `valid=False`, rather than raising or silently
  reporting 0.0/1.0.
- Corpus-level WER/CER: computed from SUMMED edits over SUMMED reference
  length across all valid samples (not the mean of per-sample ratios), which
  is the standard, less length-biased corpus aggregation.
- Missing/invalid hypotheses (None or empty string where a reference is
  non-empty) are treated as a fully-wrong prediction: scored against the
  reference as an empty hypothesis (all reference words/chars counted as
  deletions), not dropped, so model failures are not hidden from the corpus
  metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import jiwer

from clinasr.normalization import DEFAULT_NORMALIZATION, NormalizationConfig, normalize_text


@dataclass(frozen=True)
class SampleMetric:
    sample_id: str
    valid: bool
    wer: Optional[float]
    cer: Optional[float]
    reference_norm: str
    hypothesis_norm: str
    reason_invalid: Optional[str] = None


@dataclass(frozen=True)
class CorpusMetric:
    n_samples_total: int
    n_samples_valid: int
    n_samples_excluded: int
    wer: Optional[float]
    cer: Optional[float]


def _safe_hypothesis(hypothesis: Optional[str]) -> str:
    return hypothesis if hypothesis is not None else ""


def compute_sample_metric(
    sample_id: str,
    reference: Optional[str],
    hypothesis: Optional[str],
    config: NormalizationConfig = DEFAULT_NORMALIZATION,
) -> SampleMetric:
    ref_norm = normalize_text(reference, config)
    hyp_norm = normalize_text(_safe_hypothesis(hypothesis), config)

    if not ref_norm:
        return SampleMetric(
            sample_id=sample_id,
            valid=False,
            wer=None,
            cer=None,
            reference_norm=ref_norm,
            hypothesis_norm=hyp_norm,
            reason_invalid="empty_reference",
        )

    word_out = jiwer.process_words([ref_norm], [hyp_norm])
    char_out = jiwer.process_characters([ref_norm], [hyp_norm])
    return SampleMetric(
        sample_id=sample_id,
        valid=True,
        wer=word_out.wer,
        cer=char_out.cer,
        reference_norm=ref_norm,
        hypothesis_norm=hyp_norm,
    )


def compute_corpus_metric(samples: Sequence[SampleMetric]) -> CorpusMetric:
    valid = [s for s in samples if s.valid]
    excluded = len(samples) - len(valid)
    if not valid:
        return CorpusMetric(
            n_samples_total=len(samples),
            n_samples_valid=0,
            n_samples_excluded=excluded,
            wer=None,
            cer=None,
        )

    refs = [s.reference_norm for s in valid]
    hyps = [s.hypothesis_norm for s in valid]
    word_out = jiwer.process_words(refs, hyps)
    char_out = jiwer.process_characters(refs, hyps)
    return CorpusMetric(
        n_samples_total=len(samples),
        n_samples_valid=len(valid),
        n_samples_excluded=excluded,
        wer=word_out.wer,
        cer=char_out.cer,
    )


def compute_metrics(
    sample_ids: Sequence[str],
    references: Sequence[Optional[str]],
    hypotheses: Sequence[Optional[str]],
    config: NormalizationConfig = DEFAULT_NORMALIZATION,
) -> List[SampleMetric]:
    if not (len(sample_ids) == len(references) == len(hypotheses)):
        raise ValueError("sample_ids, references, and hypotheses must be the same length")
    return [
        compute_sample_metric(sid, ref, hyp, config)
        for sid, ref, hyp in zip(sample_ids, references, hypotheses)
    ]
