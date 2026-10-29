# ClinASR — Final Results Summary

Source of every number: `docs/final_results.json` (regenerate with
`python scripts/collect_final_results.py`; it recomputes metrics from `predictions.csv` and aborts if they
differ from the summaries saved at run time). Primary normalization: **v2**. All results are on one
stratified 300-utterance subset and are **not population estimates**.

## Setup
- Data: `ekacare/eka-medical-asr-evaluation-dataset` en/test @ `433e6603…` — 3,619 utterances (8.4 h);
  subset 300 utterances (47.7 min), 9 strata, ≥20 each, fingerprint `8b345d92…`.
- Models (inference only, greedy): `whisper-tiny.en` @ `87c71024…`, `whisper-base.en` @ `911407f4…`.
- Hardware: Kaggle Tesla T4, fp16, batch 16.

## Clean audio

| Normalization | WER tiny | WER base | CER tiny | CER base | base − tiny WER [95% paired bootstrap CI] |
|---|---|---|---|---|---|
| v1 | 0.2665 | 0.2035 | 0.1228 | 0.0909 | −0.0630 [−0.0778, −0.0480] |
| **v2** | **0.2577** | **0.1974** | **0.1219** | **0.0903** | **−0.0603 [−0.0753, −0.0455]** |
| v3 | 0.2528 | 0.1926 | 0.1204 | 0.0888 | −0.0601 [−0.0744, −0.0463] |

Per utterance (v2): base better 157, tiny better 44, tied 99. base.en lower WER in all 9 strata.
Hardest stratum: narration_entity × drugs (WER 1.028 / 0.729); easiest: narration_sentence ×
clinical_findings (0.137 / 0.108).

## Noise (additive white Gaussian, same 300 utterances, v2)

| | clean | 20 dB | 10 dB | 0 dB |
|---|---|---|---|---|
| WER tiny | 0.258 | 0.383 | 0.734 | 1.954 |
| WER base | 0.197 | 0.250 | 0.398 | 1.364 |
| base − tiny [95% CI] | −0.060 [−0.075, −0.045] | −0.133 [−0.263, −0.064] | −0.336 [−0.596, −0.144] | −0.590 [−1.119, −0.084] |
| WER increase vs clean, tiny | — | +0.126 | +0.476 | +1.696 |
| WER increase vs clean, base | — | +0.053 | +0.200 | +1.166 |

Under v3 the 0 dB comparison interval includes zero (−0.505 [−1.027, +0.014]); clean/20/10 dB conclusions
are unchanged. Clean condition of the noise run reproduced Stage 2 exactly.

## Entities (Layer 1: unvalidated strict lexical match rates, v2)

The matcher misses equivalent expressions, abbreviations, inflections and spoken forms; these are not
measures of true entity preservation.

| | tiny | base |
|---|---|---|
| All 1,014 entity instances | 0.524 | 0.584 |
| drugs (196) | 0.219 | 0.276 |
| diagnostics (122) | 0.336 | 0.385 |
| clinical_findings (532) | 0.658 | 0.722 |
| Dose-bearing drug entities (46): complete annotated entity matched | 0 / 46 | 1 / 46 |
| Dose-bearing drug entities (46): dose preserved exactly | 20 / 46 | 21 / 46 |

Whole-entity lexical matching cannot isolate dose preservation; the separate dose check reveals what it
obscures. Brand-name accuracy was not measured separately.

Matcher vs AI-assisted (GPT-6) labels on 150 stratified rows (not independent human or clinical
validation): 90.7% agreement, κ 0.777;
preserved P/R/F1 0.990/0.884/0.934; not_preserved 0.740/0.974/0.841.

## Layer 2 (candidate critical discrepancies, review-only)

- Clean audio: 431 automatically flagged candidate spans (tiny 221, base 210); 101 are candidate
  discrepancies outside the annotated entity coverage (not verified errors).
- 104-row review sample (stratified/priority, text-only):
  - Span-level, AI-assisted: 60 / 102 decided judged meaningful (drug-like name 20/21, medication context 27/38,
    number 12/20, negation 6/17, dose unit 7/21, inserted term 2/8).
  - Context-level, AI-assisted: 83 significant / 18 not / 3 unclear.
  - Human-reviewed overall labels: 30 rows — 25 significant, 5 not significant; earlier AI pass agreed on
    26 / 30.
- These describe the reviewed, selected cases (categories overlap); they are not population-level detector
  precision, not error rates, and not clinical judgements. No error in this study is clinically validated.
