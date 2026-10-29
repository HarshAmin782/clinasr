# ClinASR — Methodology and Limitations

## What each measurement is

| Measurement | Kind | Validated? | Where |
|---|---|---|---|
| WER / CER | Automatic, corpus-level | Standard metric; normalization declared (v1/v2/v3) | `clinasr.metrics`, `scripts/rescore.py` |
| Paired difference + 95% CI | Paired bootstrap over utterances (2,000 resamples, seed 0) | Statistical, subset-only | `clinasr.analysis` |
| Layer 1 entity match rate | Unvalidated strict lexical match rate (whole-token) | **No** — compared only with AI-assisted labels | `clinasr.entities`, `clinasr.medical_entities` |
| Dose preserved | Automatic, strict number+unit check | **No** | `medical_entities.dose_preserved` |
| Layer 2 flags | Automatic heuristics on word alignment | Flags are candidates, not errors | `clinasr.critical_errors` |
| Layer 2 span/context labels | AI-assisted (GPT-6), text-only | Not clinical | `clinasr.layer2_audit` |
| Layer 2 overall labels | 30 human, 74 AI-assisted, text-only | Not clinical | same |

## Definitions

- **Layer 1** measures *lexical preservation of each annotated entity in isolation*: does the annotated term
  appear in the hypothesis (exact or after normalization)? It does **not** assess meaning in context,
  cannot see words that were not annotated, and misses equivalent expressions, abbreviations, inflections
  and spoken forms. Rates are unvalidated strict lexical match rates, not lower bounds on true preservation.
- **Whole-entity vs dose check**: a complete dose-bearing entity fails matching if any component differs
  (name, spelling, formulation, word order, dose); only the dose check isolates the dose. Brand-name
  accuracy is not measured.
- **Layer 2** flags non-matching alignment spans touching numbers, dose units, negation, medication context,
  drug-like names, or inserted medical terms, for review. Flags are candidate discrepancies; the detector
  does not establish transcription or clinical correctness.
- **Evidence kinds** (kept separate everywhere): automatically flagged candidates; AI-assisted text-level
  assessments; human-reviewed overall assessments (30); clinically validated errors (none).
- **Span-level label**: is the flagged span itself a meaningful discrepancy? Describes the reviewed flags;
  on a selected sample it is not population-level detector precision.
- **Context-level label**: does any medically relevant information in the visible context differ? Use for
  case studies only.
- **"significant"** in review labels means a meaningful transcription discrepancy relative to the text —
  *not* clinical significance.

## Normalization versions

| | Change | Use |
|---|---|---|
| v1 | NFKC, lowercase, punctuation stripped except between digits, whitespace collapsed; numbers never rewritten | Original Stage 2 numbers |
| v2 | + split number from attached dose unit ("400mg" → "400 mg") | Primary |
| v3 | + hyphen → word break except digit ranges and minus signs; Unicode dashes mapped | Sensitivity |

New versions are applied to saved hypotheses only; stored results for earlier versions are never edited.

## Sampling

- Stage 2/4 subset: 300 utterances, stratified by recording_context × type_concept, ≥20 per stratum,
  proportional remainder; hash-ranked (no RNG); fingerprint-checked before inference. Over-samples small
  strata → per-stratum reporting; overall numbers are not population estimates.
- Layer 1 review: 150 rows, 25 per (model × automatic label). Not representative.
- Layer 2 review: 104 rows, clean audio, up to 10 per (model × first category); 30 of them were prioritized for human review.
  Not representative; 89 unique (utterance, model) pairs.

## Limitations

1. One stratified 300-utterance subset; per-stratum n 21–58.
2. Two small English Whisper models; inference only.
3. White noise only; no realistic clinical acoustics.
4. Only 46 dose-bearing drug entities in the subset; strict dose matching.
5. Text-only review, no audio replay; reference errors cannot be adjudicated.
6. No clinician validation.
7. 30 human-reviewed overall Layer 2 labels; all other labels AI-assisted (GPT-6).
8. Whole-token lexical matching; no synonym/abbreviation/number-word handling; 76 entities lack offsets.
9. Hand-written Layer 2 heuristics: over-flag unit expansions, miss unusual drug names.
10. Severe-noise hallucination makes WER depend on normalization (v2 vs v3 at 0 dB).
11. Annotation files saved through Excel can be corrupted ("5-6" → "05-Jun"); imports validate raw text
    against generated originals and take only label columns.
12. Runtime/latency not benchmarked (only single un-warmed timings of the transcription call exist).
