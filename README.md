# ClinASR — Medical speech recognition evaluated beyond WER

ClinASR benchmarks pretrained Whisper models on Indian-English medical speech and asks a narrower
question than word error rate can answer: **were the drug names, doses, negations and medical terms
transcribed correctly?** It combines standard WER/CER with a lexical medical-entity check, a dose-only
check, controlled noise experiments, and a reviewed layer of candidate critical discrepancies — all with
pinned data/model versions and explicit annotation provenance.

> Exploratory research benchmark. Inference with pretrained models only (no training). Not a clinical
> system; nothing here is clinically validated.

## Key findings (300-utterance stratified subset, not population estimates)

| | whisper-tiny.en | whisper-base.en |
|---|---|---|
| Clean WER / CER (normalization v2) | 0.258 / 0.122 | 0.197 / 0.090 |
| WER at 20 / 10 / 0 dB SNR (white noise) | 0.383 / 0.734 / 1.954 | 0.250 / 0.398 / 1.364 |
| Drug-name-only utterances, WER | 1.028 | 0.729 |
| Drug entities: unvalidated strict lexical match rate | 0.219 | 0.276 |
| 46 dose-bearing drug entities: dose expression preserved / complete entity matched | 20 / 0 | 21 / 1 |

1. **base.en beats tiny.en** on clean audio: WER −0.060, 95% paired bootstrap CI [−0.075, −0.045], and in
   all 9 strata. The ranking holds under three normalization versions.
2. **Isolated medical terms are the hard case.** Drug-name-only clips have WER 1.028 (tiny.en; rare names
   become longer English phrases: "isoflurane" → "Eyes up your lane.") and 0.729 (base.en), versus
   0.149 / 0.113 for drug sentences — a descriptive difference between recording strata.
3. **base.en is more noise-robust:** +0.200 WER at 10 dB vs +0.476 for tiny.en. At 0 dB both hallucinate
   repetition loops, and the 0 dB comparison depends on normalization (its CI includes zero under v3).
4. **Whole-entity lexical matching cannot isolate dose preservation:** the complete dose-bearing entity
   matched 0 / 1 times, while the dose expression checked separately was preserved in 20 / 21 of 46.
   Brand-name accuracy was not measured separately.
5. **WER can't show which discrepancies matter.** On a selected 104-row review sample, AI-assisted
   text-only span assessment judged flagged drug-like-name spans meaningful in 20/21 cases and dose-unit
   flags in 7/21 (mostly "mg" ↔ "milligram") — descriptive for the reviewed cases, not detector precision.

Full write-up: [FINAL_RESEARCH_REPORT.md](FINAL_RESEARCH_REPORT.md) · numbers: [FINAL_RESULTS_SUMMARY.md](FINAL_RESULTS_SUMMARY.md) ·
methods/limits: [METHODOLOGY_AND_LIMITATIONS.md](METHODOLOGY_AND_LIMITATIONS.md)

## Why medical ASR needs more than WER

WER weights "the" → "a" the same as "500 mg" → "5 g" or "without surgery" → "with surgery" (both
observed here). A single number cannot say whether medication names, doses and negations survived. ClinASR
keeps WER/CER and adds two complementary layers:

- **Layer 1 — annotated-entity preservation (strict lexical):** does each annotated medical entity appear
  in the hypothesis? Reported as unvalidated strict lexical match rates — the matcher misses equivalent
  expressions, abbreviations, inflections and spoken forms. Plus a separate check that each dose
  expression ("75 mg") is reproduced exactly.
- **Layer 2 — candidate critical discrepancies:** word-level alignment spans touching numbers, dose units,
  negation, medication context, drug-like names or inserted medical terms are flagged for review. Flags are
  review candidates, never automatic error claims.

## Architecture

```
EkaCare test split (HF Hub, pinned revision)
   │  schema audit → coverage scan (no audio) → deterministic stratified selection (fingerprint)
   ▼
Kaggle T4 job  (kaggle/run_clinasr.py, generated from src/clinasr/*)
   │  fingerprint check → audio decode (16 kHz) → whisper-tiny.en / base.en (greedy, fp16)
   │  [Stage 4: seeded white noise at 20/10/0 dB]
   ▼
predictions.csv  ──►  local scoring (no GPU)
                       ├─ WER/CER, normalization v1/v2/v3, paired bootstrap CIs
                       ├─ Layer 1: entity + dose preservation
                       ├─ Layer 2: flagged spans → review sheets → provenance-checked imports
                       └─ collect_final_results.py → docs/final_results.json → figures + report
```

## Data and models

- **Dataset:** [`ekacare/eka-medical-asr-evaluation-dataset`](https://huggingface.co/datasets/ekacare/eka-medical-asr-evaluation-dataset),
  `en`/`test`, revision `433e6603…` — 3,619 utterances, 8.4 h, 10,661 annotated entities in 5 categories.
  Evaluation-only: never used for training or prompting.
- **Subset:** 300 utterances (47.7 min), stratified by recording context × concept type (9 strata, ≥20 each),
  selected by a hash ranking before any model was run.
- **Models:** [`openai/whisper-tiny.en`](https://huggingface.co/openai/whisper-tiny.en) @ `87c71024…`,
  [`openai/whisper-base.en`](https://huggingface.co/openai/whisper-base.en) @ `911407f4…`.

## Results

| Clean audio | Noise robustness |
|---|---|
| ![](docs/figures/fig1_clean_wer_cer.png) | ![](docs/figures/fig4_noise_wer_cer.png) |

| Strict lexical entity match by category (unvalidated) | Dose expression vs complete entity |
|---|---|
| ![](docs/figures/fig3_entity_by_category.png) | ![](docs/figures/fig3b_dose_vs_whole_entity.png) |

## Medical entity and candidate-error analysis

| Layer 2 candidate flags (clean audio; counts, not error rates) | Reviewed flags: span vs context (AI-assisted) |
|---|---|
| ![](docs/figures/fig6_layer2_candidate_categories.png) | ![](docs/figures/fig7_span_vs_context.png) |

Four kinds of evidence are kept separate: **automatically flagged candidate discrepancies**,
**AI-assisted text-level assessments**, **human-reviewed overall assessments** (30 rows), and **clinically
validated errors — none**.

- 431 automatically flagged candidate spans on clean audio; 101 are candidate discrepancies outside the
  annotated entity coverage (not verified errors).
- Layer 1 matcher vs AI-assisted (GPT-6) labels on 150 stratified rows: 90.7% agreement, κ 0.777; when the
  matcher said "preserved" the labels agreed in 99 of 100 rows. Not independent human or clinical validation.
- Annotation provenance: 30 Layer 2 overall labels by a human reviewer; all other labels AI-assisted
  (OpenAI GPT-6), text-only, no audio replay, no clinician. See the report, §7.

## Noise robustness

Seeded additive white Gaussian noise at 20 / 10 / 0 dB on the same 300 utterances; both models hear
identical audio. WER increase vs clean — tiny.en +0.126 / +0.476 / +1.696, base.en +0.053 / +0.200 / +1.166.
base.en's advantage is clear at 20 and 10 dB; at 0 dB its confidence interval excludes zero under v2 but not
under v3, so no decisive 0 dB advantage is claimed. The clean condition of this run reproduced the earlier
run exactly.

## Reproduce

### What is in the repository vs. local only

| Committed (public) | Local only (not committed) |
|---|---|
| All code, tests, the generated Kaggle script, kernel metadata, the reviewed 300-utterance selection (`kaggle/stage2_selection.json`) | Run outputs under `results/` (predictions, per-run summaries, review sheets and labels) |
| `docs/final_results.json` (every reported number + SHA-256 of each source file) and `docs/figures/` | Audio and transcripts (always streamed from the Hugging Face dataset, never stored in the repo) |

Anyone can rerun the tests, the data audit, the selection (and check its fingerprint), and the GPU runs.
`scripts/collect_final_results.py` and `scripts/check_report_numbers.py` regenerate/verify the reported
numbers only on a machine that holds the original `results/` run folders (names are fixed in the script).
Regenerating `docs/figures/` from the committed `docs/final_results.json` works anywhere.

### 1. Local setup (no GPU)

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest -q
```

Linux / macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

The commands below use `python`; on Linux/macOS use `.venv/bin/python`, on Windows `.venv\Scripts\python`
(or activate the venv first).

### 2. Data audit and sample selection (network, no GPU)

```
python scripts/analyze_entity_coverage.py
```

This writes a new run folder `results/RUN_ID/` (RUN_ID = UTC timestamp + commit, printed by the script)
containing `sample_index.csv`. Pass that file to the selector; it must print fingerprint `8b345d92…`:

```
python scripts/select_stage2_samples.py results/RUN_ID/sample_index.csv
```

### 3. GPU runs on Kaggle

Create a Kaggle API token (kaggle.com → Settings → API) and save it as `~/.kaggle/kaggle.json`
(Windows: `C:\Users\YOU\.kaggle\kaggle.json`) **without a UTF-8 BOM**; it is never committed.
`scripts/kaggle_workflow.py` derives the Bearer token the Kernels API needs from that file.

```
python scripts/kaggle_workflow.py init --username YOUR_KAGGLE_USERNAME
python scripts/build_kaggle_script.py
python scripts/kaggle_workflow.py push --confirm --budget-minutes 30
python scripts/kaggle_workflow.py status
python scripts/kaggle_workflow.py fetch
```

`fetch` downloads into a new folder `results/fetched_TIMESTAMP/` (printed). The kernel runs the stage set
by `DEFAULT_STAGE` in `src/clinasr/kaggle_entrypoint.py` (4 = noise run including clean; 2 = clean
two-model run); rebuild the script after changing it. Kaggle-side package versions are pinned in
`requirements-kaggle.txt` and recorded in each run's `experiment_config.json`.

### 4. Scoring and analysis (local, no GPU)

Replace `results/fetched_TIMESTAMP` with the folder printed by `fetch`:

```
python scripts/rescore.py results/fetched_TIMESTAMP --normalization v2
python scripts/rescore.py results/fetched_TIMESTAMP --normalization v3
python scripts/layer2_candidates.py results/fetched_TIMESTAMP
python scripts/make_figures.py
```

## Repository structure

```
src/clinasr/          shared, tested logic
  normalization.py      v1/v2/v3 text normalization
  metrics.py            WER/CER (jiwer), corpus aggregation
  sampling.py           deterministic stratified selection + fingerprint
  medical_entities.py   EkaCare annotation parser, coverage, dose checks
  entities.py           lexical entity matching
  critical_errors.py    Layer 2 flagging
  entity_review.py, layer2_audit.py   annotation agreement / provenance-checked import
  analysis.py           paired bootstrap
  noise.py              seeded SNR noise
  stage2.py, stage4.py, smoke_test.py, kaggle_entrypoint.py   GPU run logic
scripts/              CLI entry points (Kaggle workflow, scoring, review, figures)
kaggle/               generated single-file Kaggle script, kernel metadata, reviewed selection
tests/                176 unit tests
docs/figures/         figures (PNG + SVG) and their sources
docs/final_results.json   every reported number, with source-file hashes
```

Run outputs (`results/`) are kept locally and not committed.

## Attribution and licences

- Dataset: [`ekacare/eka-medical-asr-evaluation-dataset`](https://huggingface.co/datasets/ekacare/eka-medical-asr-evaluation-dataset)
  by Eka Care, MIT licence (per its dataset card). This repository contains no audio and no full transcripts —
  only file identifiers of the selected subset, short quoted excerpts used as examples, and derived metrics.
- Models: [`openai/whisper-tiny.en`](https://huggingface.co/openai/whisper-tiny.en) and
  [`openai/whisper-base.en`](https://huggingface.co/openai/whisper-base.en), Apache-2.0; weights are downloaded
  from the Hugging Face Hub at run time, not redistributed. Whisper: Radford et al., 2022 (arXiv:2212.04356).

## Limitations and future directions

One stratified subset (not population estimates); two small English models; white noise only; 46
dose-bearing entities; strict lexical matching; text-only review without audio; mostly AI-assisted labels
and no clinician review. Next steps: larger and medical-domain ASR models, realistic noise, a drug lexicon
instead of suffix heuristics, number-word normalization, clinician-adjudicated review with audio, and the
dataset's non-English configurations. Details: [METHODOLOGY_AND_LIMITATIONS.md](METHODOLOGY_AND_LIMITATIONS.md).
