# Figures

All figures are drawn by `scripts/make_figures.py` from `docs/final_results.json`, which
`scripts/collect_final_results.py` builds from the saved run outputs (source files and SHA-256 listed in
the JSON under `source_files_sha256`). PNG and SVG versions of each figure are provided.

| Figure | Shows | Underlying run files |
|---|---|---|
| `fig1_clean_wer_cer` | Clean WER/CER, both models, v2; paired CI in title | Stage 2 `predictions.csv` (cross-checked with `summary_metrics_v2.csv`) |
| `fig2_entity_match_rate` | Overall unvalidated strict lexical entity match rate, v2 and v3 | Stage 2 `entity_matches_v2.csv`, `entity_matches_v3.csv` |
| `fig3_entity_by_category` | Strict lexical match rate by annotation category, v2 (not true clinical preservation) | Stage 2 `entity_matches_v2.csv` |
| `fig3b_dose_vs_whole_entity` | Complete annotated dose-bearing entity matched vs dose expression preserved, with counts (46 entities); not a brand-name accuracy measure | Stage 2 `entity_matches_v2.csv` |
| `fig4_noise_wer_cer` | WER/CER vs SNR; colour = model, solid/filled circle = v2, dashed/open square = v3 | Stage 4 `predictions.csv` (cross-checked with `summary_metrics_v2/v3.csv`) |
| `fig5_normalization_v2_vs_v3` | WER(v3) − WER(v2) per condition (negative = lower WER under v3) | Stage 4 `predictions.csv` |
| `fig6_layer2_candidate_categories` | Counts of automatically flagged candidate spans by category, clean audio (not error rates; a span can be in several categories) | Stage 2 `layer2_candidates_v3.csv` |
| `fig7_span_vs_context` | Share of reviewed flags judged meaningful by AI-assisted span-level assessment (selected cases; descriptive, not population detector precision); span vs context label distribution | Stage 2 `clinasr_layer2_104_consistency_audited.csv` |
| `fig8_human_vs_ai_labels` | Human overall labels vs earlier AI-assisted labels on the same 30 rows | `clinasr_human_review_final_30.csv`, `layer2_review_sheet_v3_labeled.csv` |

Stage 2 = `results/fetched_20261029T163039Z/`; Stage 4 = `results/fetched_20261029T164654Z/`
(local, gitignored).

Regenerate a subset with `python scripts/make_figures.py --only fig2 fig4` (all figures if `--only` is
omitted). SVGs are written without a creation date so unchanged figures are byte-identical on rerun.
Revision 2026-10-30: fig2, fig3b, fig4, fig5 and fig7 were redrawn for presentation/terminology only; their
underlying numbers did not change. fig1, fig3, fig6 and fig8 are unchanged in content.
