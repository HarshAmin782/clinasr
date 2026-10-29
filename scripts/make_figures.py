"""Draw the report figures from docs/final_results.json only (run
scripts/collect_final_results.py first). Writes PNG + SVG to docs/figures/.

Every figure title/caption states what the numbers are (automatic metric,
unvalidated matcher output, heuristic flag counts, AI-assisted or human
labels) so no plot reads as a clinical or population-level rate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "final_results.json"
OUT = ROOT / "docs" / "figures"
T, B = "openai/whisper-tiny.en", "openai/whisper-base.en"
NAMES = {T: "whisper-tiny.en", B: "whisper-base.en"}
COLORS = {T: "#4C72B0", B: "#DD8452"}
CONDS = ["clean", "snr_20db", "snr_10db", "snr_0db"]
COND_LABELS = ["clean", "20 dB", "10 dB", "0 dB"]

plt.rcParams.update({"svg.hashsalt": "clinasr", "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)


def bar_labels(ax, bars, fmt="{:.3f}"):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")


def fig1(F):
    m = F["stage2"]["by_normalization"]["v2"]["metrics"]
    p = F["stage2"]["by_normalization"]["v2"]["paired_base_minus_tiny"]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    w = 0.35
    for i, model in enumerate((T, B)):
        bars = ax.bar([x + (i - 0.5) * w for x in (0, 1)], [m[model]["ALL"]["wer"], m[model]["ALL"]["cer"]], w,
                      label=NAMES[model], color=COLORS[model])
        bar_labels(ax, bars)
    ax.set_xticks([0, 1], ["WER", "CER"])
    ax.set_ylabel("Corpus error rate (lower is better)")
    ax.set_ylim(0, 0.32)
    ax.set_title(f"Clean audio, 300 stratified samples (normalization v2)\n"
                 f"base − tiny WER = {p['difference']:+.3f}, 95% paired bootstrap CI [{p['ci_low']:+.3f}, {p['ci_high']:+.3f}]",
                 fontsize=9)
    ax.legend(frameon=False)
    save(fig, "fig1_clean_wer_cer")


def fig2(F):
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    w = 0.35
    for i, model in enumerate((T, B)):
        vals = [F["entities"][v][model]["ALL"]["rate"] for v in ("v2", "v3")]
        bars = ax.bar([x + (i - 0.5) * w for x in (0, 1)], vals, w, label=NAMES[model], color=COLORS[model])
        bar_labels(ax, bars)
    n = F["entities"]["v2"][T]["ALL"]["n"]
    ax.set_xticks([0, 1], ["normalization v2", "normalization v3"])
    ax.set_ylabel("Share of annotated entities matched")
    ax.set_ylim(0, 0.75)
    ax.set_title(f"Unvalidated strict lexical entity match rate, n = {n} entity instances per model", fontsize=9)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    save(fig, "fig2_entity_match_rate")


def fig3(F):
    cats = ["clinical_findings", "misc_medical", "advices", "diagnostics", "drugs"]
    e = F["entities"]["v2"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    w = 0.38
    for i, model in enumerate((T, B)):
        bars = ax.bar([x + (i - 0.5) * w for x in range(len(cats))], [e[model][c]["rate"] for c in cats], w,
                      label=NAMES[model], color=COLORS[model])
        bar_labels(ax, bars, "{:.2f}")
    ax.set_xticks(range(len(cats)), [f"{c}\n(n={e[T][c]['n']})" for c in cats], fontsize=8)
    ax.set_ylabel("Share matched (automatic, unvalidated)")
    ax.set_ylim(0, 0.85)
    ax.set_title("Lexical entity match rate by annotation category (normalization v2)", fontsize=9)
    ax.legend(frameon=False)
    save(fig, "fig3_entity_by_category")


def fig3b(F):
    e = F["entities"]["v2"]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    w = 0.35
    keys = ["dose_entity_whole_match", "dose_preserved"]
    for i, model in enumerate((T, B)):
        bars = ax.bar([x + (i - 0.5) * w for x in (0, 1)], [e[model][k]["rate"] for k in keys], w,
                      label=NAMES[model], color=COLORS[model])
        for b, k in zip(bars, keys):
            ax.annotate(f"{e[model][k]['matched']}/{e[model][k]['n']}\n({e[model][k]['rate']:.2f})",
                        (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom",
                        fontsize=8, xytext=(0, 2), textcoords="offset points")
    n = e[T]["dose_preserved"]["n"]
    ax.set_xticks([0, 1], ["complete annotated\nentity matched", "dose expression\npreserved exactly"])
    ax.set_ylabel(f"Share of {n} dose-bearing drug entities")
    ax.set_ylim(0, 0.7)
    ax.set_title("Complete dose-bearing drug entity vs dose expression only\n"
                 "(unvalidated lexical checks; a complete-entity failure can come from any component;\n"
                 "brand-name accuracy not measured separately; n = 46)", fontsize=8)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig3b_dose_vs_whole_entity")


def fig4(F):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)
    for ax, metric in zip(axes, ("wer", "cer")):
        for model in (T, B):
            for v, style, marker, fill in (("v2", "-", "o", True), ("v3", "--", "s", False)):
                m = F["stage4"]["by_normalization"][v]["metrics"]
                ax.plot(range(4), [m[c][model]["ALL"][metric] for c in CONDS], style, marker=marker, ms=6,
                        lw=1.8, color=COLORS[model], markerfacecolor=COLORS[model] if fill else "white",
                        label=f"{NAMES[model]}, normalization {v}")
        ax.set_xticks(range(4), COND_LABELS)
        ax.set_xlabel("SNR, additive white noise (categorical)")
        ax.set_ylabel(f"Corpus {metric.upper()}")
        ax.axhline(1.0, color="grey", lw=0.6, ls=":")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Noise robustness on the same 300 samples (values > 1 = more errors than reference words)", fontsize=9)
    save(fig, "fig4_noise_wer_cer")


def fig5(F):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    w = 0.35
    for i, model in enumerate((T, B)):
        d = [F["stage4"]["by_normalization"]["v3"]["metrics"][c][model]["ALL"]["wer"]
             - F["stage4"]["by_normalization"]["v2"]["metrics"][c][model]["ALL"]["wer"] for c in CONDS]
        bars = ax.bar([x + (i - 0.5) * w for x in range(4)], d, w, label=NAMES[model], color=COLORS[model])
        bar_labels(ax, bars, "{:+.3f}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(4), COND_LABELS)
    ax.set_ylabel("WER(v3) − WER(v2)")
    ax.set_title("Effect of normalization v3 (hyphen splitting) on WER\n"
                 "negative = lower WER under v3; positive = higher WER under v3\n"
                 "(the 0 dB increase reflects hyphen-joined repetition loops now counted word by word)", fontsize=9)
    ax.legend(frameon=False)
    save(fig, "fig5_normalization_v2_vs_v3")


def fig6(F):
    cats = ["medication_context", "number", "dose_unit", "drug_like_name", "negation", "inserted_medical_term"]
    data = F["layer2"]["by_model_category"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    w = 0.38
    for i, model in enumerate((T, B)):
        bars = ax.bar([x + (i - 0.5) * w for x in range(len(cats))], [data.get(f"{model} | {c}", 0) for c in cats], w,
                      label=NAMES[model], color=COLORS[model])
        bar_labels(ax, bars, "{:.0f}")
    ax.set_xticks(range(len(cats)), [c.replace("_", "\n") for c in cats], fontsize=8)
    ax.set_ylabel("Number of flagged spans")
    ax.set_title(f"Layer 2 heuristic flags, clean audio ({F['layer2']['n_candidates_clean']} spans; a span can have\n"
                 "several categories). Counts of review candidates, NOT error rates.", fontsize=9)
    ax.legend(frameon=False)
    save(fig, "fig6_layer2_candidate_categories")


def fig7(F):
    L = F["layer2_annotations"]
    by_cat = L["span_level_detector_precision"]["by_category"]
    cats = sorted(by_cat, key=lambda c: -by_cat[c]["rate"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    bars = ax.bar(range(len(cats)), [by_cat[c]["rate"] for c in cats], color="#55A868")
    for b, c in zip(bars, cats):
        ax.annotate(f"{by_cat[c]['significant']}/{by_cat[c]['n_decided']}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(range(len(cats)), [c.replace("_", "\n") for c in cats], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Share of reviewed flags judged\nmeaningful (span level)")
    ax.set_title("Share of reviewed flags judged meaningful by\nAI-assisted span-level assessment (text only)", fontsize=9)
    ax = axes[1]
    labels = ["significant", "not_significant", "unclear"]
    ov = L["span_level_detector_precision"]["overall"]
    span_counts = {"significant": ov["significant"], "not_significant": ov["n_decided"] - ov["significant"], "unclear": ov["unclear"]}
    ctx = L["context_label_counts"]
    bottom = [0, 0]
    colors = {"significant": "#C44E52", "not_significant": "#8C8C8C", "unclear": "#CCB974"}
    for lab in labels:
        vals = [span_counts.get(lab, 0), ctx.get(lab, 0)]
        ax.bar([0, 1], vals, bottom=bottom, color=colors[lab], label=lab)
        bottom = [bottom[0] + vals[0], bottom[1] + vals[1]]
    ax.set_xticks([0, 1], ["flagged span\nonly", "visible\ncontext"])
    ax.set_ylabel("Review rows (n = 104)")
    ax.set_title("Same rows, two questions\n(AI-assisted; not prevalence)", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.text(0.01, -0.06, "104 review cases were selected, not randomly sampled; a flag can belong to several categories; "
             "labels are AI-assisted and text-only.\nProportions describe the reviewed cases only, not population-level "
             "detector precision or error rates. Span vs context assessments answer different questions.",
             fontsize=7.5, ha="left", va="top")
    save(fig, "fig7_span_vs_context")


def fig8(F):
    h = F["layer2_annotations"]["human_vs_prior_ai_same_30_rows"]
    pairs = h["pairs_prior_ai_to_human"]
    order = ["significant->significant", "not_significant->not_significant", "not_significant->significant", "significant->not_significant"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    vals = [pairs.get(k, 0) for k in order]
    bars = ax.barh(range(4), vals, color=["#55A868", "#55A868", "#C44E52", "#C44E52"])
    for b, v in zip(bars, vals):
        ax.annotate(str(v), (b.get_width(), b.get_y() + b.get_height() / 2), va="center", xytext=(3, 0), textcoords="offset points", fontsize=8)
    ax.set_yticks(range(4), ["AI sig → human sig", "AI not → human not", "AI not → human sig", "AI sig → human not"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Rows")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_title(f"Human review vs earlier AI-assisted label, same {h['n']} rows: {h['agree']}/{h['n']} agree\n"
                 "(human may have seen the AI label; not an independent double annotation)", fontsize=9)
    save(fig, "fig8_human_vs_ai_labels")


def main(argv=None) -> int:
    import argparse

    figures = {f.__name__: f for f in (fig1, fig2, fig3, fig3b, fig4, fig5, fig6, fig7, fig8)}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=sorted(figures), help="regenerate only these figures")
    args = parser.parse_args(argv)
    F = json.loads(DATA.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.only or list(figures):
        figures[name](F)
    print(f"wrote {len(list(OUT.glob('*.png')))} PNG + {len(list(OUT.glob('*.svg')))} SVG to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
