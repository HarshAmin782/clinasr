"""Phase-6 cross-check: rebuild key reported numbers from docs/final_results.json
(formatted as the documents print them) and assert each appears in the
documents that report it. Exits non-zero on any missing or stale number.

Usage:
    python scripts/check_report_numbers.py
"""

from __future__ import annotations

import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T, B = "openai/whisper-tiny.en", "openai/whisper-base.en"


def f(x, d=3):
    return f"{x:.{d}f}".replace("-", "−")


def main() -> int:
    F = json.loads((ROOT / "docs" / "final_results.json").read_text(encoding="utf-8"))
    docs = {name: (ROOT / name).read_text(encoding="utf-8").replace("**", "") for name in
            ("FINAL_RESEARCH_REPORT.md", "FINAL_RESULTS_SUMMARY.md", "README.md")}
    s2, s4 = F["stage2"]["by_normalization"], F["stage4"]["by_normalization"]
    checks = []  # (doc, expected substring)

    for v in ("v1", "v2", "v3"):
        m, p = s2[v]["metrics"], s2[v]["paired_base_minus_tiny"]
        row = (f"{f(m[T]['ALL']['wer'],4)} | {f(m[B]['ALL']['wer'],4)} | {f(m[T]['ALL']['cer'],4)} | "
               f"{f(m[B]['ALL']['cer'],4)} | {f(p['difference'],4)} [{f(p['ci_low'],4)}, {f(p['ci_high'],4)}]")
        checks += [("FINAL_RESEARCH_REPORT.md", row), ("FINAL_RESULTS_SUMMARY.md", row)]
    p2 = s2["v2"]["paired_base_minus_tiny"]
    checks.append(("FINAL_RESEARCH_REPORT.md", f"base.en lower WER on {p2['n_b_better']}, tiny.en on {p2['n_a_better']}, tied on {p2['n_tied']}"))
    checks.append(("README.md", f"[{f(p2['ci_low'])}, {f(p2['ci_high'])}]"))

    for st, x in s2["v2"]["metrics"][T].items():
        if st == "ALL":
            continue
        y = s2["v2"]["metrics"][B][st]
        checks.append(("FINAL_RESEARCH_REPORT.md", f"| {st.replace(' | ', ' × ')} | {x['n']} | {f(x['wer'])} | {f(y['wer'])} | {f(x['cer'])} | {f(y['cer'])} |"))

    labels = {"clean": "clean", "snr_20db": "20 dB", "snr_10db": "10 dB", "snr_0db": "0 dB"}
    for c, lab in labels.items():
        m, p = s4["v2"]["metrics"][c], s4["v2"]["paired_base_minus_tiny"][c]
        checks.append(("FINAL_RESEARCH_REPORT.md", f"| {lab} | {f(m[T]['ALL']['wer'])} | {f(m[B]['ALL']['wer'])} | {f(m[T]['ALL']['cer'])} | "
                                                   f"{f(m[B]['ALL']['cer'])} | {f(p['difference'])} [{f(p['ci_low'])}, {f(p['ci_high'])}] |"))
    p0 = s4["v3"]["paired_base_minus_tiny"]["snr_0db"]
    checks.append(("FINAL_RESEARCH_REPORT.md", f"{f(p0['difference'])} [{f(p0['ci_low'])}, +{p0['ci_high']:.3f}]"))
    for model in (T, B):
        d = s4["v2"]["degradation_vs_clean"][model]
        checks.append(("FINAL_RESEARCH_REPORT.md", " | ".join(f"+{d[c]['wer_increase']:.3f} [{d[c]['ci_low']:.3f}, {d[c]['ci_high']:.3f}]" for c in ("snr_20db", "snr_10db", "snr_0db"))))
        checks.append(("README.md", " / ".join(f"+{d[c]['wer_increase']:.3f}" for c in ("snr_20db", "snr_10db", "snr_0db"))))

    for v in ("v2",):
        e = F["entities"][v]
        for cat in ("drugs", "diagnostics", "clinical_findings", "misc_medical", "advices"):
            checks.append(("FINAL_RESEARCH_REPORT.md", f"{cat} ({e[T][cat]['n']}) | {f(e[T][cat]['rate'])} | {f(e[B][cat]['rate'])}"))
        checks.append(("FINAL_RESEARCH_REPORT.md", f"| All ({e[T]['ALL']['n']:,}) | {f(e[T]['ALL']['rate'])} | {f(e[B]['ALL']['rate'])}"))
        for model in (T, B):
            dp = e[model]["dose_preserved"]
            checks.append(("FINAL_RESEARCH_REPORT.md", f"{dp['matched']} / {dp['n']} ({f(dp['rate'])})"))

    a = F["entity_agreement"]["original_labels"]
    checks += [(d, f"{a['observed_agreement']*100:.1f}%") for d in docs] + [(d, f"{a['cohen_kappa']:.3f}") for d in docs]
    L = F["layer2_annotations"]
    ov = L["span_level_detector_precision"]["overall"]
    checks.append(("FINAL_RESEARCH_REPORT.md", f"{ov['significant']} / {ov['n_decided']} ({ov['rate']:.3f})"))
    for cat, x in L["span_level_detector_precision"]["by_category"].items():
        name = {"drug_like_name": "drug-like name", "inserted_medical_term": "inserted term"}.get(cat, cat.replace("_", " "))
        checks.append(("FINAL_RESULTS_SUMMARY.md", f"{name} {x['significant']}/{x['n_decided']}"))
    cc = L["context_label_counts"]
    checks.append(("FINAL_RESEARCH_REPORT.md", f"{cc['significant']} significant, {cc['not_significant']} not, {cc['unclear']} unclear"))
    checks.append(("FINAL_RESEARCH_REPORT.md", f"{L['pairs_with_any_significant_context_label']} of {L['n_unique_sample_model_pairs']} unique"))
    h = L["human_vs_prior_ai_same_30_rows"]
    checks.append(("FINAL_RESEARCH_REPORT.md", f"agreed on {h['agree']}"))
    checks.append(("FINAL_RESEARCH_REPORT.md", f"{F['layer2']['n_candidates_clean']} flagged spans"))
    checks.append(("FINAL_RESEARCH_REPORT.md", f"{F['layer2']['not_overlapping_annotated_entities']} of them lie outside"))
    checks.append(("FINAL_RESEARCH_REPORT.md", f"{F['dataset']['n_records']:,} utterances"))
    if not F["stage4"]["clean_reproduces_stage2_v2"]:
        checks.append(("FINAL_RESEARCH_REPORT.md", "<clean run did not reproduce stage 2>"))

    missing = [(doc, exp) for doc, exp in checks if exp not in docs[doc]]
    for doc, exp in missing:
        print(f"MISSING in {doc}: {exp}")
    print(f"{len(checks) - len(missing)}/{len(checks)} reported numbers found verbatim")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
