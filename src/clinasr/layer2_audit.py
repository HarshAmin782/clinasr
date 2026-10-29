"""Import and summarize an audited (text-only) Layer 2 annotation file.

The audited annotation round uses text-only labels: "significant" means a
meaningful transcription discrepancy relative to the supplied text, NOT a
clinically significant event. It separates two questions:
- span_label_text_only: is the flagged span itself meaningfully different?
  (use this for detector precision)
- context_label_text_only: does any medically relevant information in the
  shown context differ? (use this for case studies only)
The original candidate file is never modified; the audited file is a
separate annotation version and must match it exactly on raw fields.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from clinasr.critical_errors import REVIEWER_TYPES

TEXT_ONLY_LABELS = ("significant", "not_significant", "unclear")
RAW_FIELDS = (
    "candidate_id", "sample_id", "model", "condition", "op", "ref_span", "hyp_span", "categories",
    "overlaps_annotated_entity", "ref_context", "hyp_context", "normalization", "detector_version", "flagged_by",
)


def validate_audited_against_original(original: Sequence[Dict[str, Any]], audited: Sequence[Dict[str, Any]]) -> None:
    orig = {r["candidate_id"]: r for r in original}
    aud = {r["candidate_id"]: r for r in audited}
    if len(aud) != len(audited):
        raise ValueError("duplicate candidate_id in audited file")
    if set(orig) != set(aud):
        raise ValueError(f"candidate ids differ: {len(set(orig) ^ set(aud))} mismatched")
    for cid, row in aud.items():
        for name in RAW_FIELDS:
            if name in orig[cid] and row.get(name) != orig[cid][name]:
                raise ValueError(f"{cid}: raw field {name!r} was modified")
        for col in ("review_label", "span_label_text_only", "context_label_text_only"):
            if row.get(col) not in TEXT_ONLY_LABELS:
                raise ValueError(f"{cid}: {col}={row.get(col)!r} not in {TEXT_ONLY_LABELS}")
        for col in ("reviewer_type", "span_label_reviewer_type", "context_audit_reviewer_type"):
            if row.get(col) and row[col] not in REVIEWER_TYPES:
                raise ValueError(f"{cid}: {col}={row[col]!r} not in {REVIEWER_TYPES}")


def _count(rows, key) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        k = key(r)
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def summarize_audited(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """A row with several categories counts once in each category."""
    def rate(sig, n):
        return {"significant": sig, "n_decided": n, "rate": sig / n if n else None}

    span_cat: Dict[str, List[int]] = {}
    for r in rows:
        label = r["span_label_text_only"]
        for cat in r["categories"].split(";"):
            cell = span_cat.setdefault(cat, [0, 0, 0])  # significant, decided, unclear
            if label == "unclear":
                cell[2] += 1
            else:
                cell[1] += 1
                cell[0] += label == "significant"
    span_all = [r["span_label_text_only"] for r in rows]
    decided = [label for label in span_all if label != "unclear"]

    pairs: Dict[Tuple[str, str], List[str]] = {}
    for r in rows:
        pairs.setdefault((r["sample_id"], r["model"]), []).append(r["context_label_text_only"])

    return {
        "n_rows": len(rows),
        "n_unique_sample_model_pairs": len(pairs),
        "span_level_detector_precision": {
            "overall": {**rate(decided.count("significant"), len(decided)), "unclear": span_all.count("unclear")},
            "by_category": {c: {**rate(v[0], v[1]), "unclear": v[2]} for c, v in sorted(span_cat.items())},
        },
        "context_label_counts": _count(rows, lambda r: r["context_label_text_only"]),
        "context_labels_by_model": _count(rows, lambda r: f"{r['model']} | {r['context_label_text_only']}"),
        "pairs_with_any_significant_context_label": sum(1 for v in pairs.values() if "significant" in v),
        "span_vs_context_disagreements": sum(r["span_label_text_only"] != r["context_label_text_only"] for r in rows),
        "review_label_reviewer_types": _count(rows, lambda r: r.get("reviewer_type") or "missing"),
        "span_label_reviewer_types": _count(rows, lambda r: r.get("span_label_reviewer_type") or "missing"),
        "needs_human_confirmation": _count(rows, lambda r: r.get("audit_needs_human_confirmation") or "missing"),
        "interpretation": (
            "Text-only discrepancy annotations on a priority/stratified review sample. Span-level rate = share of "
            "flagged spans judged a meaningful discrepancy (heuristic precision on this sample only). Context-level "
            "labels are case-study judgements: not detector precision, not error prevalence, and not clinical significance."
        ),
    }
