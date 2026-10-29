"""Stage 3: tools for validating the automatic entity-match labels against
human judgement.

Workflow: `rematch()` recomputes exact/normalized/omitted labels for a run
under a declared normalization; `select_review_rows()` draws a seeded,
stratified sample (by model x automatic label) for a person to label in a
`human_label` column; `agreement()` compares the two. The automatic labels
may only be described as validated once agreement has been measured on
human labels -- a model or script labelling its own output does not count.

Human label definitions (written into the review sheet):
- preserved: the hypothesis conveys this entity with no clinically
  meaningful difference (spelling/spacing/number-format differences OK).
- not_preserved: the entity is missing or changed in a way that alters
  meaning (wrong drug, wrong number, wrong unit, garbled).
- unclear: cannot judge from the text alone.
The automatic labels map to this binary as exact/normalized -> preserved,
omitted -> not_preserved.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Sequence

from clinasr.entities import EntitySpan, match_entity
from clinasr.medical_entities import dose_preserved, is_dosage_like
from clinasr.normalization import DEFAULT_NORMALIZATION, NormalizationConfig

HUMAN_LABELS = ("preserved", "not_preserved", "unclear")
AUTO_TO_BINARY = {"exact": "preserved", "normalized": "preserved", "omitted": "not_preserved"}


def rematch(
    entity_rows: Sequence[Dict[str, Any]],
    hypotheses: Dict[tuple, str],
    config: NormalizationConfig = DEFAULT_NORMALIZATION,
) -> List[Dict[str, Any]]:
    """Recompute match labels. `hypotheses` maps (sample_id, model) -> text."""
    out = []
    for row in entity_rows:
        hyp = hypotheses[(row["sample_id"], row["model"])]
        entity = EntitySpan(text=row["entity_text"], entity_type=row["entity_type"])
        result = match_entity(entity, hyp, config)
        new = dict(row)
        new["match_status"] = result.status
        new["normalization"] = config.version
        new["dosage_like"] = is_dosage_like(row["entity_text"])
        new["dose_preserved"] = dose_preserved(row["entity_text"], hyp, config)
        out.append(new)
    return out


def _rank(seed: int, row: Dict[str, Any]) -> str:
    key = f"{seed}:{row['sample_id']}:{row['model']}:{row['entity_id']}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def select_review_rows(rows: Sequence[Dict[str, Any]], per_cell: int, seed: int) -> List[Dict[str, Any]]:
    """Up to `per_cell` rows from each (model, match_status) cell, hash-ranked."""
    cells: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        cells.setdefault((row["model"], row["match_status"]), []).append(row)
    picked = []
    for key in sorted(cells):
        ranked = sorted(cells[key], key=lambda r: _rank(seed, r))
        picked.extend(ranked[:per_cell])
    return picked


def agreement(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare auto labels with filled-in `human_label`s (rows labelled
    'unclear' or left blank are counted but excluded from agreement)."""
    labelled = [r for r in rows if r.get("human_label") in ("preserved", "not_preserved")]
    unclear = sum(1 for r in rows if r.get("human_label") == "unclear")
    blank = sum(1 for r in rows if not r.get("human_label"))
    invalid = sorted({r["human_label"] for r in rows if r.get("human_label") and r["human_label"] not in HUMAN_LABELS})
    if invalid:
        raise ValueError(f"unknown human labels: {invalid}")

    confusion = {(a, h): 0 for a in ("preserved", "not_preserved") for h in ("preserved", "not_preserved")}
    by_status: Dict[str, Dict[str, int]] = {}
    for r in labelled:
        auto = AUTO_TO_BINARY[r["match_status"]]
        confusion[(auto, r["human_label"])] += 1
        cell = by_status.setdefault(r["match_status"], {"n": 0, "agree": 0})
        cell["n"] += 1
        cell["agree"] += int(auto == r["human_label"])

    n = len(labelled)
    if n == 0:
        return {"n_labelled": 0, "n_unclear": unclear, "n_blank": blank}
    agree = confusion[("preserved", "preserved")] + confusion[("not_preserved", "not_preserved")]
    p_o = agree / n
    p_auto = (confusion[("preserved", "preserved")] + confusion[("preserved", "not_preserved")]) / n
    p_human = (confusion[("preserved", "preserved")] + confusion[("not_preserved", "preserved")]) / n
    p_e = p_auto * p_human + (1 - p_auto) * (1 - p_human)
    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else None
    return {
        "n_labelled": n,
        "n_unclear": unclear,
        "n_blank": blank,
        "observed_agreement": p_o,
        "cohen_kappa": kappa,
        "confusion_auto_vs_human": {f"{a}->{h}": c for (a, h), c in confusion.items()},
        "agreement_by_auto_status": {k: v["agree"] / v["n"] for k, v in by_status.items()},
        "note": "Agreement on a stratified sample; not a population estimate of label accuracy.",
    }
