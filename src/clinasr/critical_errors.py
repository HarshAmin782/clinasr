"""Layer 2: candidate critical transcription errors for manual review.

Layer 1 (clinasr.entities / medical_entities) measures LEXICAL preservation
of each annotated entity in isolation: is the annotated term present in the
hypothesis under a declared normalization? It does not judge meaning in
context, and it cannot see errors in words nobody annotated.

Layer 2 looks at every non-matching span of a word-level alignment between
reference and hypothesis (jiwer), and FLAGS spans that touch heuristically
risky content: numbers, dose units, negation, medication context, drug-like
names, or medical terms that the model inserted. A flag means "a person
should look at this", never "this is a clinically significant error";
significance is only ever set by a reviewer, and every review records
who/what the reviewer was.

Heuristics are deliberately simple, deterministic and listed below so they
can be audited. They will produce false positives (and miss errors).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from clinasr.normalization import NORMALIZATION_V3, NormalizationConfig, normalize_text

DETECTOR_VERSION = "layer2-heuristics-v1"
FLAG_PROVENANCE = f"{DETECTOR_VERSION} (automatic heuristic flag; not a clinical judgement)"

REVIEW_LABELS = ("clinically_significant", "not_significant", "unclear")
REVIEWER_TYPES = ("ai_assisted", "human", "clinician")

NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "half", "once", "twice", "thrice", "single", "double",
}
UNITS = {
    "mg", "mcg", "µg", "g", "gm", "gram", "grams", "kg", "ml", "l", "litre", "liter", "iu", "unit",
    "units", "milligram", "milligrams", "microgram", "micrograms", "millilitre", "milliliter",
    "millilitres", "milliliters", "percent", "dl", "deciliter", "decilitre", "mmol", "mmhg",
}
NEGATIONS = {"no", "not", "without", "never", "denies", "denied", "nil", "none", "negative", "absent", "non", "nor", "cannot"}
MED_FORM_WORDS = {
    "tablet", "tablets", "tab", "tabs", "capsule", "capsules", "cap", "syrup", "injection", "inj",
    "drops", "drop", "ointment", "cream", "suspension", "gel", "inhaler", "sachet", "lotion", "spray",
}
DRUG_SUFFIX = re.compile(
    r"^[a-z]{3,}(mycin|cillin|cycline|pril|sartan|olol|statin|azole|prazole|dipine|floxacin|"
    r"tidine|vir|mab|tinib|oxetine|zepam|azepam|parin|formin|gliptin|glitazone|semide|thiazide|"
    r"profen|fenac|caine|lukast|triptan|dronate|olone|sone)$"
)
# Inserted-term check: small, explicit list; extend deliberately, not ad hoc.
MEDICAL_TERMS = {
    "covid", "cancer", "tumor", "tumour", "diabetes", "diabetic", "fever", "pain", "infection",
    "stroke", "hiv", "aids", "tb", "tuberculosis", "pneumonia", "hepatitis", "malaria", "dengue",
    "asthma", "fracture", "bleeding", "seizure", "seizures", "allergy", "allergic", "insulin",
    "antibiotic", "antibiotics", "steroid", "steroids", "surgery", "biopsy", "chemotherapy",
    "hypertension", "infarction", "sepsis", "overdose", "positive", "negative", "abnormal",
}
CONTEXT_TOKENS = 6
MED_CONTEXT_WINDOW = 2


@dataclass
class Candidate:
    candidate_id: str
    sample_id: str
    model: str
    condition: str
    op: str  # substitute | delete | insert
    ref_span: str
    hyp_span: str
    categories: List[str]
    overlaps_annotated_entity: bool
    ref_context: str
    hyp_context: str
    normalization: str
    detector_version: str = DETECTOR_VERSION
    flagged_by: str = FLAG_PROVENANCE


def _is_number(tok: str) -> bool:
    return any(c.isdigit() for c in tok) or tok in NUMBER_WORDS


def _near(tokens: Sequence[str], start: int, end: int, vocab: Set[str], window: int) -> bool:
    lo, hi = max(0, start - window), min(len(tokens), end + window)
    return any(t in vocab for t in tokens[lo:hi])


def categorize(
    op: str,
    ref_tokens: Sequence[str],
    hyp_tokens: Sequence[str],
    ref_span: Tuple[int, int],
    hyp_span: Tuple[int, int],
) -> List[str]:
    rs, re_ = ref_span
    hs, he = hyp_span
    ref_part, hyp_part = list(ref_tokens[rs:re_]), list(hyp_tokens[hs:he])
    both = ref_part + hyp_part
    cats: List[str] = []
    if any(_is_number(t) for t in both):
        cats.append("number")
    if any(t in UNITS for t in both):
        cats.append("dose_unit")
    if any(t in NEGATIONS for t in both):
        cats.append("negation")
    if any(DRUG_SUFFIX.match(t) for t in both):
        cats.append("drug_like_name")
    ref_med = rs < re_ and (
        _near(ref_tokens, rs, re_, MED_FORM_WORDS, MED_CONTEXT_WINDOW) or _near(ref_tokens, rs, re_, UNITS, MED_CONTEXT_WINDOW)
    )
    hyp_med = op == "insert" and _near(hyp_tokens, hs, he, MED_FORM_WORDS, MED_CONTEXT_WINDOW)
    if ref_med or hyp_med:
        cats.append("medication_context")
    if op in ("insert", "substitute"):
        ref_set = set(ref_tokens)
        inserted = [t for t in hyp_part if t not in ref_set]
        if any(t in MEDICAL_TERMS or DRUG_SUFFIX.match(t) for t in inserted):
            cats.append("inserted_medical_term")
    return cats


def _context(tokens: Sequence[str], start: int, end: int) -> str:
    lo, hi = max(0, start - CONTEXT_TOKENS), min(len(tokens), end + CONTEXT_TOKENS)
    return " ".join(list(tokens[lo:start]) + ["[["] + list(tokens[start:end]) + ["]]"] + list(tokens[end:hi]))


def _entity_token_set(entity_texts: Iterable[str], config: NormalizationConfig) -> Set[str]:
    stop = {"of", "the", "a", "an", "and", "in", "to", "for", "with", "on", "at", "is", "by"}
    tokens: Set[str] = set()
    for text in entity_texts:
        tokens.update(t for t in normalize_text(text, config).split() if t not in stop)
    return tokens


def extract_candidates(
    sample_id: str,
    model: str,
    reference: Optional[str],
    hypothesis: Optional[str],
    condition: str = "clean",
    entity_texts: Iterable[str] = (),
    config: NormalizationConfig = NORMALIZATION_V3,
    max_span_tokens: int = 12,
) -> List[Candidate]:
    """Flag risky non-matching alignment spans for one (sample, model).

    Spans longer than `max_span_tokens` (e.g. a hallucinated repetition
    loop) are truncated in the stored text but still flagged once.
    """
    import jiwer

    ref = normalize_text(reference, config)
    hyp = normalize_text(hypothesis or "", config)
    if not ref:
        return []
    out = jiwer.process_words([ref], [hyp if hyp else ""])
    ref_tokens, hyp_tokens = out.references[0], out.hypotheses[0]
    entity_tokens = _entity_token_set(entity_texts, config)

    candidates: List[Candidate] = []
    for index, chunk in enumerate(out.alignments[0]):
        if chunk.type == "equal":
            continue
        rspan = (chunk.ref_start_idx, chunk.ref_end_idx)
        hspan = (chunk.hyp_start_idx, chunk.hyp_end_idx)
        cats = categorize(chunk.type, ref_tokens, hyp_tokens, rspan, hspan)
        if not cats:
            continue

        def clip(tokens, span):
            part = list(tokens[span[0]:span[1]])
            return " ".join(part[:max_span_tokens]) + (f" …(+{len(part) - max_span_tokens})" if len(part) > max_span_tokens else "")

        key = f"{sample_id}|{model}|{condition}|{config.version}|{index}"
        candidates.append(
            Candidate(
                candidate_id=hashlib.sha256(key.encode("utf-8")).hexdigest()[:16],
                sample_id=sample_id,
                model=model,
                condition=condition,
                op=chunk.type,
                ref_span=clip(ref_tokens, rspan),
                hyp_span=clip(hyp_tokens, hspan),
                categories=cats,
                overlaps_annotated_entity=bool(set(ref_tokens[rspan[0]:rspan[1]]) & entity_tokens),
                ref_context=_context(ref_tokens, *rspan),
                hyp_context=_context(hyp_tokens, *hspan) if len(hyp_tokens[hspan[0]:hspan[1]]) <= max_span_tokens
                else _context(hyp_tokens, hspan[0], hspan[0] + max_span_tokens) + " …",
                normalization=config.version,
            )
        )
    return candidates


def select_for_review(candidates: Sequence[Candidate], per_category: int, seed: int) -> List[Candidate]:
    """Seeded, hash-ranked sample: up to `per_category` candidates per
    (model, first category) cell. Stratified, NOT representative."""
    cells: Dict[Tuple[str, str], List[Candidate]] = {}
    for c in candidates:
        cells.setdefault((c.model, c.categories[0]), []).append(c)
    chosen: List[Candidate] = []
    for key in sorted(cells):
        ranked = sorted(cells[key], key=lambda c: hashlib.sha256(f"{seed}:{c.candidate_id}".encode()).hexdigest())
        chosen.extend(ranked[:per_category])
    return chosen


def validate_review_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Check a filled Layer 2 review sheet: labels and reviewer provenance
    must use the allowed values, and every labelled row must say who/what
    reviewed it. Returns counts per (reviewer_type, review_label)."""
    counts: Dict[str, int] = {}
    for row in rows:
        label = (row.get("review_label") or "").strip()
        reviewer = (row.get("reviewer_type") or "").strip()
        if not label:
            continue
        if label not in REVIEW_LABELS:
            raise ValueError(f"row {row.get('candidate_id')}: unknown review_label {label!r}")
        if reviewer not in REVIEWER_TYPES:
            raise ValueError(f"row {row.get('candidate_id')}: labelled without a valid reviewer_type ({reviewer!r})")
        key = f"{reviewer}:{label}"
        counts[key] = counts.get(key, 0) + 1
    return counts
