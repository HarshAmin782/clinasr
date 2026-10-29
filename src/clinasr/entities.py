"""Medical-entity preservation matching.

This module is schema-agnostic on purpose: PROJECT_SPEC.md and
prompts/03-medical-entities.md both require inspecting the real
`medical_entities` annotation structure on the EkaCare dataset BEFORE
writing dataset-specific metrics. That inspection (schema_audit.py, run
against the live dataset) has not happened yet in this repository, so this
module only provides the generic matching engine. Phase 3 adapts the real
annotation records into `EntitySpan` objects and wires this engine in.

Matching policy (avoids the "optimistic substring match" failure mode
explicitly called out in prompts/03-medical-entities.md, e.g. the digits
"50" wrongly matching inside the hypothesis text "150mg"):
- Matching is done against WORD-BOUNDARY tokens of the normalized hypothesis,
  never raw substring search.
- "exact": the normalized entity text appears as a contiguous run of
  normalized hypothesis tokens, matched at word boundaries.
- "normalized": only matches after normalization (case/whitespace/
  punctuation policy from normalization.py) where the raw strings differed.
  Every "exact" match is also a "normalized" match, matched before "exact"
  is checked so callers see the most specific label.
- "omitted": entity text not found in the hypothesis at all.
Raw hypothesis text is always preserved on the result so downstream review
can distinguish "matched" from "a human should double check this."

Substitution and false-positive detection (an entity-shaped token appearing
in the hypothesis that does not correspond to any reference entity) require
knowing which hypothesis tokens are "entity-shaped" for the real dataset,
which depends on the annotation schema this module deliberately does not
assume. That detection is Phase 3 work once schema_audit.py has run against
the live dataset; see docs/ISSUES.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from clinasr.normalization import DEFAULT_NORMALIZATION, NormalizationConfig, normalize_text


@dataclass(frozen=True)
class EntitySpan:
    text: str
    entity_type: Optional[str] = None
    entity_id: Optional[str] = None


@dataclass(frozen=True)
class EntityMatchResult:
    entity: EntitySpan
    status: str  # "exact" | "normalized" | "omitted" | "false_positive_only"
    hypothesis_raw: str


def _tokenize(text: str) -> List[str]:
    return text.split() if text else []


def _contains_subsequence(haystack_tokens: Sequence[str], needle_tokens: Sequence[str]) -> bool:
    if not needle_tokens:
        return False
    n = len(needle_tokens)
    for i in range(len(haystack_tokens) - n + 1):
        if list(haystack_tokens[i : i + n]) == list(needle_tokens):
            return True
    return False


def match_entity(
    entity: EntitySpan,
    hypothesis_raw: str,
    config: NormalizationConfig = DEFAULT_NORMALIZATION,
) -> EntityMatchResult:
    """Match a single entity against one hypothesis transcript."""
    hyp_norm = normalize_text(hypothesis_raw, config)
    hyp_tokens = _tokenize(hyp_norm)

    raw_entity_tokens = _tokenize(entity.text)
    norm_entity_tokens = _tokenize(normalize_text(entity.text, config))

    # "exact": the un-normalized entity text appears verbatim at a token
    # boundary in the RAW hypothesis (stronger than normalized match).
    raw_hyp_tokens = _tokenize(hypothesis_raw or "")
    if _contains_subsequence(raw_hyp_tokens, raw_entity_tokens):
        status = "exact"
    elif _contains_subsequence(hyp_tokens, norm_entity_tokens):
        status = "normalized"
    else:
        status = "omitted"

    return EntityMatchResult(entity=entity, status=status, hypothesis_raw=hypothesis_raw or "")


def match_entities(
    entities: Iterable[EntitySpan],
    hypothesis_raw: str,
    config: NormalizationConfig = DEFAULT_NORMALIZATION,
) -> List[EntityMatchResult]:
    return [match_entity(e, hypothesis_raw, config) for e in entities]


def summarize(results: Sequence[EntityMatchResult]) -> dict:
    total = len(results)
    counts = {"exact": 0, "normalized": 0, "omitted": 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    recall = (counts["exact"] + counts["normalized"]) / total if total else None
    return {
        "n_entities": total,
        "n_exact": counts["exact"],
        "n_normalized": counts["normalized"],
        "n_omitted": counts["omitted"],
        "recall": recall,
    }
