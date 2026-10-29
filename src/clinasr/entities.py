"""Medical-entity preservation matching.

This module is the schema-agnostic matching engine. Parsing the real
EkaCare `medical_entities` annotation format into `EntitySpan` objects lives
in clinasr.medical_entities, which depends on this module (not the other
way round).

VALIDATION STATUS: these matching rules have NOT been validated against
human judgement on real ASR output yet. Summary numbers from `summarize()`
are therefore reported as an unvalidated "match_rate", never as entity
recall/precision, and carry `"validated": False`.

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

Substitution and false-positive detection are NOT implemented: deciding
that a hypothesis token sequence is a *wrong* entity (rather than ordinary
non-entity speech) needs a validated entity recognizer or lexicon for the
hypothesis side, which this project does not have yet. See docs/ISSUES.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from clinasr.normalization import DEFAULT_NORMALIZATION, NormalizationConfig, normalize_text


@dataclass(frozen=True)
class EntitySpan:
    text: str
    entity_type: Optional[str] = None
    entity_id: Optional[str] = None
    # Character offsets into the reference transcript, as (start, end)
    # pairs, end-exclusive. A tuple of pairs because the real annotation
    # format allows several spans per entity.
    offsets: Optional[Tuple[Tuple[int, int], ...]] = None


@dataclass(frozen=True)
class EntityMatchResult:
    entity: EntitySpan
    status: str  # "exact" | "normalized" | "omitted"
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
    match_rate = (counts["exact"] + counts["normalized"]) / total if total else None
    return {
        "n_entities": total,
        "n_exact": counts["exact"],
        "n_normalized": counts["normalized"],
        "n_omitted": counts["omitted"],
        "match_rate": match_rate,
        "validated": False,
    }


def summarize_by_type(results: Sequence[EntityMatchResult]) -> dict:
    """`summarize()` broken down by `entity.entity_type`."""
    groups: dict = {}
    for r in results:
        groups.setdefault(r.entity.entity_type, []).append(r)
    return {entity_type: summarize(group) for entity_type, group in sorted(groups.items(), key=lambda kv: str(kv[0]))}
