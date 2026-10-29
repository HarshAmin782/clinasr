"""Adapter for the EkaCare dataset's real `medical_entities` annotations.

Observed format (real records, 2026-10-29; see docs/WORKLOG.md): the field
is a JSON-encoded STRING holding a list of entries, each entry exactly

    [entity_text, category, [[start, end], ...]]

e.g. '[["azithromycin", "drugs", [[51, 63]]]]', where (start, end) are
end-exclusive character offsets into the record's `text`. An empty
annotation is the string '[]'. Observed categories: clinical_findings,
drugs, advices, diagnostics, misc_medical. There is NO separate dosage
category -- dosage strings such as "500 mg" or "25 mcg" appear inside
`drugs` entities, so dosage detection here is a documented heuristic.

Parsing is defensive even though no malformed record has been observed
yet: the full split has not been inspected, and a crash or a silently
dropped entity would bias any entity metric. Every parse keeps the raw
string and reports a status plus per-entry problems.

Nothing in this module reports entity precision/recall: the matching rules
in clinasr.entities have not been validated (see `validated: False`).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from clinasr.entities import EntitySpan, match_entities, summarize, summarize_by_type
from clinasr.normalization import DEFAULT_NORMALIZATION, NormalizationConfig, normalize_text

STATUS_OK = "ok"
STATUS_EMPTY = "empty"
STATUS_MISSING = "missing"
STATUS_MALFORMED = "malformed"
STATUS_PARTIAL = "partial"
STATUS_UNUSABLE = "unusable"

DRUG_CATEGORY = "drugs"

# Heuristic only: a number followed by a common dose unit.
_DOSAGE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|gm|ml|iu|units?|%)(?![a-z])",
    re.IGNORECASE,
)

# Captures number + unit on already-normalized text (v2 splits "400mg" -> "400 mg").
_DOSE_CAPTURE = re.compile(r"\b(\d+(?:\.\d+)?) (mg|mcg|µg|gm|g|ml|iu|units?)(?![a-z])")


@dataclass(frozen=True)
class ParsedAnnotation:
    raw: Any
    status: str
    entities: Tuple[EntitySpan, ...] = ()
    problems: Tuple[str, ...] = ()


def _parse_offsets(value: Any) -> Optional[Tuple[Tuple[int, int], ...]]:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    pairs = []
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return None
        start, end = pair
        if isinstance(start, bool) or isinstance(end, bool):
            return None
        if not isinstance(start, int) or not isinstance(end, int):
            return None
        if start < 0 or end <= start:
            return None
        pairs.append((start, end))
    return tuple(pairs)


def parse_medical_entities(raw: Any) -> ParsedAnnotation:
    """Parse one record's raw `medical_entities` value. Never raises."""
    if raw is None:
        return ParsedAnnotation(raw=raw, status=STATUS_MISSING)
    if isinstance(raw, str):
        if not raw.strip():
            return ParsedAnnotation(raw=raw, status=STATUS_MISSING, problems=("blank_string",))
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            return ParsedAnnotation(raw=raw, status=STATUS_MALFORMED, problems=(f"invalid_json: {exc}",))
    elif isinstance(raw, list):
        data = raw  # tolerate an already-decoded list
    else:
        return ParsedAnnotation(raw=raw, status=STATUS_MALFORMED, problems=(f"unexpected_type: {type(raw).__name__}",))

    if not isinstance(data, list):
        return ParsedAnnotation(raw=raw, status=STATUS_MALFORMED, problems=(f"top_level_not_list: {type(data).__name__}",))
    if not data:
        return ParsedAnnotation(raw=raw, status=STATUS_EMPTY)

    entities: List[EntitySpan] = []
    problems: List[str] = []
    for index, entry in enumerate(data):
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            problems.append(f"entry_{index}: not_a_3_element_list")
            continue
        text, category, offsets_raw = entry
        if not isinstance(text, str) or not text.strip():
            problems.append(f"entry_{index}: missing_text")
            continue
        if not isinstance(category, str) or not category.strip():
            problems.append(f"entry_{index}: missing_category")
            continue
        offsets = _parse_offsets(offsets_raw)
        if offsets is None:
            # Entity text is still usable for text matching; keep it.
            problems.append(f"entry_{index}: bad_offsets")
        entities.append(EntitySpan(text=text, entity_type=category, entity_id=str(index), offsets=offsets))

    if not entities:
        status = STATUS_UNUSABLE
    elif problems:
        status = STATUS_PARTIAL
    else:
        status = STATUS_OK
    return ParsedAnnotation(raw=raw, status=status, entities=tuple(entities), problems=tuple(problems))


def offsets_align(entity: EntitySpan, reference_text: Optional[str]) -> Optional[bool]:
    """True if every offset span slices exactly the entity text out of the
    reference (case-sensitive); None if there is nothing to check."""
    if not entity.offsets or reference_text is None:
        return None
    return all(reference_text[start:end] == entity.text for start, end in entity.offsets)


def is_dosage_like(text: str) -> bool:
    """Heuristic: does this text contain a number + dose unit (e.g. "500 mg")?"""
    return bool(_DOSAGE_PATTERN.search(text or ""))


def assert_unique_ids(ids: Sequence[Any], id_name: str = "file_name") -> None:
    """Raise ValueError if any id is missing or repeated. Stage 2 compares
    models by sample id, so a duplicate would silently corrupt every join."""
    missing = [i for i, value in enumerate(ids) if value is None or (isinstance(value, str) and not value.strip())]
    if missing:
        raise ValueError(f"{len(missing)} record(s) have no {id_name} (positions {missing[:10]})")
    duplicates = sorted(str(value) for value, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {id_name} values ({len(duplicates)}): {duplicates[:10]}")


def _problem_kind(problem: str) -> str:
    # "entry_3: bad_offsets" -> "bad_offsets"; "invalid_json: <msg>" -> "invalid_json"
    if problem.startswith("entry_"):
        return problem.split(": ", 1)[-1]
    return problem.split(":", 1)[0]


@dataclass
class CoverageReport:
    n_records: int = 0
    status_counts: Dict[str, int] = field(default_factory=dict)
    n_samples_with_entities: int = 0
    n_entities: int = 0
    category_entity_counts: Dict[str, int] = field(default_factory=dict)
    category_sample_counts: Dict[str, int] = field(default_factory=dict)
    n_samples_with_drugs: int = 0
    n_dosage_like_drug_entities: int = 0
    n_samples_with_dosage_like_drug_entity: int = 0
    n_dosage_like_entities_other_categories: int = 0
    n_entities_with_offsets: int = 0
    n_offsets_aligned: int = 0
    n_offsets_misaligned: int = 0
    misaligned_examples: List[Dict[str, Any]] = field(default_factory=list)
    problem_counts: Dict[str, int] = field(default_factory=dict)
    unusable_sample_ids: List[str] = field(default_factory=list)


def analyze_coverage(
    records: Iterable[Dict[str, Any]],
    id_field: str = "file_name",
    text_field: str = "text",
    entities_field: str = "medical_entities",
    max_examples: int = 10,
) -> CoverageReport:
    report = CoverageReport()
    status_counts: Counter = Counter()
    category_entities: Counter = Counter()
    category_samples: Counter = Counter()
    problem_counts: Counter = Counter()

    for record in records:
        report.n_records += 1
        sample_id = record.get(id_field)
        parsed = parse_medical_entities(record.get(entities_field))
        status_counts[parsed.status] += 1
        for problem in parsed.problems:
            problem_counts[_problem_kind(problem)] += 1
        if parsed.status in (STATUS_MISSING, STATUS_MALFORMED, STATUS_UNUSABLE):
            report.unusable_sample_ids.append(str(sample_id))
        if not parsed.entities:
            continue

        report.n_samples_with_entities += 1
        report.n_entities += len(parsed.entities)
        sample_has_dosage_drug = False
        for category in {e.entity_type for e in parsed.entities}:
            category_samples[category] += 1
        for entity in parsed.entities:
            category_entities[entity.entity_type] += 1
            dosage_like = is_dosage_like(entity.text)
            if entity.entity_type == DRUG_CATEGORY and dosage_like:
                report.n_dosage_like_drug_entities += 1
                sample_has_dosage_drug = True
            elif dosage_like:
                report.n_dosage_like_entities_other_categories += 1
            aligned = offsets_align(entity, record.get(text_field))
            if aligned is None:
                continue
            report.n_entities_with_offsets += 1
            if aligned:
                report.n_offsets_aligned += 1
            else:
                report.n_offsets_misaligned += 1
                if len(report.misaligned_examples) < max_examples:
                    text = record.get(text_field) or ""
                    report.misaligned_examples.append(
                        {
                            "sample_id": sample_id,
                            "entity_text": entity.text,
                            "offsets": [list(p) for p in entity.offsets],
                            "reference_slices": [text[s:e] for s, e in entity.offsets],
                        }
                    )
        if any(e.entity_type == DRUG_CATEGORY for e in parsed.entities):
            report.n_samples_with_drugs += 1
        if sample_has_dosage_drug:
            report.n_samples_with_dosage_like_drug_entity += 1

    report.status_counts = dict(status_counts)
    report.category_entity_counts = dict(category_entities.most_common())
    report.category_sample_counts = dict(category_samples.most_common())
    report.problem_counts = dict(problem_counts)
    return report


def summarize_strata(records: Iterable[Dict[str, Any]], keys: Sequence[str]) -> Dict[str, Any]:
    """Counts per value of each stratification key, and per key combination,
    plus duration totals -- input for designing a stratified subset."""
    per_key: Dict[str, Counter] = {k: Counter() for k in keys}
    combos: Counter = Counter()
    combo_duration: Counter = Counter()
    n = 0
    total_duration = 0.0
    for record in records:
        n += 1
        values = tuple(str(record.get(k)) for k in keys)
        for k, v in zip(keys, values):
            per_key[k][v] += 1
        combos[values] += 1
        duration = record.get("duration")
        if isinstance(duration, (int, float)):
            total_duration += float(duration)
            combo_duration[values] += float(duration)
    return {
        "n_records": n,
        "total_duration_seconds": total_duration,
        "per_key": {k: dict(c.most_common()) for k, c in per_key.items()},
        "per_combination": [
            {"values": dict(zip(keys, combo)), "count": count, "duration_seconds": combo_duration[combo]}
            for combo, count in combos.most_common()
        ],
    }


def evaluate_sample_entities(
    sample_id: str,
    parsed: ParsedAnnotation,
    hypothesis: Optional[str],
    config: NormalizationConfig = DEFAULT_NORMALIZATION,
) -> Dict[str, Any]:
    """Per-entity match rows for one sample plus summaries.

    Returns unvalidated match labels (exact / normalized / omitted) only.
    Do not report the summary as recall until the matching rules have been
    checked against human judgement on real ASR output.
    """
    results = match_entities(parsed.entities, hypothesis or "", config)
    rows = [
        {
            "sample_id": sample_id,
            "entity_id": r.entity.entity_id,
            "entity_text": r.entity.text,
            "entity_type": r.entity.entity_type,
            "dosage_like": is_dosage_like(r.entity.text),
            "match_status": r.status,
            "hypothesis": r.hypothesis_raw,
        }
        for r in results
    ]
    return {
        "sample_id": sample_id,
        "annotation_status": parsed.status,
        "annotation_raw": parsed.raw,
        "rows": rows,
        "summary": summarize(results),
        "summary_by_type": summarize_by_type(results),
    }


def dose_expressions(text: str, config: NormalizationConfig = DEFAULT_NORMALIZATION) -> List[str]:
    """Normalized number+unit strings inside `text`, e.g.
    "Nervsenz 75mg/10mg Tablet" -> ["75 mg", "10 mg"]."""
    return [f"{m.group(1)} {m.group(2).lower()}" for m in _DOSE_CAPTURE.finditer(normalize_text(text, config))]


def dose_preserved(entity_text: str, hypothesis: Optional[str], config: NormalizationConfig = DEFAULT_NORMALIZATION) -> Optional[bool]:
    """True if every dose expression in the entity appears as a token
    sequence in the normalized hypothesis; None if the entity has no dose.

    Separates "was the dose kept" from "was the brand name kept". Strict:
    spoken forms ("five hundred", "grams per deciliter") count as not
    preserved. Unvalidated like all entity matching here.
    """
    doses = dose_expressions(entity_text, config)
    if not doses:
        return None
    hyp_tokens = normalize_text(hypothesis or "", config).split()
    for dose in doses:
        needle = dose.split()
        if not any(hyp_tokens[i : i + len(needle)] == needle for i in range(len(hyp_tokens) - len(needle) + 1)):
            return False
    return True
