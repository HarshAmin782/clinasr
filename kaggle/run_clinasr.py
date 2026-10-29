"""GENERATED FILE -- DO NOT EDIT DIRECTLY.

Built by scripts/build_kaggle_script.py from src/clinasr/*.py.
Source commit: 27b4cd080c26
To change behavior, edit the source modules under src/clinasr/ and
rebuild. This file is what Kaggle CLI uploads as the single code_file.
"""

from __future__ import annotations

# ---- begin src/clinasr/normalization.py (sha256:2b256ceb9d98046c) ----
"""Text normalization for WER/CER scoring.

Policy (documented per PROJECT_SPEC.md "Measurements" section):
- Lowercasing is applied to both reference and hypothesis symmetrically, so it
  never changes which words are counted as errors.
- Punctuation is stripped EXCEPT when it sits between two digits, so dosage
  and frequency expressions such as "2.5 mg", "1/2 tablet" or a "1-0-1"
  dosing schedule keep their decimal points, slashes, and hyphens.
- Digits themselves are never rewritten (no digit<->word conversion). This is
  a deliberate, conservative choice: automatic number normalization is a
  common source of silently hidden transcription errors in medical ASR
  evaluation, and this project prioritizes not over-normalizing dosage-
  relevant content over cosmetic WER improvements. See docs/ISSUES.md for the
  known limitation this creates (inconsistent spelled-out vs numeral forms
  are NOT reconciled).
- Whitespace is collapsed to single spaces after the above steps.
"""


import re
import unicodedata
from dataclasses import dataclass

# Punctuation characters considered strippable.
_PUNCT_CHARS = set("""!"#$%&'()*+,./:;<=>?@[]^_`{|}~""")
_WHITESPACE_PATTERN = re.compile(r"\s+")
# v2 rule (declared 2026-10-29 after Stage 2 showed "400mg" vs "400 mg"
# counted as a word error): put a space between a number and a directly
# attached dose unit. Applied identically to reference and hypothesis.
_NUMBER_UNIT_PATTERN = re.compile(r"(\d)(mg|mcg|gm|g|ml|iu|units?)(?![a-z])")


# Unicode hyphens/dashes that v3 maps to ASCII "-" before hyphen handling.
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")


def _split_hyphens(text: str) -> str:
    """v3 rule: a hyphen becomes a word break, EXCEPT
    - between two digits (ranges / schedules: "6-8", "1-0-1"), and
    - as a minus sign: followed by a digit and preceded by start-of-text or
      a non-alphanumeric character ("-5", "temp -2.5").
    This is a lexical rule ("statin-induced" == "statin induced"); it does
    not claim the hyphenated and spaced forms are clinically equivalent in
    general, only that they are spelling variants of the same words.
    """
    chars = list(text.translate(_DASHES))
    for i, ch in enumerate(chars):
        if ch != "-":
            continue
        prev = chars[i - 1] if i > 0 else ""
        nxt = chars[i + 1] if i + 1 < len(chars) else ""
        if prev.isdigit() and nxt.isdigit():
            continue
        if nxt.isdigit() and (prev == "" or not prev.isalnum()):
            continue
        chars[i] = " "
    return "".join(chars)


def _strip_punctuation(text: str, preserve_inter_digit: bool) -> str:
    """Replace strippable punctuation with a space, one character at a time.

    When preserve_inter_digit is True, a punctuation character flanked by a
    digit on BOTH sides (e.g. the "." in "2.5", the "/" in "1/2", the "-" in
    "1-0-1") is kept as-is instead of being blanked out.
    """
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch not in _PUNCT_CHARS:
            continue
        if preserve_inter_digit:
            prev_digit = i > 0 and chars[i - 1].isdigit()
            next_digit = i + 1 < len(chars) and chars[i + 1].isdigit()
            if prev_digit and next_digit:
                continue
        chars[i] = " "
    return "".join(chars)


@dataclass(frozen=True)
class NormalizationConfig:
    lowercase: bool = True
    strip_punctuation: bool = True
    preserve_inter_digit_punctuation: bool = True
    collapse_whitespace: bool = True
    unicode_normalize_form: str = "NFKC"
    split_number_unit: bool = True
    # v3: hyphens were never in _PUNCT_CHARS, so v1/v2 kept "statin-induced"
    # as one token. Off by default so v2 results stay exactly reproducible.
    split_hyphens: bool = False
    version: str = "v2"


DEFAULT_NORMALIZATION = NormalizationConfig()
# The policy Stage 2 (run 20261029T162830Z) was scored with, kept so those
# numbers stay reproducible.
NORMALIZATION_V1 = NormalizationConfig(split_number_unit=False, version="v1")
NORMALIZATION_V2 = DEFAULT_NORMALIZATION
NORMALIZATION_V3 = NormalizationConfig(split_hyphens=True, version="v3")
NORMALIZATION_VERSIONS = {"v1": NORMALIZATION_V1, "v2": NORMALIZATION_V2, "v3": NORMALIZATION_V3}


def normalize_text(text: str, config: NormalizationConfig = DEFAULT_NORMALIZATION) -> str:
    """Apply the documented normalization policy to a single text string.

    Returns an empty string for None/empty input rather than raising, since
    callers (metrics.py) must define an explicit, non-crashing denominator
    policy for missing hypotheses.
    """
    if text is None:
        return ""
    if config.unicode_normalize_form:
        text = unicodedata.normalize(config.unicode_normalize_form, text)
    if config.lowercase:
        text = text.lower()
    if config.split_hyphens:
        text = _split_hyphens(text)
    if config.strip_punctuation:
        text = _strip_punctuation(text, config.preserve_inter_digit_punctuation)
    if config.split_number_unit:
        text = _NUMBER_UNIT_PATTERN.sub(r"\1 \2", text)
    if config.collapse_whitespace:
        text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text
# ---- end src/clinasr/normalization.py ----

# ---- begin src/clinasr/metrics.py (sha256:19c6e8ace505e2df) ----
"""WER/CER computation shared by local tests and the bundled Kaggle script.

Metric denominator policy (PROJECT_SPEC.md "Measurements"):
- Sample-level WER/CER: jiwer's standard ratio (edits / reference length) for
  that single sample. If the reference is empty, the sample is excluded from
  sample-level aggregation (undefined denominator) and flagged in the
  returned record via `valid=False`, rather than raising or silently
  reporting 0.0/1.0.
- Corpus-level WER/CER: computed from SUMMED edits over SUMMED reference
  length across all valid samples (not the mean of per-sample ratios), which
  is the standard, less length-biased corpus aggregation.
- Missing/invalid hypotheses (None or empty string where a reference is
  non-empty) are treated as a fully-wrong prediction: scored against the
  reference as an empty hypothesis (all reference words/chars counted as
  deletions), not dropped, so model failures are not hidden from the corpus
  metric.

`jiwer` is imported lazily inside the functions that need it, not at module
level. The bundled kaggle/run_clinasr.py concatenates every src/clinasr
module into one file, so a top-level `import jiwer` would make even
`--self-test` (which never computes a metric) fail at script load on a bare
Kaggle image that hasn't installed jiwer yet. See docs/ISSUES.md #3.
"""


from dataclasses import dataclass
from typing import List, Optional, Sequence



@dataclass(frozen=True)
class SampleMetric:
    sample_id: str
    valid: bool
    wer: Optional[float]
    cer: Optional[float]
    reference_norm: str
    hypothesis_norm: str
    reason_invalid: Optional[str] = None


@dataclass(frozen=True)
class CorpusMetric:
    n_samples_total: int
    n_samples_valid: int
    n_samples_excluded: int
    wer: Optional[float]
    cer: Optional[float]


def _safe_hypothesis(hypothesis: Optional[str]) -> str:
    return hypothesis if hypothesis is not None else ""


def compute_sample_metric(
    sample_id: str,
    reference: Optional[str],
    hypothesis: Optional[str],
    config: NormalizationConfig = DEFAULT_NORMALIZATION,
) -> SampleMetric:
    ref_norm = normalize_text(reference, config)
    hyp_norm = normalize_text(_safe_hypothesis(hypothesis), config)

    if not ref_norm:
        return SampleMetric(
            sample_id=sample_id,
            valid=False,
            wer=None,
            cer=None,
            reference_norm=ref_norm,
            hypothesis_norm=hyp_norm,
            reason_invalid="empty_reference",
        )

    import jiwer

    word_out = jiwer.process_words([ref_norm], [hyp_norm])
    char_out = jiwer.process_characters([ref_norm], [hyp_norm])
    return SampleMetric(
        sample_id=sample_id,
        valid=True,
        wer=word_out.wer,
        cer=char_out.cer,
        reference_norm=ref_norm,
        hypothesis_norm=hyp_norm,
    )


def compute_corpus_metric(samples: Sequence[SampleMetric]) -> CorpusMetric:
    valid = [s for s in samples if s.valid]
    excluded = len(samples) - len(valid)
    if not valid:
        return CorpusMetric(
            n_samples_total=len(samples),
            n_samples_valid=0,
            n_samples_excluded=excluded,
            wer=None,
            cer=None,
        )

    import jiwer

    refs = [s.reference_norm for s in valid]
    hyps = [s.hypothesis_norm for s in valid]
    word_out = jiwer.process_words(refs, hyps)
    char_out = jiwer.process_characters(refs, hyps)
    return CorpusMetric(
        n_samples_total=len(samples),
        n_samples_valid=len(valid),
        n_samples_excluded=excluded,
        wer=word_out.wer,
        cer=char_out.cer,
    )


def compute_metrics(
    sample_ids: Sequence[str],
    references: Sequence[Optional[str]],
    hypotheses: Sequence[Optional[str]],
    config: NormalizationConfig = DEFAULT_NORMALIZATION,
) -> List[SampleMetric]:
    if not (len(sample_ids) == len(references) == len(hypotheses)):
        raise ValueError("sample_ids, references, and hypotheses must be the same length")
    return [
        compute_sample_metric(sid, ref, hyp, config)
        for sid, ref, hyp in zip(sample_ids, references, hypotheses)
    ]
# ---- end src/clinasr/metrics.py ----

# ---- begin src/clinasr/entities.py (sha256:2650bb25c957a318) ----
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


from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple



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
# ---- end src/clinasr/entities.py ----

# ---- begin src/clinasr/medical_entities.py (sha256:eb09da71c55bb837) ----
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


import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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
# ---- end src/clinasr/medical_entities.py ----

# ---- begin src/clinasr/noise.py (sha256:68b1b8316d1cbb82) ----
"""Seeded additive white-noise augmentation at a target SNR.

Safety policy for edge cases (required by the repository-hardening task;
see prompts/04-robustness.md):
- Signal with (near-)zero power (silence, or shorter than `min_samples`):
  noise cannot be scaled to a meaningful target SNR (division by ~0 power).
  The original signal is returned UNCHANGED with status="skipped_silent" (or
  "skipped_short"), never a divide-by-zero crash and never a fabricated
  noisy signal.
- `snr_db=None` is the explicit "clean" condition: a deliberate no-op,
  returned with status="clean", not an error.
- NaN/inf SNR values, or a `snr_db` not drawn from the configured condition
  set, raise ValueError: those indicate a configuration bug, not a
  recoverable per-sample audio condition, and silently clamping them would
  hide the bug.
- The noisy signal is peak-normalized back into [-1, 1] ONLY if adding noise
  pushed it outside that range (clipping would otherwise distort the
  waveform); otherwise the signal is left exactly as computed. This is
  recorded on the result as `was_clipped`.
- Noise is deterministic for a given (seed, sample_id, snr_db) triple via a
  dedicated `numpy.random.default_rng` seeded from a stable hash, so re-runs
  and different SNR conditions for the same sample use independent but
  reproducible noise draws.
"""


import hashlib
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

MIN_SAMPLES = 16  # shorter clips are not considered meaningful for SNR scaling
SILENCE_POWER_EPS = 1e-9


@dataclass(frozen=True)
class NoiseResult:
    signal: np.ndarray
    status: str  # "clean" | "ok" | "skipped_silent" | "skipped_short"
    requested_snr_db: Optional[float]
    achieved_snr_db: Optional[float]
    was_clipped: bool


def _seed_for(base_seed: int, sample_id: str, snr_db: Optional[float]) -> int:
    key = f"{base_seed}|{sample_id}|{snr_db}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _rms_power(signal: np.ndarray) -> float:
    if signal.size == 0:
        return 0.0
    return float(np.mean(np.square(signal, dtype=np.float64)))


def add_noise_at_snr(
    signal: np.ndarray,
    snr_db: Optional[float],
    base_seed: int,
    sample_id: str,
) -> NoiseResult:
    if snr_db is None:
        return NoiseResult(
            signal=np.asarray(signal),
            status="clean",
            requested_snr_db=None,
            achieved_snr_db=None,
            was_clipped=False,
        )

    if not math.isfinite(snr_db):
        raise ValueError(f"snr_db must be finite, got {snr_db!r}")

    signal = np.asarray(signal, dtype=np.float64)

    if signal.shape[0] < MIN_SAMPLES:
        return NoiseResult(
            signal=signal,
            status="skipped_short",
            requested_snr_db=snr_db,
            achieved_snr_db=None,
            was_clipped=False,
        )

    signal_power = _rms_power(signal)
    if signal_power < SILENCE_POWER_EPS:
        return NoiseResult(
            signal=signal,
            status="skipped_silent",
            requested_snr_db=snr_db,
            achieved_snr_db=None,
            was_clipped=False,
        )

    rng = np.random.default_rng(_seed_for(base_seed, sample_id, snr_db))
    noise = rng.standard_normal(signal.shape[0])
    noise_power = _rms_power(noise)

    target_noise_power = signal_power / (10 ** (snr_db / 10))
    scale = math.sqrt(target_noise_power / noise_power) if noise_power > 0 else 0.0
    scaled_noise = noise * scale

    noisy = signal + scaled_noise
    achieved_noise_power = _rms_power(scaled_noise)
    achieved_snr_db = (
        10 * math.log10(signal_power / achieved_noise_power)
        if achieved_noise_power > 0
        else float("inf")
    )

    peak = float(np.max(np.abs(noisy))) if noisy.size else 0.0
    was_clipped = peak > 1.0
    if was_clipped:
        noisy = noisy / peak

    return NoiseResult(
        signal=noisy,
        status="ok",
        requested_snr_db=snr_db,
        achieved_snr_db=achieved_snr_db,
        was_clipped=was_clipped,
    )
# ---- end src/clinasr/noise.py ----

# ---- begin src/clinasr/config.py (sha256:a550635ff4a7bb65) ----
"""Experiment configuration schema.

One `ExperimentConfig` is created per run and written into that run's
manifest (see run_manifest.py). It is the single place that records what
was actually fixed for a run: seeds, sample selection, model identities and
their RESOLVED revisions, normalization policy, noise parameters, and which
metrics were computed. Package/model versions are never hardcoded here —
`resolve_installed_versions()` reads them from the actual environment at
run time via `importlib.metadata`, per the research-integrity rule against
inventing version/revision numbers.
"""


import dataclasses
import importlib.metadata
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional



@dataclass
class ModelSpec:
    model_id: str
    # Populated at run time from the loaded model/tokenizer, e.g. the
    # Hugging Face Hub commit SHA actually resolved for `model_id`. Left
    # None until a real load happens; never guessed ahead of time.
    resolved_revision: Optional[str] = None


@dataclass
class SampleSelection:
    method: str  # e.g. "smoke_test_first_n" | "stratified_fixed_subset"
    size: int
    stratify_by: List[str] = field(default_factory=list)
    sample_ids: List[str] = field(default_factory=list)


@dataclass
class NoiseConfig:
    snrs_db: List[Optional[float]] = field(default_factory=lambda: [None, 20.0, 10.0, 0.0])
    seed: int = 0


@dataclass
class ExperimentConfig:
    dataset_id: str
    dataset_config: str
    dataset_split: str
    seed: int
    sample_selection: SampleSelection
    models: List[ModelSpec]
    normalization: NormalizationConfig
    noise: NoiseConfig
    metrics: List[str] = field(default_factory=lambda: ["wer", "cer"])
    # Filled via resolve_installed_versions(); not populated at construction.
    software_versions: Dict[str, str] = field(default_factory=dict)
    dataset_revision: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        data = dict(data)
        data["sample_selection"] = SampleSelection(**data["sample_selection"])
        data["models"] = [ModelSpec(**m) for m in data["models"]]
        data["normalization"] = NormalizationConfig(**data["normalization"])
        data["noise"] = NoiseConfig(**data["noise"])
        return cls(**data)


def resolve_installed_versions(package_names: List[str]) -> Dict[str, str]:
    """Return actually-installed versions for the given packages.

    Packages that are not installed in the current environment are recorded
    as "not-installed" rather than omitted, so a report never implies a
    version that was never actually checked.
    """
    versions: Dict[str, str] = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions
# ---- end src/clinasr/config.py ----

# ---- begin src/clinasr/run_manifest.py (sha256:a251307848a4db35) ----
"""Run isolation: run IDs, per-run result directories, and manifests.

Policy: every run gets its own directory under the results root, named from
a timestamp plus the short git commit hash of the code that produced it (or
"nogit" if no repository/commit is available, e.g. a Kaggle sandbox without
.git). A run never overwrites another run's directory; collisions raise
FileExistsError, since this project's integrity rules require every result
to be traceable to exactly one run's manifest.
"""


import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def get_git_commit(repo_dir: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(repo_dir) if repo_dir else None,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def make_run_id(now: Optional[datetime] = None, git_commit: Optional[str] = None) -> str:
    now = now or datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    commit = git_commit if git_commit else "nogit"
    return f"{stamp}_{commit}"


@dataclass
class RunHandle:
    run_id: str
    run_dir: Path
    manifest_path: Path
    log_path: Path

    def write_manifest(self, extra: Dict[str, Any]) -> None:
        manifest = {
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            **extra,
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")

    def get_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"clinasr.run.{self.run_id}")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.FileHandler(self.log_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(handler)
        return logger


def new_run(results_root: Path, git_commit: Optional[str] = None) -> RunHandle:
    """Create a fresh, never-overwritten run directory under results_root."""
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    commit = git_commit if git_commit is not None else get_git_commit()
    run_id = make_run_id(git_commit=commit)
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)  # raises FileExistsError on any collision
    return RunHandle(
        run_id=run_id,
        run_dir=run_dir,
        manifest_path=run_dir / "manifest.json",
        log_path=run_dir / "run.log",
    )
# ---- end src/clinasr/run_manifest.py ----

# ---- begin src/clinasr/schema_audit.py (sha256:6ccd053b8abb2161) ----
"""Dataset schema auditing, run BEFORE any stratified sampling.

PROJECT_SPEC.md lists `audio`, `text`, `session_id`, `speaker`,
`recording_context`, `type_concept`, `medical_entities` as "expected
fields... inspect schema/annotation representation before use" — i.e. the
spec itself says not to assume they exist. `audit_records()` below is the
generic, dependency-free core: it works on any list of dict-like records,
so it is fully unit-testable without a network call or the `datasets`
package installed. `audit_hf_dataset()` is the thin, real-dataset wrapper
that Phase 1 runs once (locally or on Kaggle) against a small streamed
sample of ekacare/eka-medical-asr-evaluation-dataset before sampling the
200-400 item benchmark subset; it imports `datasets` lazily so that merely
importing this module never requires that dependency.
"""


from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class FieldReport:
    name: str
    observed_types: List[str]
    n_missing: int
    n_present: int
    example_value_repr: Optional[str]


@dataclass
class SchemaAuditReport:
    n_records_inspected: int
    fields: Dict[str, FieldReport] = field(default_factory=dict)

    def missing_expected(self, expected_fields: List[str]) -> List[str]:
        return [f for f in expected_fields if f not in self.fields]


def _type_name(value: Any) -> str:
    return type(value).__name__


def audit_records(records: Iterable[Dict[str, Any]], example_repr_max_len: int = 120) -> SchemaAuditReport:
    """Inspect field names, observed Python types, and missing-value counts.

    A field is "missing" for a record if the key is absent OR its value is
    None. Every record's full key set contributes to `fields`, so a field
    that only appears on some records (schema drift) is still reported with
    an accurate n_present/n_missing split rather than silently ignored.
    """
    records = list(records)
    field_names: List[str] = []
    for record in records:
        for key in record.keys():
            if key not in field_names:
                field_names.append(key)

    reports: Dict[str, FieldReport] = {}
    for name in field_names:
        observed_types: List[str] = []
        n_missing = 0
        n_present = 0
        example_repr: Optional[str] = None
        for record in records:
            value = record.get(name, None)
            if value is None:
                n_missing += 1
                continue
            n_present += 1
            type_name = _type_name(value)
            if type_name not in observed_types:
                observed_types.append(type_name)
            if example_repr is None:
                text = repr(value)
                example_repr = text if len(text) <= example_repr_max_len else text[:example_repr_max_len] + "..."
        reports[name] = FieldReport(
            name=name,
            observed_types=observed_types,
            n_missing=n_missing,
            n_present=n_present,
            example_value_repr=example_repr,
        )

    return SchemaAuditReport(n_records_inspected=len(records), fields=reports)


def audit_hf_dataset(
    dataset_id: str,
    config_name: str,
    split: str,
    n: int = 5,
    decode_audio: bool = False,
) -> SchemaAuditReport:
    """Stream the first `n` records of a Hugging Face dataset and audit them.

    Requires network access and the `datasets` package; neither is assumed
    to be available when this module is merely imported for unit testing.

    `decode_audio=False` (the default) casts any `datasets.Audio` column to
    `decode=False` before iterating, so the field comes back as a raw
    `{"bytes": ..., "path": ...}` dict instead of being decoded into a
    waveform array. A schema audit only needs field names/types/presence, not
    decoded audio, and as of `datasets` 4.x decoding requires the optional
    `torchcodec` package to be installed -- forcing that dependency onto a
    plain schema-inspection call would be disproportionate. Pass
    `decode_audio=True` once a real decode is actually needed and torchcodec
    (or an equivalent backend) is available.
    """
    import datasets  # local import: optional, heavy dependency

    stream = datasets.load_dataset(dataset_id, config_name, split=split, streaming=True)
    if not decode_audio:
        for column_name, feature in stream.features.items():
            if isinstance(feature, datasets.Audio):
                stream = stream.cast_column(column_name, datasets.Audio(decode=False))

    records = []
    for i, record in enumerate(stream):
        if i >= n:
            break
        records.append(dict(record))
    return audit_records(records)
# ---- end src/clinasr/schema_audit.py ----

# ---- begin src/clinasr/smoke_test.py (sha256:c2b36c1cfd77f2ae) ----
"""Phase 1, stage 1: five-sample Whisper-tiny.en smoke test on the real
EkaCare dataset. Runs on a Kaggle GPU; nothing here is unit-testable end to
end without one. Pure/data-shaping pieces (sample selection determinism,
CSV row construction, CSV writing) ARE split out and unit tested
(tests/test_smoke_test.py); the pieces that need torch/transformers/network
(model loading, audio decoding, transcription, HF revision resolution) are
isolated into their own small functions so the untestable surface is as
small as possible, per AGENTS.md's "report exact output and explain what
can't be validated until a Kaggle GPU run."

Research-integrity notes (AGENTS.md):
- Decoding settings are fixed and printed: greedy decoding (num_beams=1,
  do_sample=False), no custom language/task forcing (whisper-tiny.en is an
  English-only checkpoint).
- The model revision actually used is resolved from the loaded model's own
  config at run time (`_resolved_revision_from_model`), never hardcoded.
- Sample selection is `smoke_test_first_n`: the first N records of the
  streamed `test` split, in dataset order, for a fixed `n`. This is
  explicitly NOT a stratified or random sample (that is stage 2's job) --
  it exists only to prove the pipeline runs end to end.
"""


import csv
import dataclasses
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


DATASET_ID = "ekacare/eka-medical-asr-evaluation-dataset"
DATASET_CONFIG = "en"
DATASET_SPLIT = "test"
MODEL_ID = "openai/whisper-tiny.en"
SMOKE_TEST_N = 5
SEED = 0

# (pip install name, import name) -- import name differs for some packages.
BOOTSTRAP_PACKAGES = [
    ("datasets", "datasets"),
    ("transformers", "transformers"),
    ("jiwer", "jiwer"),
    ("torchcodec", "torchcodec"),
]


def ensure_installed(pip_name: str, import_name: Optional[str] = None) -> bool:
    """Import `import_name`, pip-installing `pip_name` first if it's missing.

    Returns True if the package is importable after this call, False if the
    install itself failed (e.g. no matching wheel for this CUDA/torch
    build) -- callers must handle False explicitly rather than assume
    success, since a failed audio-codec install should not silently produce
    empty transcriptions.
    """
    import_name = import_name or pip_name
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        pass
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", pip_name],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"WARNING: pip install {pip_name!r} failed:\n{result.stderr[-2000:]}")
        return False
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def bootstrap_all() -> Dict[str, bool]:
    return {pip_name: ensure_installed(pip_name, import_name) for pip_name, import_name in BOOTSTRAP_PACKAGES}


def load_smoke_records(n: int = SMOKE_TEST_N) -> List[Dict[str, Any]]:
    """Stream the first `n` records of the real test split, WITH decoded audio."""
    import datasets

    stream = datasets.load_dataset(DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT, streaming=True)
    records = []
    for i, record in enumerate(stream):
        if i >= n:
            break
        records.append(dict(record))
    return records


def resolve_model_revision(model_id: str) -> Optional[str]:
    """Return the Hub commit SHA the loaded model actually resolved to."""
    try:
        from huggingface_hub import model_info

        return model_info(model_id).sha
    except Exception as exc:  # pragma: no cover - network/HF-hub dependent
        print(f"WARNING: could not resolve revision for {model_id!r}: {exc}")
        return None


def transcribe_records(records: Sequence[Dict[str, Any]], model_id: str = MODEL_ID) -> List[str]:
    """Run whisper-tiny.en with a fixed, documented decoding configuration."""
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline("automatic-speech-recognition", model=model_id, device=device)

    hypotheses = []
    for record in records:
        audio = record["audio"]
        result = asr(
            {"array": audio["array"], "sampling_rate": audio["sampling_rate"]},
            generate_kwargs={"num_beams": 1, "do_sample": False},
        )
        hypotheses.append(result["text"])
    return hypotheses


def build_prediction_rows(
    records: Sequence[Dict[str, Any]],
    hypotheses: Sequence[str],
    model_id: str,
    condition: str = "clean",
) -> List[Dict[str, Any]]:
    """Pure, unit-testable: shape raw records+hypotheses into output rows."""
    if len(records) != len(hypotheses):
        raise ValueError("records and hypotheses must be the same length")
    rows = []
    for record, hypothesis in zip(records, hypotheses):
        rows.append(
            {
                # `session_id` is NOT a unique per-record identifier: the
                # real dataset's first 5 streamed records already contained
                # a repeated session_id across two different utterances
                # (one session = multiple recorded segments). `file_name`
                # is the real per-record unique path and is used as the
                # sample_id instead; `session_id` stays as its own grouping
                # column so cross-model joins by sample_id are unambiguous.
                # See docs/ISSUES.md.
                "sample_id": record.get("file_name"),
                "reference": record.get("text"),
                "hypothesis": hypothesis,
                "model": model_id,
                "condition": condition,
                "session_id": record.get("session_id"),
                "recording_context": record.get("recording_context"),
            }
        )
    return rows


def write_predictions_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    fieldnames = ["sample_id", "reference", "hypothesis", "model", "condition", "session_id", "recording_context"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_csv(
    model_id: str,
    condition: str,
    corpus_metric,
    path: Path,
) -> None:
    fieldnames = ["model", "condition", "wer", "cer", "n_samples_total", "n_samples_valid", "n_samples_excluded"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "model": model_id,
                "condition": condition,
                "wer": corpus_metric.wer,
                "cer": corpus_metric.cer,
                "n_samples_total": corpus_metric.n_samples_total,
                "n_samples_valid": corpus_metric.n_samples_valid,
                "n_samples_excluded": corpus_metric.n_samples_excluded,
            }
        )


def build_experiment_config(
    model_id: str,
    resolved_revision: Optional[str],
    n: int,
    sample_ids: Sequence[str],
) -> ExperimentConfig:
    versions = resolve_installed_versions(["jiwer", "numpy", "torch", "transformers", "datasets", "torchcodec"])
    return ExperimentConfig(
        dataset_id=DATASET_ID,
        dataset_config=DATASET_CONFIG,
        dataset_split=DATASET_SPLIT,
        seed=SEED,
        sample_selection=SampleSelection(method="smoke_test_first_n", size=n, sample_ids=list(sample_ids)),
        models=[ModelSpec(model_id=model_id, resolved_revision=resolved_revision)],
        normalization=DEFAULT_NORMALIZATION,
        noise=NoiseConfig(snrs_db=[None]),
        software_versions=versions,
    )


def run_smoke_test(output_dir: Path, n: int = SMOKE_TEST_N, model_id: str = MODEL_ID) -> int:
    """Orchestrates the real Phase-1 smoke test. Requires network + GPU/CPU
    inference; not run in unit tests. Writes predictions.csv,
    summary_metrics.csv, and experiment_config.json into `output_dir`
    (Kaggle: /kaggle/working), plus a run manifest/log under
    `output_dir/run/` for traceability.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Bootstrapping required packages...")
    bootstrap_results = bootstrap_all()
    for pkg, ok in bootstrap_results.items():
        print(f"  {pkg}: {'ok' if ok else 'FAILED'}")
    if not bootstrap_results.get("datasets") or not bootstrap_results.get("transformers"):
        print("ERROR: required packages missing after bootstrap; aborting smoke test.")
        return 1

    handle: RunHandle = new_run(output_dir / "run", git_commit=get_git_commit())
    logger = handle.get_logger()
    logger.info("Starting smoke test: model=%s n=%d", model_id, n)

    print(f"Streaming first {n} records from {DATASET_ID} ({DATASET_CONFIG}/{DATASET_SPLIT})...")
    records = load_smoke_records(n=n)
    sample_ids = [r.get("session_id") for r in records]
    print(f"Loaded {len(records)} records: {sample_ids}")

    print(f"Resolving revision for {model_id}...")
    resolved_revision = resolve_model_revision(model_id)
    print(f"Resolved revision: {resolved_revision}")

    print("Running inference (greedy, num_beams=1, do_sample=False)...")
    hypotheses = transcribe_records(records, model_id=model_id)
    for sid, hyp in zip(sample_ids, hypotheses):
        print(f"  {sid}: {hyp!r}")

    rows = build_prediction_rows(records, hypotheses, model_id=model_id, condition="clean")
    write_predictions_csv(rows, output_dir / "predictions.csv")

    sample_metrics = compute_metrics(
        [r["sample_id"] for r in rows],
        [r["reference"] for r in rows],
        [r["hypothesis"] for r in rows],
    )
    corpus_metric = compute_corpus_metric(sample_metrics)
    write_summary_csv(model_id, "clean", corpus_metric, output_dir / "summary_metrics.csv")
    print(
        f"Corpus WER={corpus_metric.wer} CER={corpus_metric.cer} "
        f"(valid={corpus_metric.n_samples_valid}/{corpus_metric.n_samples_total})"
    )

    config = build_experiment_config(model_id, resolved_revision, n, sample_ids)
    (output_dir / "experiment_config.json").write_text(config.to_json(), encoding="utf-8")

    handle.write_manifest(
        {
            "git_commit": get_git_commit(),
            "model_id": model_id,
            "resolved_revision": resolved_revision,
            "sample_ids": sample_ids,
            "corpus_metric": dataclasses.asdict(corpus_metric),
            "bootstrap_results": bootstrap_results,
        }
    )
    logger.info("Smoke test complete. wer=%s cer=%s", corpus_metric.wer, corpus_metric.cer)
    print(f"Done. Outputs written to {output_dir}")
    return 0
# ---- end src/clinasr/smoke_test.py ----

# ---- begin src/clinasr/sampling.py (sha256:8fb04cdb383e4050) ----
"""Deterministic stratified sample selection for Stage 2.

Selection must come out identical wherever it runs (local Python 3.9, the
Kaggle image's 3.13), so it uses no RNG at all: within a stratum, records
are ordered by sha256(f"{seed}:{sample_id}") and the first k are taken.
That ranking depends only on the seed and the ids, not on stream order,
Python version, or platform. `selection_fingerprint()` hashes the sorted
selected ids so a remote run can prove it evaluated exactly the set that
was reviewed locally.

Allocation: every stratum first gets min(stratum_size, min_per_stratum),
so small strata are represented; the remainder is distributed in
proportion to each stratum's remaining capacity (largest-remainder method,
ties broken by stratum key) so large strata are not over-represented
relative to the population more than necessary. Selection is fixed BEFORE
any model is run -- never re-drawn after looking at results.
"""


import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


StratumKey = Tuple[str, ...]


def stable_rank(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def selection_fingerprint(sample_ids: Iterable[str]) -> str:
    joined = "\n".join(sorted(sample_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def allocate(stratum_sizes: Dict[StratumKey, int], total: int, min_per_stratum: int) -> Dict[StratumKey, int]:
    """Return how many samples to draw from each stratum (sums to `total`)."""
    population = sum(stratum_sizes.values())
    if total <= 0:
        raise ValueError("total must be positive")
    if total > population:
        raise ValueError(f"requested {total} samples but only {population} eligible records exist")

    keys = sorted(stratum_sizes, key=lambda k: tuple(str(x) for x in k))
    base = {k: min(stratum_sizes[k], min_per_stratum) for k in keys}
    remaining = total - sum(base.values())
    if remaining < 0:
        raise ValueError(
            f"min_per_stratum={min_per_stratum} across {len(keys)} strata needs "
            f"{sum(base.values())} samples, more than total={total}"
        )

    capacity = {k: stratum_sizes[k] - base[k] for k in keys}
    total_capacity = sum(capacity.values())
    extra = {k: 0 for k in keys}
    if remaining and total_capacity:
        quotas = {k: remaining * capacity[k] / total_capacity for k in keys}
        extra = {k: int(quotas[k]) for k in keys}
        leftover = remaining - sum(extra.values())
        by_remainder = sorted(keys, key=lambda k: (-(quotas[k] - extra[k]), tuple(str(x) for x in k)))
        for k in by_remainder:
            if leftover == 0:
                break
            if extra[k] < capacity[k]:
                extra[k] += 1
                leftover -= 1
    result = {k: base[k] + extra[k] for k in keys}
    if sum(result.values()) != total:
        raise AssertionError(f"allocation sums to {sum(result.values())}, expected {total}")
    return result


@dataclass
class Selection:
    sample_ids: List[str]
    seed: int
    total_requested: int
    min_per_stratum: int
    strata_keys: List[str]
    allocation: Dict[str, int] = field(default_factory=dict)
    stratum_population: Dict[str, int] = field(default_factory=dict)
    n_records_seen: int = 0
    n_eligible: int = 0
    n_excluded: int = 0
    exclusion_reasons: Dict[str, int] = field(default_factory=dict)
    fingerprint: str = ""


def _label(key: StratumKey) -> str:
    return " | ".join(str(x) for x in key)


def stratified_select(
    records: Iterable[Dict[str, Any]],
    strata_keys: Sequence[str],
    total: int,
    seed: int,
    min_per_stratum: int,
    id_field: str = "file_name",
    exclude: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
) -> Selection:
    """Select `total` record ids, stratified by `strata_keys`.

    `exclude(record)` returns a reason string to drop a record from the
    eligible pool (e.g. too long for the decoding setup) or None to keep it.
    Raises ValueError on missing/duplicate ids.
    """
    records = list(records)
    assert_unique_ids([r.get(id_field) for r in records], id_field)

    reasons: Dict[str, int] = {}
    groups: Dict[StratumKey, List[str]] = {}
    for record in records:
        reason = exclude(record) if exclude else None
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        key = tuple(str(record.get(k)) for k in strata_keys)
        groups.setdefault(key, []).append(str(record[id_field]))

    sizes = {k: len(v) for k, v in groups.items()}
    allocation = allocate(sizes, total, min_per_stratum)

    selected: List[str] = []
    for key, k in allocation.items():
        ranked = sorted(groups[key], key=lambda sid: stable_rank(seed, sid))
        selected.extend(ranked[:k])
    selected.sort()

    return Selection(
        sample_ids=selected,
        seed=seed,
        total_requested=total,
        min_per_stratum=min_per_stratum,
        strata_keys=list(strata_keys),
        allocation={_label(k): v for k, v in allocation.items()},
        stratum_population={_label(k): sizes[k] for k in allocation},
        n_records_seen=len(records),
        n_eligible=sum(sizes.values()),
        n_excluded=sum(reasons.values()),
        exclusion_reasons=reasons,
        fingerprint=selection_fingerprint(selected),
    )
# ---- end src/clinasr/sampling.py ----

# ---- begin src/clinasr/stage2.py (sha256:1a1676c4818c7915) ----
"""Stage 2: Whisper-tiny.en vs Whisper-base.en on one fixed stratified subset.

Design (docs/WORKLOG.md has the rationale and the real population counts):
- Dataset pinned to a Hub revision; models pinned to Hub revisions.
- Strata: recording_context x type_concept. 300 samples, at least 20 per
  stratum, seed STAGE2_SEED, hash-ranked selection (clinasr.sampling).
- The selection is recomputed on Kaggle from streamed metadata and must
  match STAGE2_EXPECTED_FINGERPRINT (computed locally by
  scripts/select_stage2_samples.py and reviewed before any GPU run). A
  mismatch aborts the run before inference.
- Both models decode the identical audio with identical greedy settings.
- Entity matching is written out per entity but labelled unvalidated.
"""


import csv
import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


DATASET_ID = "ekacare/eka-medical-asr-evaluation-dataset"
DATASET_CONFIG = "en"
DATASET_SPLIT = "test"
DATASET_REVISION = "433e66037c8aebb8f396b938624a5ab6afc4ca36"
MODELS = {
    "openai/whisper-tiny.en": "87c7102498dcde7456f24cfd30239ca606ed9063",
    "openai/whisper-base.en": "911407f4214e0e1d82085af863093ec0b66f9cd6",
}
STRATA_KEYS = ["recording_context", "type_concept"]
STAGE2_TOTAL = 300
STAGE2_MIN_PER_STRATUM = 20
STAGE2_SEED = 20261029
# Set from the local, reviewed selection (scripts/select_stage2_samples.py).
STAGE2_EXPECTED_FINGERPRINT = "8b345d92bc204dedebe465065dac1993ac1e48c6cd55f408a12e47117d84a6c3"
BATCH_SIZE = 16
GENERATE_KWARGS = {"num_beams": 1, "do_sample": False}


def select(records: Sequence[Dict[str, Any]]) -> Selection:
    return stratified_select(
        records,
        STRATA_KEYS,
        total=STAGE2_TOTAL,
        seed=STAGE2_SEED,
        min_per_stratum=STAGE2_MIN_PER_STRATUM,
    )


def _stream(decode_audio: bool):
    import datasets

    stream = datasets.load_dataset(
        DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT, streaming=True, revision=DATASET_REVISION
    )
    if not decode_audio:
        stream = stream.cast_column("audio", datasets.Audio(decode=False)).remove_columns(["audio"])
    return stream


def load_metadata() -> List[Dict[str, Any]]:
    return [dict(r) for r in _stream(decode_audio=False)]


def load_selected_audio(sample_ids: Sequence[str]) -> List[Dict[str, Any]]:
    wanted = set(sample_ids)
    found: Dict[str, Dict[str, Any]] = {}
    for record in _stream(decode_audio=True):
        if record["file_name"] in wanted:
            found[record["file_name"]] = dict(record)
            if len(found) == len(wanted):
                break
    missing = wanted - set(found)
    if missing:
        raise RuntimeError(f"{len(missing)} selected samples not found in the stream: {sorted(missing)[:5]}")
    return [found[sid] for sid in sorted(found)]


def transcribe(records: Sequence[Dict[str, Any]], model_id: str, revision: str) -> Dict[str, Any]:
    import torch
    from transformers import pipeline

    use_cuda = torch.cuda.is_available()
    asr = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        revision=revision,
        device=0 if use_cuda else -1,
        dtype=torch.float16 if use_cuda else torch.float32,
    )
    inputs = [{"array": r["audio"]["array"], "sampling_rate": r["audio"]["sampling_rate"]} for r in records]
    start = time.perf_counter()
    outputs = asr(inputs, batch_size=BATCH_SIZE, generate_kwargs=dict(GENERATE_KWARGS))
    elapsed = time.perf_counter() - start
    return {
        "hypotheses": [o["text"] for o in outputs],
        "seconds": elapsed,
        "device": "cuda" if use_cuda else "cpu",
        "dtype": "float16" if use_cuda else "float32",
    }


def prediction_rows(records: Sequence[Dict[str, Any]], hypotheses: Sequence[str], model_id: str) -> List[Dict[str, Any]]:
    if len(records) != len(hypotheses):
        raise ValueError("records and hypotheses must be the same length")
    return [
        {
            "sample_id": r["file_name"],
            "model": model_id,
            "condition": "clean",
            "recording_context": r.get("recording_context"),
            "type_concept": r.get("type_concept"),
            "session_id": r.get("session_id"),
            "duration": r.get("duration"),
            "reference": r.get("text"),
            "hypothesis": h,
        }
        for r, h in zip(records, hypotheses)
    ]


def summary_rows(rows: Sequence[Dict[str, Any]], model_id: str, config=DEFAULT_NORMALIZATION) -> List[Dict[str, Any]]:
    """Corpus WER/CER overall and per stratum for one model's rows."""
    groups: Dict[str, List[Dict[str, Any]]] = {"ALL": list(rows)}
    for row in rows:
        groups.setdefault(f"{row['recording_context']} | {row['type_concept']}", []).append(row)
    out = []
    for stratum, group in groups.items():
        metrics = compute_metrics([g["sample_id"] for g in group], [g["reference"] for g in group], [g["hypothesis"] for g in group], config)
        corpus = compute_corpus_metric(metrics)
        out.append(
            {
                "model": model_id,
                "condition": "clean",
                "normalization": config.version,
                "stratum": stratum,
                "n_samples": corpus.n_samples_total,
                "n_valid": corpus.n_samples_valid,
                "wer": corpus.wer,
                "cer": corpus.cer,
            }
        )
    return out


def entity_rows(records: Sequence[Dict[str, Any]], hypotheses: Sequence[str], model_id: str) -> List[Dict[str, Any]]:
    rows = []
    for record, hypothesis in zip(records, hypotheses):
        parsed = parse_medical_entities(record.get("medical_entities"))
        result = evaluate_sample_entities(record["file_name"], parsed, hypothesis)
        for row in result["rows"]:
            row = dict(row)
            row.pop("hypothesis", None)  # already in predictions.csv
            row["model"] = model_id
            row["annotation_status"] = parsed.status
            row["matching_validated"] = False
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_stage2(output_dir: Path, expected_fingerprint: Optional[str] = None) -> int:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = expected_fingerprint or STAGE2_EXPECTED_FINGERPRINT

    bootstrap = bootstrap_all()
    print(f"bootstrap: {bootstrap}")
    if not all(bootstrap.values()):
        print("ERROR: package bootstrap failed; aborting before inference.")
        return 1

    handle = new_run(output_dir / "run", git_commit=get_git_commit())
    logger = handle.get_logger()

    metadata = load_metadata()
    selection = select(metadata)
    (output_dir / "selection.json").write_text(json.dumps(dataclasses.asdict(selection), indent=2), encoding="utf-8")
    print(f"selected {len(selection.sample_ids)} samples, fingerprint {selection.fingerprint}")
    if selection.fingerprint != expected:
        print(f"ERROR: selection fingerprint {selection.fingerprint} != expected {expected}; aborting before inference.")
        return 2

    records = load_selected_audio(selection.sample_ids)
    all_predictions, all_summaries, all_entities, timings = [], [], [], {}
    for model_id, revision in MODELS.items():
        print(f"transcribing with {model_id}@{revision[:10]} ...")
        result = transcribe(records, model_id, revision)
        timings[model_id] = {k: result[k] for k in ("seconds", "device", "dtype")}
        preds = prediction_rows(records, result["hypotheses"], model_id)
        all_predictions.extend(preds)
        all_summaries.extend(summary_rows(preds, model_id))
        all_entities.extend(entity_rows(records, result["hypotheses"], model_id))
        logger.info("%s done in %.1fs on %s", model_id, result["seconds"], result["device"])

    _write_csv(output_dir / "predictions.csv", all_predictions)
    _write_csv(output_dir / "summary_metrics.csv", all_summaries)
    _write_csv(output_dir / "entity_matches_unvalidated.csv", all_entities)

    config = ExperimentConfig(
        dataset_id=DATASET_ID,
        dataset_config=DATASET_CONFIG,
        dataset_split=DATASET_SPLIT,
        seed=STAGE2_SEED,
        sample_selection=SampleSelection(
            method="stratified_hash_ranked",
            size=len(selection.sample_ids),
            stratify_by=list(STRATA_KEYS),
            sample_ids=list(selection.sample_ids),
        ),
        models=[ModelSpec(model_id=m, resolved_revision=r) for m, r in MODELS.items()],
        normalization=DEFAULT_NORMALIZATION,
        noise=NoiseConfig(snrs_db=[None]),
        software_versions=resolve_installed_versions(["jiwer", "numpy", "torch", "transformers", "datasets", "torchcodec"]),
        dataset_revision=DATASET_REVISION,
    )
    (output_dir / "experiment_config.json").write_text(config.to_json(), encoding="utf-8")
    handle.write_manifest(
        {
            "stage": 2,
            "git_commit": get_git_commit(),
            "selection_fingerprint": selection.fingerprint,
            "timings": timings,
            "generate_kwargs": GENERATE_KWARGS,
            "batch_size": BATCH_SIZE,
        }
    )
    for row in all_summaries:
        if row["stratum"] == "ALL":
            print(f"{row['model']}: WER={row['wer']:.4f} CER={row['cer']:.4f} (n={row['n_valid']})")
    return 0
# ---- end src/clinasr/stage2.py ----

# ---- begin src/clinasr/stage4.py (sha256:1605bd05f3694591) ----
"""Stage 4: noise robustness on the Stage 2 sample set.

Same 300 samples (same fingerprint check as Stage 2), same two pinned
models, same greedy decoding. Conditions: clean, 20, 10, 0 dB SNR additive
white Gaussian noise from clinasr.noise (seeded per sample and condition,
so every model hears the identical noisy waveform). Every per-sample noise
outcome (status, achieved SNR, whether peak renormalization was needed) is
written to noise_log.csv. Scored with the declared default normalization
(v2), whose version is recorded in every summary row.
"""


import csv
import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


SNRS_DB: List[Optional[float]] = [None, 20.0, 10.0, 0.0]
NOISE_SEED = 20261029


def condition_name(snr_db: Optional[float]) -> str:
    return "clean" if snr_db is None else f"snr_{int(snr_db)}db"


def noisy_records(records: Sequence[Dict[str, Any]], snr_db: Optional[float], seed: int = NOISE_SEED):
    """Return (records with replaced audio arrays, per-sample noise log rows)."""
    out, log = [], []
    for record in records:
        audio = record["audio"]
        signal = np.asarray(audio["array"], dtype=np.float64)
        result = add_noise_at_snr(signal, snr_db, base_seed=seed, sample_id=record["file_name"])
        new = dict(record)
        new["audio"] = {"array": result.signal.astype(np.float32), "sampling_rate": audio["sampling_rate"]}
        out.append(new)
        log.append(
            {
                "sample_id": record["file_name"],
                "condition": condition_name(snr_db),
                "status": result.status,
                "requested_snr_db": result.requested_snr_db,
                "achieved_snr_db": result.achieved_snr_db,
                "was_clipped": result.was_clipped,
            }
        )
    return out, log


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_stage4(output_dir: Path) -> int:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bootstrap = bootstrap_all()
    print(f"bootstrap: {bootstrap}")
    if not all(bootstrap.values()):
        print("ERROR: package bootstrap failed; aborting before inference.")
        return 1

    handle = new_run(output_dir / "run", git_commit=get_git_commit())
    logger = handle.get_logger()

    selection = select(load_metadata())
    (output_dir / "selection.json").write_text(json.dumps(dataclasses.asdict(selection), indent=2), encoding="utf-8")
    if selection.fingerprint != STAGE2_EXPECTED_FINGERPRINT:
        print(f"ERROR: selection fingerprint {selection.fingerprint} != expected; aborting before inference.")
        return 2
    clean_records = load_selected_audio(selection.sample_ids)

    predictions, summaries, noise_log, timings = [], [], [], {}
    for snr in SNRS_DB:
        cond = condition_name(snr)
        records, log = noisy_records(clean_records, snr)
        noise_log.extend(log)
        for model_id, revision in MODELS.items():
            print(f"[{cond}] {model_id} ...")
            result = transcribe(records, model_id, revision)
            timings[f"{cond}/{model_id}"] = {k: result[k] for k in ("seconds", "device", "dtype")}
            rows = prediction_rows(records, result["hypotheses"], model_id)
            for row in rows:
                row["condition"] = cond
            predictions.extend(rows)
            for s in summary_rows(rows, model_id, DEFAULT_NORMALIZATION):
                s["condition"] = cond
                summaries.append(s)
            logger.info("%s %s done in %.1fs", cond, model_id, result["seconds"])

    _write_csv(output_dir / "predictions.csv", predictions)
    _write_csv(output_dir / "summary_metrics.csv", summaries)
    _write_csv(output_dir / "noise_log.csv", noise_log)

    config = ExperimentConfig(
        dataset_id=DATASET_ID,
        dataset_config=DATASET_CONFIG,
        dataset_split=DATASET_SPLIT,
        seed=STAGE2_SEED,
        sample_selection=SampleSelection(
            method="stratified_hash_ranked",
            size=len(selection.sample_ids),
            stratify_by=list(STRATA_KEYS),
            sample_ids=list(selection.sample_ids),
        ),
        models=[ModelSpec(model_id=m, resolved_revision=r) for m, r in MODELS.items()],
        normalization=DEFAULT_NORMALIZATION,
        noise=NoiseConfig(snrs_db=list(SNRS_DB), seed=NOISE_SEED),
        metrics=["wer", "cer"],
        software_versions=resolve_installed_versions(["jiwer", "numpy", "torch", "transformers", "datasets", "torchcodec"]),
        dataset_revision=DATASET_REVISION,
    )
    (output_dir / "experiment_config.json").write_text(config.to_json(), encoding="utf-8")
    statuses: Dict[str, int] = {}
    for row in noise_log:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    handle.write_manifest(
        {
            "stage": 4,
            "git_commit": get_git_commit(),
            "selection_fingerprint": selection.fingerprint,
            "noise_type": "additive white Gaussian",
            "noise_status_counts": statuses,
            "timings": timings,
        }
    )
    for s in summaries:
        if s["stratum"] == "ALL":
            print(f"{s['condition']:10s} {s['model']}: WER={s['wer']:.4f} CER={s['cer']:.4f}")
    return 0
# ---- end src/clinasr/stage4.py ----

# ---- begin src/clinasr/kaggle_entrypoint.py (sha256:afdba1f9f30f8925) ----
"""Entry point bundled into kaggle/run_clinasr.py by scripts/build_kaggle_script.py.

`--self-test` (no GPU, no dataset, no model): proves the packaging pipeline
(src/clinasr -> single self-contained script) carries real, tested logic
end-to-end -- environment reporting, the schema-audit engine against a tiny
inline fixture, and an ExperimentConfig skeleton with versions resolved
from whatever environment actually runs the script.

Default mode (no flags): runs the real Phase 1 five-sample smoke test
(clinasr.smoke_test.run_smoke_test) against the live EkaCare dataset with
openai/whisper-tiny.en. This needs network access and a Kaggle GPU runtime
with (or able to pip-install) torch/transformers/datasets/torchcodec; it is
not exercised by this repository's unit tests.
"""


import argparse
import json
import platform
import sys
from pathlib import Path


# Kaggle runs the script with no arguments, so the stage it executes is
# fixed here at build time. 4 = noise robustness on the Stage 2 subset.
DEFAULT_STAGE = 4


def _report_environment() -> dict:
    """Print AND return the environment report, so a caller with no `logs`
    command available (this project's kaggle CLI version has none) can
    persist it to a file instead of losing it once the console closes."""
    report = {
        "python": sys.version.splitlines()[0],
        "platform": platform.platform(),
        "torch": None,
        "cuda_available": None,
        "cuda_device": None,
    }
    print(f"python: {report['python']}")
    print(f"platform: {report['platform']}")
    try:
        import torch  # local import: optional, heavy dependency

        report["torch"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        print(f"torch: {report['torch']}")
        print(f"cuda_available: {report['cuda_available']}")
        if report["cuda_available"]:
            report["cuda_device"] = torch.cuda.get_device_name(0)
            print(f"cuda_device: {report['cuda_device']}")
    except ImportError:
        print("torch: not available in this environment")
    return report


def _self_test_schema_audit() -> None:
    fixture = [
        {"session_id": "s1", "text": "take 2.5 mg daily", "recording_context": "pharmacy"},
        {"session_id": "s2", "text": "follow up next week", "recording_context": None},
    ]
    report = audit_records(fixture)
    print(f"schema_audit: inspected {report.n_records_inspected} fixture records")
    for name, field_report in report.fields.items():
        print(
            f"  field={name!r} types={field_report.observed_types} "
            f"present={field_report.n_present} missing={field_report.n_missing}"
        )


def _self_test_config_skeleton() -> None:
    versions = resolve_installed_versions(["jiwer", "numpy", "torch", "transformers", "datasets"])
    skeleton = ExperimentConfig(
        dataset_id="ekacare/eka-medical-asr-evaluation-dataset",
        dataset_config="en",
        dataset_split="test",
        seed=0,
        sample_selection=SampleSelection(method="smoke_test_first_n", size=5),
        models=[ModelSpec(model_id="openai/whisper-tiny.en")],
        normalization=NormalizationConfig(),
        noise=NoiseConfig(),
        software_versions=versions,
    )
    print("experiment_config_skeleton:")
    print(skeleton.to_json())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ClinASR Kaggle entrypoint")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run packaging self-checks only; no dataset access, no model inference.",
    )
    parser.add_argument("--stage", type=int, choices=[1, 2, 4], default=DEFAULT_STAGE, help="1 = smoke test, 2 = two-model stratified run, 4 = noise robustness.")
    parser.add_argument(
        "--n",
        type=int,
        default=SMOKE_TEST_N,
        help=f"Number of smoke-test samples (default {SMOKE_TEST_N}).",
    )
    parser.add_argument("--model-id", default=MODEL_ID, help=f"Model to run (default {MODEL_ID}).")
    parser.add_argument(
        "--output-dir",
        default="/kaggle/working",
        help="Where to write predictions.csv/summary_metrics.csv/experiment_config.json.",
    )
    args = parser.parse_args(argv)

    env_report = _report_environment()

    if args.self_test:
        _self_test_schema_audit()
        _self_test_config_skeleton()
        print("self-test: OK (no inference performed)")
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "environment.json").write_text(json.dumps(env_report, indent=2), encoding="utf-8")

    if args.stage == 4:
        return run_stage4(output_dir)
    if args.stage == 2:
        return run_stage2(output_dir)
    return run_smoke_test(output_dir, n=args.n, model_id=args.model_id)


if __name__ == "__main__":
    sys.exit(main())
# ---- end src/clinasr/kaggle_entrypoint.py ----

