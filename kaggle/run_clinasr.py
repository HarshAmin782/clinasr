"""GENERATED FILE -- DO NOT EDIT DIRECTLY.

Built by scripts/build_kaggle_script.py from src/clinasr/*.py.
Source commit: aa34eda4b915
To change behavior, edit the source modules under src/clinasr/ and
rebuild. This file is what Kaggle CLI uploads as the single code_file.
"""

from __future__ import annotations

# ---- begin src/clinasr/normalization.py (sha256:bf05bbf060efe009) ----
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


DEFAULT_NORMALIZATION = NormalizationConfig()


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
    if config.strip_punctuation:
        text = _strip_punctuation(text, config.preserve_inter_digit_punctuation)
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

# ---- begin src/clinasr/entities.py (sha256:b4a61fa55072132a) ----
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


from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence



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
# ---- end src/clinasr/entities.py ----

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

# ---- begin src/clinasr/kaggle_entrypoint.py (sha256:c4a71ffa8c057f6a) ----
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

    return run_smoke_test(output_dir, n=args.n, model_id=args.model_id)


if __name__ == "__main__":
    sys.exit(main())
# ---- end src/clinasr/kaggle_entrypoint.py ----

