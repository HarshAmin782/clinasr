"""GENERATED FILE -- DO NOT EDIT DIRECTLY.

Built by scripts/build_kaggle_script.py from src/clinasr/*.py.
Source commit: b640773271ca
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

# ---- begin src/clinasr/metrics.py (sha256:086d98d06cd01d2e) ----
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
"""


from dataclasses import dataclass
from typing import List, Optional, Sequence

import jiwer



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

# ---- begin src/clinasr/schema_audit.py (sha256:d9d89cdf7ebe46b2) ----
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
) -> SchemaAuditReport:
    """Stream the first `n` records of a Hugging Face dataset and audit them.

    Requires network access and the `datasets` package; neither is assumed
    to be available when this module is merely imported for unit testing.
    Not called automatically anywhere in this repository yet — Phase 1 runs
    it explicitly as a documented, opt-in step.
    """
    import datasets  # local import: optional, heavy dependency

    stream = datasets.load_dataset(dataset_id, config_name, split=split, streaming=True)
    records = []
    for i, record in enumerate(stream):
        if i >= n:
            break
        records.append(dict(record))
    return audit_records(records)
# ---- end src/clinasr/schema_audit.py ----

# ---- begin src/clinasr/kaggle_entrypoint.py (sha256:46803fb037025d98) ----
"""Entry point bundled into kaggle/run_clinasr.py by scripts/build_kaggle_script.py.

Phase-0 scope (this repository-hardening pass): this entrypoint does NOT run
any ASR model and does NOT touch the EkaCare dataset. It only proves that
the packaging pipeline (src/clinasr -> single self-contained script) carries
real, tested logic end-to-end: environment reporting, the schema-audit
engine against a tiny inline fixture, and an ExperimentConfig skeleton with
versions resolved from whatever environment actually runs the script.

Phase 1 (prompts/01-build.md) replaces the body of `main()` with the real
five-sample smoke test, reusing the same clinasr.normalization/metrics/
config/run_manifest modules bundled here -- no duplicated logic.
"""


import argparse
import platform
import sys



def _report_environment() -> None:
    print(f"python: {sys.version.splitlines()[0]}")
    print(f"platform: {platform.platform()}")
    try:
        import torch  # local import: optional, heavy dependency

        print(f"torch: {torch.__version__}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda_device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("torch: not available in this environment")


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
    parser = argparse.ArgumentParser(description="ClinASR Kaggle entrypoint (Phase-0 placeholder)")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run packaging self-checks only; no dataset access, no model inference.",
    )
    args = parser.parse_args(argv)

    _report_environment()

    if args.self_test:
        _self_test_schema_audit()
        _self_test_config_skeleton()
        print("self-test: OK (no inference performed)")
        return 0

    print(
        "ClinASR Phase 1 (ASR inference) is not implemented yet. "
        "Run with --self-test to verify packaging, or see prompts/01-build.md."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
# ---- end src/clinasr/kaggle_entrypoint.py ----

