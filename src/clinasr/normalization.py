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

from __future__ import annotations

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
    version: str = "v2"


DEFAULT_NORMALIZATION = NormalizationConfig()
# The policy Stage 2 (run 20261029T162830Z) was scored with, kept so those
# numbers stay reproducible.
NORMALIZATION_V1 = NormalizationConfig(split_number_unit=False, version="v1")


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
    if config.split_number_unit:
        text = _NUMBER_UNIT_PATTERN.sub(r"\1 \2", text)
    if config.collapse_whitespace:
        text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text
