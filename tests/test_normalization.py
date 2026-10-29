from clinasr.normalization import NormalizationConfig, normalize_text


def test_lowercases_by_default():
    assert normalize_text("Hello WORLD") == "hello world"


def test_strips_punctuation_not_between_digits():
    assert normalize_text("Hello, world!") == "hello world"


def test_preserves_decimal_dosage():
    assert normalize_text("Take 2.5 mg now.") == "take 2.5 mg now"


def test_preserves_dosage_schedule_hyphens():
    assert normalize_text("Dose is 1-0-1 tablets.") == "dose is 1-0-1 tablets"


def test_preserves_fraction_slash():
    assert normalize_text("Give 1/2 tablet.") == "give 1/2 tablet"


def test_does_not_preserve_leading_or_trailing_punct_on_numbers():
    # A trailing period after a number that is NOT between two digits is stripped.
    assert normalize_text("Take 5mg.") == "take 5 mg"  # v2 splits number+unit


def test_none_input_returns_empty_string():
    assert normalize_text(None) == ""


def test_collapses_whitespace():
    assert normalize_text("too   many    spaces") == "too many spaces"


def test_can_disable_lowercase():
    config = NormalizationConfig(lowercase=False)
    assert normalize_text("Hello", config) == "Hello"


def test_can_disable_inter_digit_preservation():
    config = NormalizationConfig(preserve_inter_digit_punctuation=False)
    assert normalize_text("2.5 mg", config) == "2 5 mg"


def test_v2_splits_number_from_attached_unit():
    assert normalize_text("Zentel 400mg Oral") == "zentel 400 mg oral"
    assert normalize_text("25MCG") == "25 mcg"
    assert normalize_text("2.5ml") == "2.5 ml"
    assert normalize_text("10units") == "10 units"
    assert normalize_text("400 mg") == "400 mg"


def test_v2_does_not_split_unknown_suffixes():
    assert normalize_text("200mgdl") == "200mgdl"
    assert normalize_text("covid19") == "covid19"


def test_v1_keeps_attached_units():
    from clinasr.normalization import NORMALIZATION_V1

    assert normalize_text("400mg", NORMALIZATION_V1) == "400mg"
    assert NORMALIZATION_V1.version == "v1"


# ---- v3 hyphen rule -------------------------------------------------------
import pytest  # noqa: E402

from clinasr.normalization import NORMALIZATION_V2, NORMALIZATION_V3  # noqa: E402


@pytest.mark.parametrize(
    "text,expected",
    [
        ("statin-induced myopathy", "statin induced myopathy"),
        ("CT-scan of the thorax", "ct scan of the thorax"),
        ("non-tender abdomen", "non tender abdomen"),
        ("X-ray chest", "x ray chest"),
        ("anti-HBs positive", "anti hbs positive"),
        ("COVID-19", "covid 19"),
        ("beta-blocker", "beta blocker"),
        # ranges and schedules keep their hyphen
        ("sleep 6-8 hours", "sleep 6-8 hours"),
        ("dose 1-0-1", "dose 1-0-1"),
        # negative values keep their minus sign
        ("temperature -5 degrees", "temperature -5 degrees"),
        ("-2.5 SD", "-2.5 sd"),
        ("(-3)", "-3"),
        # unicode dashes are mapped first
        ("6\u20138 hours", "6-8 hours"),
        ("statin\u2010induced", "statin induced"),
        ("value \u22124", "value -4"),
    ],
)
def test_v3_hyphen_rule(text, expected):
    assert normalize_text(text, NORMALIZATION_V3) == expected


def test_v2_unchanged_by_v3_addition():
    assert NORMALIZATION_V2.version == "v2" and NORMALIZATION_V2.split_hyphens is False
    assert normalize_text("statin-induced myopathy", NORMALIZATION_V2) == "statin-induced myopathy"
    assert normalize_text("Zentel 400mg", NORMALIZATION_V2) == "zentel 400 mg"


def test_v3_hyphen_word_break_does_not_merge_tokens():
    # The rule only ever ADDS a word break; "xray" is not equated with "x ray".
    assert normalize_text("xray", NORMALIZATION_V3) == "xray"
