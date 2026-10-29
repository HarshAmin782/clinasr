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
    assert normalize_text("Take 5mg.") == "take 5mg"


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
