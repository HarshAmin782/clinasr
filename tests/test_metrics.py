from clinasr.metrics import compute_corpus_metric, compute_metrics, compute_sample_metric


def test_identical_text_has_zero_wer_cer():
    m = compute_sample_metric("s1", "the quick brown fox", "the quick brown fox")
    assert m.valid
    assert m.wer == 0.0
    assert m.cer == 0.0


def test_empty_reference_is_marked_invalid_not_crashed():
    m = compute_sample_metric("s1", "", "anything")
    assert not m.valid
    assert m.reason_invalid == "empty_reference"
    assert m.wer is None


def test_none_reference_is_marked_invalid():
    m = compute_sample_metric("s1", None, "anything")
    assert not m.valid


def test_missing_hypothesis_scored_as_fully_wrong_not_dropped():
    m = compute_sample_metric("s1", "the quick brown fox", None)
    assert m.valid
    assert m.wer == 1.0


def test_corpus_metric_excludes_invalid_samples_but_counts_them():
    samples = compute_metrics(
        ["s1", "s2"],
        ["hello world", ""],
        ["hello world", "ignored because empty ref"],
    )
    corpus = compute_corpus_metric(samples)
    assert corpus.n_samples_total == 2
    assert corpus.n_samples_valid == 1
    assert corpus.n_samples_excluded == 1
    assert corpus.wer == 0.0


def test_corpus_metric_with_no_valid_samples_returns_none():
    samples = compute_metrics(["s1"], [""], ["x"])
    corpus = compute_corpus_metric(samples)
    assert corpus.wer is None
    assert corpus.cer is None


def test_compute_metrics_requires_equal_length_inputs():
    import pytest

    with pytest.raises(ValueError):
        compute_metrics(["s1"], ["a", "b"], ["a"])


def test_module_imports_without_jiwer_installed(monkeypatch):
    """Importing clinasr.metrics must not require jiwer (docs/ISSUES.md #3):
    a bare Kaggle image without jiwer preinstalled would otherwise crash on
    script load, before --self-test even runs."""
    import builtins
    import importlib
    import sys

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "jiwer":
            raise ImportError("simulated: jiwer not installed")
        return real_import(name, *args, **kwargs)

    sys.modules.pop("jiwer", None)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    import clinasr.metrics as metrics_module

    importlib.reload(metrics_module)  # must not raise ImportError

