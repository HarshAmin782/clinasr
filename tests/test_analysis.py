import pytest

from clinasr.analysis import paired_bootstrap, word_edit_counts


def _rows(pairs_a, pairs_b, a="tiny", b="base"):
    rows = []
    for i, (ref, hyp) in enumerate(pairs_a):
        rows.append({"sample_id": f"s{i}", "model": a, "reference": ref, "hypothesis": hyp})
    for i, (ref, hyp) in enumerate(pairs_b):
        rows.append({"sample_id": f"s{i}", "model": b, "reference": ref, "hypothesis": hyp})
    return rows


def test_word_edit_counts():
    assert word_edit_counts("take two tablets", "take two tablets") == (0, 3)
    assert word_edit_counts("take two tablets", "take tablets") == (1, 3)
    assert word_edit_counts("", "anything") == (0, 0)
    assert word_edit_counts("a b", None) == (2, 2)


def test_identical_models_give_zero_difference():
    pairs = [("take two tablets", "take two tablet"), ("fever since monday", "fever since monday")] * 10
    result = paired_bootstrap(_rows(pairs, pairs), "tiny", "base", n_bootstrap=200)
    assert result.difference == 0.0
    assert result.ci_low == result.ci_high == 0.0
    assert result.n_tied == 20


def test_clearly_better_model_ci_excludes_zero_and_is_reproducible():
    refs = [f"patient has fever and cough number {i}" for i in range(40)]
    worse = [(r, "patient has") for r in refs]
    better = [(r, r) for r in refs]
    r1 = paired_bootstrap(_rows(worse, better), "tiny", "base", n_bootstrap=500, seed=3)
    r2 = paired_bootstrap(_rows(worse, better), "tiny", "base", n_bootstrap=500, seed=3)
    assert r1 == r2
    assert r1.wer_b == 0.0 and r1.wer_a > 0.5
    assert r1.ci_high < 0
    assert r1.n_b_better == 40


def test_mismatched_sample_sets_fail():
    rows = _rows([("a", "a"), ("b", "b")], [("a", "a")])
    with pytest.raises(ValueError, match="same samples"):
        paired_bootstrap(rows, "tiny", "base")


def test_duplicate_sample_fails():
    rows = _rows([("a", "a")], [("a", "a")])
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicate"):
        paired_bootstrap(rows, "tiny", "base")
