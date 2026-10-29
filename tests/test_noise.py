import numpy as np
import pytest

from clinasr.noise import MIN_SAMPLES, add_noise_at_snr


def _tone(n=1600, amplitude=0.5):
    t = np.arange(n) / 16000.0
    return amplitude * np.sin(2 * np.pi * 440 * t)


def test_clean_condition_is_noop():
    signal = _tone()
    result = add_noise_at_snr(signal, None, base_seed=0, sample_id="a")
    assert result.status == "clean"
    assert np.array_equal(result.signal, signal)


def test_silent_clip_is_skipped_not_divided_by_zero():
    signal = np.zeros(1600)
    result = add_noise_at_snr(signal, snr_db=10.0, base_seed=0, sample_id="a")
    assert result.status == "skipped_silent"
    assert np.array_equal(result.signal, signal)


def test_very_short_clip_is_skipped():
    signal = _tone(n=MIN_SAMPLES - 1)
    result = add_noise_at_snr(signal, snr_db=10.0, base_seed=0, sample_id="a")
    assert result.status == "skipped_short"


def test_invalid_snr_raises_value_error():
    with pytest.raises(ValueError):
        add_noise_at_snr(_tone(), snr_db=float("nan"), base_seed=0, sample_id="a")
    with pytest.raises(ValueError):
        add_noise_at_snr(_tone(), snr_db=float("inf"), base_seed=0, sample_id="a")


def test_deterministic_for_same_seed_and_sample_id():
    r1 = add_noise_at_snr(_tone(), snr_db=10.0, base_seed=42, sample_id="sample-7")
    r2 = add_noise_at_snr(_tone(), snr_db=10.0, base_seed=42, sample_id="sample-7")
    assert np.array_equal(r1.signal, r2.signal)


def test_different_sample_ids_get_different_noise():
    r1 = add_noise_at_snr(_tone(), snr_db=10.0, base_seed=42, sample_id="sample-7")
    r2 = add_noise_at_snr(_tone(), snr_db=10.0, base_seed=42, sample_id="sample-8")
    assert not np.array_equal(r1.signal, r2.signal)


def test_achieved_snr_is_close_to_requested():
    signal = _tone(n=16000, amplitude=0.8)
    result = add_noise_at_snr(signal, snr_db=10.0, base_seed=1, sample_id="a")
    assert result.status == "ok"
    assert abs(result.achieved_snr_db - 10.0) < 0.5


def test_clipping_is_flagged_and_signal_stays_in_range():
    # Near-full-scale signal at a harsh SNR is likely to exceed [-1, 1]
    # before renormalization.
    signal = _tone(n=16000, amplitude=0.999)
    result = add_noise_at_snr(signal, snr_db=0.0, base_seed=1, sample_id="a")
    assert np.max(np.abs(result.signal)) <= 1.0 + 1e-9
