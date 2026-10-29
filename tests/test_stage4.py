import numpy as np

from clinasr.stage4 import SNRS_DB, condition_name, noisy_records


def _records():
    t = np.arange(16000) / 16000.0
    return [
        {"file_name": "/a/1.wav", "audio": {"array": (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), "sampling_rate": 16000}},
        {"file_name": "/a/2.wav", "audio": {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000}},
    ]


def test_conditions():
    assert [condition_name(s) for s in SNRS_DB] == ["clean", "snr_20db", "snr_10db", "snr_0db"]


def test_clean_condition_leaves_audio_unchanged():
    recs = _records()
    out, log = noisy_records(recs, None)
    assert np.array_equal(out[0]["audio"]["array"], recs[0]["audio"]["array"])
    assert [r["status"] for r in log] == ["clean", "clean"]


def test_noisy_condition_is_deterministic_and_logs_status():
    a, log = noisy_records(_records(), 10.0)
    b, _ = noisy_records(_records(), 10.0)
    assert np.array_equal(a[0]["audio"]["array"], b[0]["audio"]["array"])
    assert a[0]["audio"]["array"].dtype == np.float32
    assert log[0]["status"] == "ok" and abs(log[0]["achieved_snr_db"] - 10.0) < 0.5
    assert log[1]["status"] == "skipped_silent"  # silent clip left unchanged, logged


def test_original_records_not_mutated():
    recs = _records()
    before = recs[0]["audio"]["array"].copy()
    noisy_records(recs, 0.0)
    assert np.array_equal(recs[0]["audio"]["array"], before)
