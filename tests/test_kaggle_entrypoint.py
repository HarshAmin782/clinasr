from clinasr.kaggle_entrypoint import _report_environment


def test_report_environment_returns_a_dict_with_expected_keys():
    report = _report_environment()
    assert set(report.keys()) == {"python", "platform", "torch", "cuda_available", "cuda_device"}
    assert isinstance(report["python"], str) and report["python"]
    assert isinstance(report["platform"], str) and report["platform"]
    # On this machine torch is not installed, so these must be None, not
    # crash and not silently omitted (the real Kaggle run needs this file
    # to actually report GPU allocation, since there's no `logs` command).
    assert report["torch"] is None
    assert report["cuda_available"] is None
