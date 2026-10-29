from clinasr.config import (
    ExperimentConfig,
    ModelSpec,
    NoiseConfig,
    SampleSelection,
    resolve_installed_versions,
)
from clinasr.normalization import NormalizationConfig


def _make_config():
    return ExperimentConfig(
        dataset_id="ekacare/eka-medical-asr-evaluation-dataset",
        dataset_config="en",
        dataset_split="test",
        seed=42,
        sample_selection=SampleSelection(method="smoke_test_first_n", size=5),
        models=[ModelSpec(model_id="openai/whisper-tiny.en")],
        normalization=NormalizationConfig(),
        noise=NoiseConfig(),
    )


def test_round_trips_through_json():
    config = _make_config()
    restored = ExperimentConfig.from_dict(__import__("json").loads(config.to_json()))
    assert restored == config


def test_resolve_installed_versions_marks_missing_packages():
    versions = resolve_installed_versions(["pytest", "a-package-that-does-not-exist-xyz"])
    assert versions["pytest"] != "not-installed"
    assert versions["a-package-that-does-not-exist-xyz"] == "not-installed"


def test_model_spec_revision_defaults_to_unresolved():
    spec = ModelSpec(model_id="openai/whisper-tiny.en")
    assert spec.resolved_revision is None
