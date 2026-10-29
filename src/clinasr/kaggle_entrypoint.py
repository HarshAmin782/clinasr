"""Entry point bundled into kaggle/run_clinasr.py by scripts/build_kaggle_script.py.

`--self-test` (no GPU, no dataset, no model): proves the packaging pipeline
(src/clinasr -> single self-contained script) carries real, tested logic
end-to-end -- environment reporting, the schema-audit engine against a tiny
inline fixture, and an ExperimentConfig skeleton with versions resolved
from whatever environment actually runs the script.

Default mode (no flags): runs the real Phase 1 five-sample smoke test
(clinasr.smoke_test.run_smoke_test) against the live EkaCare dataset with
openai/whisper-tiny.en. This needs network access and a Kaggle GPU runtime
with (or able to pip-install) torch/transformers/datasets/torchcodec; it is
not exercised by this repository's unit tests.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from clinasr.config import ExperimentConfig, ModelSpec, NoiseConfig, SampleSelection, resolve_installed_versions
from clinasr.normalization import NormalizationConfig
from clinasr.schema_audit import audit_records
from clinasr.smoke_test import MODEL_ID, SMOKE_TEST_N, run_smoke_test


def _report_environment() -> dict:
    """Print AND return the environment report, so a caller with no `logs`
    command available (this project's kaggle CLI version has none) can
    persist it to a file instead of losing it once the console closes."""
    report = {
        "python": sys.version.splitlines()[0],
        "platform": platform.platform(),
        "torch": None,
        "cuda_available": None,
        "cuda_device": None,
    }
    print(f"python: {report['python']}")
    print(f"platform: {report['platform']}")
    try:
        import torch  # local import: optional, heavy dependency

        report["torch"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        print(f"torch: {report['torch']}")
        print(f"cuda_available: {report['cuda_available']}")
        if report["cuda_available"]:
            report["cuda_device"] = torch.cuda.get_device_name(0)
            print(f"cuda_device: {report['cuda_device']}")
    except ImportError:
        print("torch: not available in this environment")
    return report


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
    parser = argparse.ArgumentParser(description="ClinASR Kaggle entrypoint")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run packaging self-checks only; no dataset access, no model inference.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=SMOKE_TEST_N,
        help=f"Number of smoke-test samples (default {SMOKE_TEST_N}).",
    )
    parser.add_argument("--model-id", default=MODEL_ID, help=f"Model to run (default {MODEL_ID}).")
    parser.add_argument(
        "--output-dir",
        default="/kaggle/working",
        help="Where to write predictions.csv/summary_metrics.csv/experiment_config.json.",
    )
    args = parser.parse_args(argv)

    env_report = _report_environment()

    if args.self_test:
        _self_test_schema_audit()
        _self_test_config_skeleton()
        print("self-test: OK (no inference performed)")
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "environment.json").write_text(json.dumps(env_report, indent=2), encoding="utf-8")

    return run_smoke_test(output_dir, n=args.n, model_id=args.model_id)


if __name__ == "__main__":
    sys.exit(main())
