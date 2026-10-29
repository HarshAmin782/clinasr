"""Shared, CPU-only, dependency-light ClinASR logic.

This package is the single source of truth for normalization, metrics,
entity matching, noise augmentation, experiment configuration, run
manifests, and schema auditing. scripts/build_kaggle_script.py bundles
these modules into kaggle/run_clinasr.py for upload as a Kaggle code_file;
nothing here should assume a GPU or import torch/transformers at module
level.
"""

__all__ = [
    "normalization",
    "metrics",
    "entities",
    "medical_entities",
    "noise",
    "config",
    "run_manifest",
    "schema_audit",
    "smoke_test",
    "sampling",
    "stage2",
    "stage4",
    "kaggle_entrypoint",
]
