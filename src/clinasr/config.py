"""Experiment configuration schema.

One `ExperimentConfig` is created per run and written into that run's
manifest (see run_manifest.py). It is the single place that records what
was actually fixed for a run: seeds, sample selection, model identities and
their RESOLVED revisions, normalization policy, noise parameters, and which
metrics were computed. Package/model versions are never hardcoded here —
`resolve_installed_versions()` reads them from the actual environment at
run time via `importlib.metadata`, per the research-integrity rule against
inventing version/revision numbers.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from clinasr.normalization import NormalizationConfig


@dataclass
class ModelSpec:
    model_id: str
    # Populated at run time from the loaded model/tokenizer, e.g. the
    # Hugging Face Hub commit SHA actually resolved for `model_id`. Left
    # None until a real load happens; never guessed ahead of time.
    resolved_revision: Optional[str] = None


@dataclass
class SampleSelection:
    method: str  # e.g. "smoke_test_first_n" | "stratified_fixed_subset"
    size: int
    stratify_by: List[str] = field(default_factory=list)
    sample_ids: List[str] = field(default_factory=list)


@dataclass
class NoiseConfig:
    snrs_db: List[Optional[float]] = field(default_factory=lambda: [None, 20.0, 10.0, 0.0])
    seed: int = 0


@dataclass
class ExperimentConfig:
    dataset_id: str
    dataset_config: str
    dataset_split: str
    seed: int
    sample_selection: SampleSelection
    models: List[ModelSpec]
    normalization: NormalizationConfig
    noise: NoiseConfig
    metrics: List[str] = field(default_factory=lambda: ["wer", "cer"])
    # Filled via resolve_installed_versions(); not populated at construction.
    software_versions: Dict[str, str] = field(default_factory=dict)
    dataset_revision: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        data = dict(data)
        data["sample_selection"] = SampleSelection(**data["sample_selection"])
        data["models"] = [ModelSpec(**m) for m in data["models"]]
        data["normalization"] = NormalizationConfig(**data["normalization"])
        data["noise"] = NoiseConfig(**data["noise"])
        return cls(**data)


def resolve_installed_versions(package_names: List[str]) -> Dict[str, str]:
    """Return actually-installed versions for the given packages.

    Packages that are not installed in the current environment are recorded
    as "not-installed" rather than omitted, so a report never implies a
    version that was never actually checked.
    """
    versions: Dict[str, str] = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions
