"""Dataset schema auditing, run BEFORE any stratified sampling.

PROJECT_SPEC.md lists `audio`, `text`, `session_id`, `speaker`,
`recording_context`, `type_concept`, `medical_entities` as "expected
fields... inspect schema/annotation representation before use" — i.e. the
spec itself says not to assume they exist. `audit_records()` below is the
generic, dependency-free core: it works on any list of dict-like records,
so it is fully unit-testable without a network call or the `datasets`
package installed. `audit_hf_dataset()` is the thin, real-dataset wrapper
that Phase 1 runs once (locally or on Kaggle) against a small streamed
sample of ekacare/eka-medical-asr-evaluation-dataset before sampling the
200-400 item benchmark subset; it imports `datasets` lazily so that merely
importing this module never requires that dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class FieldReport:
    name: str
    observed_types: List[str]
    n_missing: int
    n_present: int
    example_value_repr: Optional[str]


@dataclass
class SchemaAuditReport:
    n_records_inspected: int
    fields: Dict[str, FieldReport] = field(default_factory=dict)

    def missing_expected(self, expected_fields: List[str]) -> List[str]:
        return [f for f in expected_fields if f not in self.fields]


def _type_name(value: Any) -> str:
    return type(value).__name__


def audit_records(records: Iterable[Dict[str, Any]], example_repr_max_len: int = 120) -> SchemaAuditReport:
    """Inspect field names, observed Python types, and missing-value counts.

    A field is "missing" for a record if the key is absent OR its value is
    None. Every record's full key set contributes to `fields`, so a field
    that only appears on some records (schema drift) is still reported with
    an accurate n_present/n_missing split rather than silently ignored.
    """
    records = list(records)
    field_names: List[str] = []
    for record in records:
        for key in record.keys():
            if key not in field_names:
                field_names.append(key)

    reports: Dict[str, FieldReport] = {}
    for name in field_names:
        observed_types: List[str] = []
        n_missing = 0
        n_present = 0
        example_repr: Optional[str] = None
        for record in records:
            value = record.get(name, None)
            if value is None:
                n_missing += 1
                continue
            n_present += 1
            type_name = _type_name(value)
            if type_name not in observed_types:
                observed_types.append(type_name)
            if example_repr is None:
                text = repr(value)
                example_repr = text if len(text) <= example_repr_max_len else text[:example_repr_max_len] + "..."
        reports[name] = FieldReport(
            name=name,
            observed_types=observed_types,
            n_missing=n_missing,
            n_present=n_present,
            example_value_repr=example_repr,
        )

    return SchemaAuditReport(n_records_inspected=len(records), fields=reports)


def audit_hf_dataset(
    dataset_id: str,
    config_name: str,
    split: str,
    n: int = 5,
) -> SchemaAuditReport:
    """Stream the first `n` records of a Hugging Face dataset and audit them.

    Requires network access and the `datasets` package; neither is assumed
    to be available when this module is merely imported for unit testing.
    Not called automatically anywhere in this repository yet — Phase 1 runs
    it explicitly as a documented, opt-in step.
    """
    import datasets  # local import: optional, heavy dependency

    stream = datasets.load_dataset(dataset_id, config_name, split=split, streaming=True)
    records = []
    for i, record in enumerate(stream):
        if i >= n:
            break
        records.append(dict(record))
    return audit_records(records)
