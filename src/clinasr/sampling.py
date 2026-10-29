"""Deterministic stratified sample selection for Stage 2.

Selection must come out identical wherever it runs (local Python 3.9, the
Kaggle image's 3.13), so it uses no RNG at all: within a stratum, records
are ordered by sha256(f"{seed}:{sample_id}") and the first k are taken.
That ranking depends only on the seed and the ids, not on stream order,
Python version, or platform. `selection_fingerprint()` hashes the sorted
selected ids so a remote run can prove it evaluated exactly the set that
was reviewed locally.

Allocation: every stratum first gets min(stratum_size, min_per_stratum),
so small strata are represented; the remainder is distributed in
proportion to each stratum's remaining capacity (largest-remainder method,
ties broken by stratum key) so large strata are not over-represented
relative to the population more than necessary. Selection is fixed BEFORE
any model is run -- never re-drawn after looking at results.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from clinasr.medical_entities import assert_unique_ids

StratumKey = Tuple[str, ...]


def stable_rank(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def selection_fingerprint(sample_ids: Iterable[str]) -> str:
    joined = "\n".join(sorted(sample_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def allocate(stratum_sizes: Dict[StratumKey, int], total: int, min_per_stratum: int) -> Dict[StratumKey, int]:
    """Return how many samples to draw from each stratum (sums to `total`)."""
    population = sum(stratum_sizes.values())
    if total <= 0:
        raise ValueError("total must be positive")
    if total > population:
        raise ValueError(f"requested {total} samples but only {population} eligible records exist")

    keys = sorted(stratum_sizes, key=lambda k: tuple(str(x) for x in k))
    base = {k: min(stratum_sizes[k], min_per_stratum) for k in keys}
    remaining = total - sum(base.values())
    if remaining < 0:
        raise ValueError(
            f"min_per_stratum={min_per_stratum} across {len(keys)} strata needs "
            f"{sum(base.values())} samples, more than total={total}"
        )

    capacity = {k: stratum_sizes[k] - base[k] for k in keys}
    total_capacity = sum(capacity.values())
    extra = {k: 0 for k in keys}
    if remaining and total_capacity:
        quotas = {k: remaining * capacity[k] / total_capacity for k in keys}
        extra = {k: int(quotas[k]) for k in keys}
        leftover = remaining - sum(extra.values())
        by_remainder = sorted(keys, key=lambda k: (-(quotas[k] - extra[k]), tuple(str(x) for x in k)))
        for k in by_remainder:
            if leftover == 0:
                break
            if extra[k] < capacity[k]:
                extra[k] += 1
                leftover -= 1
    result = {k: base[k] + extra[k] for k in keys}
    if sum(result.values()) != total:
        raise AssertionError(f"allocation sums to {sum(result.values())}, expected {total}")
    return result


@dataclass
class Selection:
    sample_ids: List[str]
    seed: int
    total_requested: int
    min_per_stratum: int
    strata_keys: List[str]
    allocation: Dict[str, int] = field(default_factory=dict)
    stratum_population: Dict[str, int] = field(default_factory=dict)
    n_records_seen: int = 0
    n_eligible: int = 0
    n_excluded: int = 0
    exclusion_reasons: Dict[str, int] = field(default_factory=dict)
    fingerprint: str = ""


def _label(key: StratumKey) -> str:
    return " | ".join(str(x) for x in key)


def stratified_select(
    records: Iterable[Dict[str, Any]],
    strata_keys: Sequence[str],
    total: int,
    seed: int,
    min_per_stratum: int,
    id_field: str = "file_name",
    exclude: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
) -> Selection:
    """Select `total` record ids, stratified by `strata_keys`.

    `exclude(record)` returns a reason string to drop a record from the
    eligible pool (e.g. too long for the decoding setup) or None to keep it.
    Raises ValueError on missing/duplicate ids.
    """
    records = list(records)
    assert_unique_ids([r.get(id_field) for r in records], id_field)

    reasons: Dict[str, int] = {}
    groups: Dict[StratumKey, List[str]] = {}
    for record in records:
        reason = exclude(record) if exclude else None
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        key = tuple(str(record.get(k)) for k in strata_keys)
        groups.setdefault(key, []).append(str(record[id_field]))

    sizes = {k: len(v) for k, v in groups.items()}
    allocation = allocate(sizes, total, min_per_stratum)

    selected: List[str] = []
    for key, k in allocation.items():
        ranked = sorted(groups[key], key=lambda sid: stable_rank(seed, sid))
        selected.extend(ranked[:k])
    selected.sort()

    return Selection(
        sample_ids=selected,
        seed=seed,
        total_requested=total,
        min_per_stratum=min_per_stratum,
        strata_keys=list(strata_keys),
        allocation={_label(k): v for k, v in allocation.items()},
        stratum_population={_label(k): sizes[k] for k in allocation},
        n_records_seen=len(records),
        n_eligible=sum(sizes.values()),
        n_excluded=sum(reasons.values()),
        exclusion_reasons=reasons,
        fingerprint=selection_fingerprint(selected),
    )
