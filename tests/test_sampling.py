import pytest

from clinasr.sampling import allocate, selection_fingerprint, stable_rank, stratified_select


def _records(spec):
    """spec: {(ctx, concept): n} -> index-like records with unique file_names."""
    records = []
    for (ctx, concept), n in spec.items():
        for i in range(n):
            records.append(
                {"file_name": f"/{ctx}/{concept}/{i}.m4a", "recording_context": ctx, "type_concept": concept, "duration": 5.0}
            )
    return records


SPEC = {("conversation", "drugs"): 500, ("conversation", "misc_medical"): 300, ("dictation", "drugs"): 40, ("dictation", "advices"): 5}


def test_stable_rank_is_deterministic_and_seed_dependent():
    assert stable_rank(0, "a") == stable_rank(0, "a")
    assert stable_rank(0, "a") != stable_rank(1, "a")


def test_allocate_sums_to_total_and_respects_minimum_and_capacity():
    sizes = {k: v for k, v in SPEC.items()}
    alloc = allocate(sizes, total=300, min_per_stratum=20)
    assert sum(alloc.values()) == 300
    assert alloc[("dictation", "advices")] == 5  # smaller than the minimum: take all
    assert alloc[("dictation", "drugs")] >= 20
    for k, v in alloc.items():
        assert v <= sizes[k]
    # Larger strata get more.
    assert alloc[("conversation", "drugs")] > alloc[("conversation", "misc_medical")]


def test_allocate_rejects_impossible_requests():
    with pytest.raises(ValueError, match="only"):
        allocate({("a",): 3}, total=5, min_per_stratum=1)
    with pytest.raises(ValueError, match="min_per_stratum"):
        allocate({("a",): 50, ("b",): 50, ("c",): 50}, total=20, min_per_stratum=10)
    with pytest.raises(ValueError):
        allocate({("a",): 3}, total=0, min_per_stratum=1)


def test_allocate_exact_population():
    alloc = allocate({("a",): 3, ("b",): 2}, total=5, min_per_stratum=1)
    assert alloc == {("a",): 3, ("b",): 2}


def test_selection_is_deterministic_and_order_independent():
    records = _records(SPEC)
    a = stratified_select(records, ["recording_context", "type_concept"], 300, seed=7, min_per_stratum=20)
    b = stratified_select(list(reversed(records)), ["recording_context", "type_concept"], 300, seed=7, min_per_stratum=20)
    assert a.sample_ids == b.sample_ids
    assert a.fingerprint == b.fingerprint == selection_fingerprint(a.sample_ids)
    assert len(a.sample_ids) == len(set(a.sample_ids)) == 300


def test_different_seed_changes_selection():
    records = _records(SPEC)
    a = stratified_select(records, ["recording_context", "type_concept"], 300, seed=1, min_per_stratum=20)
    b = stratified_select(records, ["recording_context", "type_concept"], 300, seed=2, min_per_stratum=20)
    assert a.fingerprint != b.fingerprint


def test_selection_reports_allocation_and_population():
    records = _records(SPEC)
    sel = stratified_select(records, ["recording_context", "type_concept"], 300, seed=7, min_per_stratum=20)
    assert sel.stratum_population["dictation | advices"] == 5
    assert sel.allocation["dictation | advices"] == 5
    assert sum(sel.allocation.values()) == 300
    picked_advices = [s for s in sel.sample_ids if "/dictation/advices/" in s]
    assert len(picked_advices) == 5


def test_exclusions_are_counted_and_not_selected():
    records = _records({("c", "x"): 10})
    records[0]["duration"] = 45.0
    records[1]["duration"] = 50.0

    def too_long(r):
        return "duration_over_30s" if r["duration"] > 30 else None

    sel = stratified_select(records, ["recording_context", "type_concept"], 8, seed=0, min_per_stratum=1, exclude=too_long)
    assert sel.n_excluded == 2
    assert sel.exclusion_reasons == {"duration_over_30s": 2}
    assert records[0]["file_name"] not in sel.sample_ids
    assert len(sel.sample_ids) == 8


def test_duplicate_ids_fail_explicitly():
    records = _records({("c", "x"): 3})
    records.append(dict(records[0]))
    with pytest.raises(ValueError, match="duplicate"):
        stratified_select(records, ["recording_context"], 2, seed=0, min_per_stratum=1)
