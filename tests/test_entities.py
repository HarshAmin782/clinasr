from clinasr.entities import EntitySpan, match_entities, match_entity, summarize


def test_exact_match():
    r = match_entity(EntitySpan(text="metformin"), "take metformin twice daily")
    assert r.status == "exact"


def test_normalized_match_on_case_difference():
    r = match_entity(EntitySpan(text="Metformin"), "take metformin twice daily")
    assert r.status == "normalized"


def test_omitted_when_absent():
    r = match_entity(EntitySpan(text="metformin"), "take the pill twice daily")
    assert r.status == "omitted"


def test_does_not_match_via_optimistic_substring_overlap():
    # "50" must not be considered present just because "150mg" contains the
    # characters "50" -- it is not a token-boundary match.
    r = match_entity(EntitySpan(text="50"), "take 150mg now")
    assert r.status == "omitted"


def test_does_match_when_number_is_its_own_token():
    r = match_entity(EntitySpan(text="50"), "take 50 mg now")
    assert r.status == "exact"


def test_multiword_entity_requires_contiguous_tokens():
    r = match_entity(EntitySpan(text="chest pain"), "patient reports pain in chest area")
    assert r.status == "omitted"
    r2 = match_entity(EntitySpan(text="chest pain"), "patient reports chest pain today")
    assert r2.status == "exact"


def test_summarize_counts_and_recall():
    results = match_entities(
        [EntitySpan(text="metformin"), EntitySpan(text="insulin")],
        "take metformin daily",
    )
    summary = summarize(results)
    assert summary["n_entities"] == 2
    assert summary["n_exact"] == 1
    assert summary["n_omitted"] == 1
    assert summary["recall"] == 0.5


def test_summarize_empty_entity_list_has_no_recall():
    summary = summarize([])
    assert summary["n_entities"] == 0
    assert summary["recall"] is None
