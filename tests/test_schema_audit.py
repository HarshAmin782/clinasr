from clinasr.schema_audit import audit_records


def test_reports_field_types_and_missingness():
    records = [
        {"session_id": "s1", "text": "hello", "recording_context": "clinic"},
        {"session_id": "s2", "text": "world", "recording_context": None},
        {"session_id": "s3", "text": 123, "recording_context": "pharmacy"},
    ]
    report = audit_records(records)
    assert report.n_records_inspected == 3

    ctx = report.fields["recording_context"]
    assert ctx.n_present == 2
    assert ctx.n_missing == 1

    text = report.fields["text"]
    assert set(text.observed_types) == {"str", "int"}


def test_schema_drift_field_present_on_some_records_only():
    records = [{"a": 1}, {"a": 2, "b": "extra"}]
    report = audit_records(records)
    assert report.fields["b"].n_present == 1
    assert report.fields["b"].n_missing == 1


def test_missing_expected_lists_absent_fields():
    report = audit_records([{"a": 1}])
    missing = report.missing_expected(["a", "medical_entities"])
    assert missing == ["medical_entities"]


def test_empty_records_list():
    report = audit_records([])
    assert report.n_records_inspected == 0
    assert report.fields == {}
