from cli.evaluation.routing_trace import normalize_routing_diagnostic


def test_normalize_preserves_bounded_decision_error():
    payload = {
        "decision_error": 'decision evaluation failed: decision "guarded" could not be resolved',
        "applied_unknown_policies": {"guarded": "fail_request"},
    }
    diagnostic = normalize_routing_diagnostic("case-1", payload)
    assert diagnostic.decision_error is not None
    assert "guarded" in diagnostic.decision_error
    assert '"' not in diagnostic.decision_error
    assert diagnostic.applied_unknown_policies == (("guarded", "fail_request"),)


def test_normalize_bounds_decision_error_length():
    diagnostic = normalize_routing_diagnostic("case-2", {"decision_error": "x" * 5000})
    assert diagnostic.decision_error is not None
    assert len(diagnostic.decision_error) == 200


def test_normalize_omits_absent_or_invalid_decision_error():
    assert normalize_routing_diagnostic("case-3", {}).decision_error is None
    assert (
        normalize_routing_diagnostic("case-4", {"decision_error": 7}).decision_error
        is None
    )
