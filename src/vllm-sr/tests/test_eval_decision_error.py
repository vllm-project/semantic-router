"""503 rendering for the eval command's decision-unresolved payload."""

from __future__ import annotations

from cli.commands.eval import _format_error_response


def test_format_error_response_renders_decision_error_shape() -> None:
    """The eval endpoint's 503 carries the evaluation payload, not the envelope."""

    class FakeResp:
        status_code = 503
        text = "{}"

        def json(self):
            return {
                "original_text": "hello",
                "decision_error": "decision unresolved",
                "applied_unknown_policies": {"guarded": "fail_request"},
            }

    msg = _format_error_response(FakeResp())
    assert "503" in msg
    assert "decision unresolved" in msg
    assert "guarded=fail_request" in msg


def test_format_error_response_keeps_legacy_503_envelope() -> None:
    """A 503 with the classic error envelope and no decision fields renders as before."""

    class FakeResp:
        status_code = 503
        text = "{}"

        def json(self):
            return {
                "error": {"code": "CLASSIFICATION_ERROR", "message": "classifier down"}
            }

    msg = _format_error_response(FakeResp())
    assert "503" in msg
    assert "CLASSIFICATION_ERROR" in msg
    assert "classifier down" in msg
