"""Tests for RouterClient HTTP error handling."""

from __future__ import annotations

import io
import json
from unittest import mock
from urllib import error

import pytest

from tuning.client import RouterClient


def _http_error(code: int, body: bytes) -> error.HTTPError:
    return error.HTTPError(
        url="http://localhost:8080/api/v1/eval",
        code=code,
        msg="Service Unavailable",
        hdrs=None,
        fp=io.BytesIO(body),
    )


def test_eval_probe_returns_decision_error_payload():
    payload = {
        "decision_error": "decision unresolved",
        "applied_unknown_policies": {"guarded": "fail_request"},
    }
    client = RouterClient()
    with mock.patch(
        "tuning.client.request.urlopen",
        side_effect=_http_error(503, json.dumps(payload).encode()),
    ):
        assert client.eval_probe("hello") == payload


def test_eval_probe_raises_typed_error_for_non_json_body():
    client = RouterClient()
    with mock.patch(
        "tuning.client.request.urlopen",
        side_effect=_http_error(502, b"<html>bad gateway</html>"),
    ):
        with pytest.raises(RuntimeError, match="HTTP 502"):
            client.eval_probe("hello")


def test_run_probes_reports_unresolved_decision_as_bounded_failure():
    payload = {
        "decision_error": "decision unresolved",
        "applied_unknown_policies": {"guarded": "fail_request"},
    }
    client = RouterClient()
    with mock.patch.object(client, "eval_probe", return_value=payload):
        results = client.run_probes(
            [{"id": "p1", "query": "hello", "expected_decision": "guarded"}]
        )
    assert results == [
        {
            "id": "p1",
            "query": "hello",
            "expected": "guarded",
            "actual": "UNRESOLVED",
            "correct": False,
            "error": "decision unresolved",
            "applied_unknown_policies": {"guarded": "fail_request"},
            "tags": [],
        }
    ]
