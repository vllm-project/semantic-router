"""Tests for ``vllm-sr storage response-cache``.

The command manages the router's response cache via:
- GET /api/v1/storage/response-cache/capabilities
- GET /api/v1/storage/response-cache/stats
- GET /api/v1/storage/response-cache/health
- POST /api/v1/storage/response-cache/flush
- POST /api/v1/storage/response-cache/invalidate
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

main = importlib.import_module("cli.main").main
management_client = importlib.import_module("cli.router_management_client")


def _fake_response(status_code: int, json_body=None, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 400
    resp.text = text
    resp.headers = {}
    if json_body is None:
        resp.json.side_effect = ValueError("no json")
    else:
        resp.json.return_value = json_body
    return resp


_SAMPLE_CAPABILITIES = {
    "backend": "redis",
    "exact": True,
    "semantic": True,
    "per_entry_ttl": True,
    "stats": True,
    "invalidate": True,
    "flush": True,
    "shared": True,
    "consistency_model": "eventual",
}

_SAMPLE_HEALTH = {
    "status": "healthy",
    "capabilities": _SAMPLE_CAPABILITIES,
}

_SAMPLE_STATS = {
    "total_entries": 120,
    "hit_count": 48,
    "miss_count": 52,
    "hit_ratio": 0.48,
    "exact_hit_count": 32,
    "semantic_hit_count": 16,
    "l1_hit_count": 20,
    "l2_hit_count": 28,
    "l1_entries": 60,
    "singleflight_waiters": 0,
    "invalidation_count": 3,
}


def test_response_cache_registered_on_storage_help():
    runner = CliRunner()
    result = runner.invoke(main, ["storage", "--help"])

    assert result.exit_code == 0
    assert "response-cache" in result.output


def test_response_cache_capabilities_formatted(monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()
    monkeypatch.setattr(
        management_client.requests,
        "request",
        lambda *a, **k: _fake_response(200, _SAMPLE_CAPABILITIES),
    )

    result = runner.invoke(main, ["storage", "response-cache", "capabilities"])

    assert result.exit_code == 0
    assert "Response cache capabilities" in result.output
    assert "Backend" in result.output
    assert "redis" in result.output
    assert "Exact" in result.output
    assert "true" in result.output
    assert "Semantic" in result.output
    assert "Per-entry TTL" in result.output
    assert "Stats" in result.output
    assert "Invalidate" in result.output
    assert "Flush" in result.output
    assert "Shared" in result.output
    assert "Consistency model" in result.output
    assert "eventual" in result.output


def test_response_cache_capabilities_json(monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()
    monkeypatch.setattr(
        management_client.requests,
        "request",
        lambda *a, **k: _fake_response(200, _SAMPLE_CAPABILITIES),
    )

    result = runner.invoke(
        main, ["storage", "response-cache", "capabilities", "--json"]
    )

    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["backend"] == "redis"
    assert parsed["exact"] is True
    assert parsed["semantic"] is True
    assert parsed["per_entry_ttl"] is True
    assert parsed["stats"] is True
    assert parsed["invalidate"] is True
    assert parsed["flush"] is True
    assert parsed["shared"] is True
    assert parsed["consistency_model"] == "eventual"


def test_response_cache_stats_formatted(monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()
    monkeypatch.setattr(
        management_client.requests,
        "request",
        lambda *a, **k: _fake_response(200, _SAMPLE_STATS),
    )

    result = runner.invoke(main, ["storage", "response-cache", "stats"])

    assert result.exit_code == 0
    assert "Response cache statistics" in result.output
    assert "Total entries" in result.output
    assert "120" in result.output
    assert "Hit count" in result.output
    assert "48" in result.output
    assert "Miss count" in result.output
    assert "52" in result.output
    assert "Hit ratio" in result.output
    assert "0.48" in result.output
    assert "Exact hit count" in result.output
    assert "32" in result.output
    assert "Semantic hit count" in result.output
    assert "16" in result.output
    assert "L1 hit count" in result.output
    assert "20" in result.output
    assert "L2 hit count" in result.output
    assert "28" in result.output
    assert "L1 entries" in result.output
    assert "60" in result.output
    assert "Singleflight waiters" in result.output
    assert "0" in result.output
    assert "Invalidation count" in result.output
    assert "3" in result.output


def test_response_cache_stats_json(monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()
    monkeypatch.setattr(
        management_client.requests,
        "request",
        lambda *a, **k: _fake_response(200, _SAMPLE_STATS),
    )

    result = runner.invoke(main, ["storage", "response-cache", "stats", "--json"])

    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["total_entries"] == 120
    assert parsed["hit_ratio"] == 0.48


def test_response_cache_health_formatted(monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()
    monkeypatch.setattr(
        management_client.requests,
        "request",
        lambda *a, **k: _fake_response(200, _SAMPLE_HEALTH),
    )

    result = runner.invoke(main, ["storage", "response-cache", "health"])

    assert result.exit_code == 0
    assert "Response cache health" in result.output
    assert "Status" in result.output
    assert "healthy" in result.output
    assert "Backend" in result.output
    assert "redis" in result.output


def test_response_cache_health_json(monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()
    monkeypatch.setattr(
        management_client.requests,
        "request",
        lambda *a, **k: _fake_response(200, _SAMPLE_HEALTH),
    )

    result = runner.invoke(main, ["storage", "response-cache", "health", "--json"])

    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["status"] == "healthy"
    assert parsed["capabilities"]["backend"] == "redis"


def test_response_cache_flush_with_yes(monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()
    flush_result = {"deleted": 15, "matched_estimate": 15}
    request_mock = MagicMock(return_value=_fake_response(200, flush_result))
    monkeypatch.setattr(management_client.requests, "request", request_mock)

    result = runner.invoke(main, ["storage", "response-cache", "flush", "--yes"])

    assert result.exit_code == 0
    assert "Response cache flushed" in result.output
    assert "Deleted" in result.output
    assert "15" in result.output
    assert request_mock.call_args.kwargs["json"] == {"confirm": "flush response cache"}


def test_response_cache_flush_interactive_confirm(monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()
    flush_result = {"deleted": 8, "matched_estimate": 8}
    request_mock = MagicMock(return_value=_fake_response(200, flush_result))
    monkeypatch.setattr(management_client.requests, "request", request_mock)
    monkeypatch.setattr("cli.commands.response_cache._is_interactive", lambda: True)

    result = runner.invoke(main, ["storage", "response-cache", "flush"], input="y\n")

    assert result.exit_code == 0
    assert "Response cache flushed" in result.output
    assert request_mock.call_args.kwargs["json"] == {"confirm": "flush response cache"}


def test_response_cache_flush_interactive_abort(monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()
    monkeypatch.setattr("cli.commands.response_cache._is_interactive", lambda: True)

    result = runner.invoke(main, ["storage", "response-cache", "flush"], input="n\n")

    assert result.exit_code == 0
    assert "Aborted." in result.output


def test_response_cache_flush_non_tty_abort():
    runner = CliRunner()
    result = runner.invoke(main, ["storage", "response-cache", "flush"])

    assert result.exit_code != 0
    assert (
        "Confirmation required. Run with --yes in non-interactive environments."
        in result.output
    )


def test_response_cache_flush_with_partition_filters(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = CliRunner()
    flush_result = {"deleted": 5, "matched_estimate": 5}
    request_mock = MagicMock(return_value=_fake_response(200, flush_result))
    monkeypatch.setattr(management_client.requests, "request", request_mock)

    result = runner.invoke(
        main,
        [
            "storage",
            "response-cache",
            "flush",
            "--yes",
            "--recipe",
            "code-fast",
            "--decision",
            "decision-1",
            "--request-model",
            "gpt-4o",
            "--selected-model",
            "qwen-2.5",
            "--protocol",
            "openai",
            "--namespace",
            "team-a",
            "--epoch",
            "v2",
        ],
    )

    assert result.exit_code == 0
    payload = request_mock.call_args.kwargs["json"]
    assert payload["confirm"] == "flush response cache"
    assert payload["selector"]["partition"] == {
        "recipe": "code-fast",
        "decision": "decision-1",
        "request_model": "gpt-4o",
        "selected_model": "qwen-2.5",
        "protocol": "openai",
        "namespace": "team-a",
        "epoch": "v2",
    }
    assert payload["selector"]["epoch"] == "v2"


def test_response_cache_flush_json(monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()
    flush_result = {"deleted": 5, "matched_estimate": 5}
    request_mock = MagicMock(return_value=_fake_response(200, flush_result))
    monkeypatch.setattr(management_client.requests, "request", request_mock)

    result = runner.invoke(
        main, ["storage", "response-cache", "flush", "--yes", "--json"]
    )

    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["deleted"] == 5
    assert parsed["matched_estimate"] == 5


def test_response_cache_invalidate_dry_run_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = CliRunner()
    invalidation_result = {
        "dry_run": True,
        "matched_estimate": 10,
        "deleted": 0,
    }
    request_mock = MagicMock(return_value=_fake_response(200, invalidation_result))
    monkeypatch.setattr(management_client.requests, "request", request_mock)

    result = runner.invoke(
        main,
        [
            "storage",
            "response-cache",
            "invalidate",
            "--recipe",
            "default",
        ],
    )

    assert result.exit_code == 0
    assert "Response cache invalidation (dry-run)" in result.output
    assert "Matched estimate" in result.output
    assert "10" in result.output
    assert "Deleted" in result.output
    assert "0" in result.output
    assert "Pass --execute to perform actual invalidation." in result.output
    payload = request_mock.call_args.kwargs["json"]
    assert payload["dry_run"] is True
    assert payload["selector"]["partition"] == {"recipe": "default"}


def test_response_cache_invalidate_execute(monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()
    invalidation_result = {
        "dry_run": False,
        "matched_estimate": 10,
        "deleted": 10,
    }
    request_mock = MagicMock(return_value=_fake_response(200, invalidation_result))
    monkeypatch.setattr(management_client.requests, "request", request_mock)

    result = runner.invoke(
        main,
        [
            "storage",
            "response-cache",
            "invalidate",
            "--execute",
            "--recipe",
            "default",
        ],
    )

    assert result.exit_code == 0
    assert "Response cache invalidation completed" in result.output
    assert "Deleted" in result.output
    assert "10" in result.output
    payload = request_mock.call_args.kwargs["json"]
    assert payload["dry_run"] is False


def test_response_cache_invalidate_with_partition_filters(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = CliRunner()
    invalidation_result = {
        "dry_run": True,
        "matched_estimate": 3,
        "deleted": 0,
    }
    request_mock = MagicMock(return_value=_fake_response(200, invalidation_result))
    monkeypatch.setattr(management_client.requests, "request", request_mock)

    result = runner.invoke(
        main,
        [
            "storage",
            "response-cache",
            "invalidate",
            "--recipe",
            "code-fast",
            "--decision",
            "decision-1",
            "--request-model",
            "gpt-4o",
            "--selected-model",
            "qwen-2.5",
            "--protocol",
            "openai",
            "--namespace",
            "team-a",
            "--epoch",
            "v1",
        ],
    )

    assert result.exit_code == 0
    payload = request_mock.call_args.kwargs["json"]
    assert payload["selector"]["partition"] == {
        "recipe": "code-fast",
        "decision": "decision-1",
        "request_model": "gpt-4o",
        "selected_model": "qwen-2.5",
        "protocol": "openai",
        "namespace": "team-a",
        "epoch": "v1",
    }
    assert payload["selector"]["epoch"] == "v1"


def test_response_cache_invalidate_json(monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()
    invalidation_result = {
        "dry_run": True,
        "matched_estimate": 3,
        "deleted": 0,
    }
    request_mock = MagicMock(return_value=_fake_response(200, invalidation_result))
    monkeypatch.setattr(management_client.requests, "request", request_mock)

    result = runner.invoke(
        main,
        [
            "storage",
            "response-cache",
            "invalidate",
            "--json",
        ],
    )

    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["dry_run"] is True
    assert parsed["matched_estimate"] == 3
    assert parsed["deleted"] == 0


def test_response_cache_reports_401_unauthorized(
    caplog, monkeypatch: pytest.MonkeyPatch
):
    runner = CliRunner()
    monkeypatch.setattr(
        management_client.requests,
        "request",
        lambda *a, **k: _fake_response(
            401,
            json_body={"error": "UNAUTHORIZED", "message": "unauthorized access"},
        ),
    )

    with caplog.at_level("ERROR"):
        result = runner.invoke(main, ["storage", "response-cache", "stats"])

    assert result.exit_code == 1
    combined = "\n".join(record.message for record in caplog.records)
    assert "HTTP 401" in combined
    assert "unauthorized access" in combined


def test_response_cache_reports_403_forbidden(caplog, monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()
    monkeypatch.setattr(
        management_client.requests,
        "request",
        lambda *a, **k: _fake_response(
            403,
            json_body={
                "error": "FORBIDDEN",
                "message": "insufficient permissions",
            },
        ),
    )

    with caplog.at_level("ERROR"):
        result = runner.invoke(main, ["storage", "response-cache", "stats"])

    assert result.exit_code == 1
    combined = "\n".join(record.message for record in caplog.records)
    assert "HTTP 403" in combined
    assert "insufficient permissions" in combined


def test_response_cache_reports_503_cache_unavailable(
    caplog, monkeypatch: pytest.MonkeyPatch
):
    runner = CliRunner()
    monkeypatch.setattr(
        management_client.requests,
        "request",
        lambda *a, **k: _fake_response(
            503,
            json_body={
                "error": "CACHE_UNAVAILABLE",
                "message": "Response cache is unavailable",
            },
        ),
    )

    with caplog.at_level("ERROR"):
        result = runner.invoke(main, ["storage", "response-cache", "stats"])

    assert result.exit_code == 1
    combined = "\n".join(record.message for record in caplog.records)
    assert "HTTP 503" in combined
    assert "Response cache is unavailable" in combined


def test_response_cache_reports_unreachable_router(
    caplog, monkeypatch: pytest.MonkeyPatch
):
    runner = CliRunner()

    def _boom(*a, **k):
        raise management_client.requests.ConnectionError("connection refused")

    monkeypatch.setattr(management_client.requests, "request", _boom)

    with caplog.at_level("ERROR"):
        result = runner.invoke(main, ["storage", "response-cache", "stats"])

    assert result.exit_code == 1
    combined = "\n".join(record.message for record in caplog.records)
    assert "Router management API is not reachable at http://localhost:8080" in combined


def test_response_cache_uses_canonical_path_and_token_env(
    monkeypatch: pytest.MonkeyPatch,
):
    request = MagicMock(return_value=_fake_response(200, _SAMPLE_STATS))
    monkeypatch.setattr(management_client.requests, "request", request)
    monkeypatch.setenv("CUSTOM_MGMT_TOKEN", "secret-bearer-token")

    result = CliRunner().invoke(
        main,
        [
            "storage",
            "response-cache",
            "stats",
            "--endpoint",
            "http://router.internal:8080/api/v1",
            "--token-env",
            "CUSTOM_MGMT_TOKEN",
        ],
    )

    assert result.exit_code == 0
    assert request.call_args.args[:2] == (
        "GET",
        "http://router.internal:8080/api/v1/storage/response-cache/stats",
    )
    assert (
        request.call_args.kwargs["headers"]["Authorization"]
        == "Bearer secret-bearer-token"
    )
    assert "secret-bearer-token" not in result.output
