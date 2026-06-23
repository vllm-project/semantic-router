"""Tests for vllm-sr eval command.

Unit tests: mock requests.post via MagicMock (same pattern as test_chat_command.py).
Integration tests: spin up a real in-process HTTP server so the full HTTP
parsing chain (headers, body, status code) is exercised without a live router.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests
from cli.commands.eval import (
    _format_error_response,
    _normalize_endpoint,
    _parse_messages_json,
    _prompt_to_messages,
    _summarize_response,
)
from cli.commands.eval import (
    eval as eval_command,
)
from cli.commands.recipe_learning import (
    EvalCase,
    build_recipe_learning_artifact,
    candidate_replay_endpoints,
    default_replay_endpoint,
    fetch_replay_payload,
    normalize_replay_endpoint,
    normalize_replay_payload,
)
from cli.commands.recipe_learning_metrics import record_switched
from click.testing import CliRunner

# ---------------------------------------------------------------------------
# Fixture: real in-process HTTP server
# ---------------------------------------------------------------------------


def _make_handler(status: int, body: Any, content_type: str = "application/json"):
    """Return a BaseHTTPRequestHandler subclass that always responds with the
    given status code and JSON-encoded body."""
    body_bytes = (
        json.dumps(body).encode()
        if not isinstance(body, (bytes, str))
        else body.encode() if isinstance(body, str) else body
    )

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            # Drain request body so the client doesn't get a broken-pipe error.
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)

        def log_message(self, fmt, *args):  # silence server logs in test output
            pass

    return _Handler


@pytest.fixture()
def router_server(request):
    """Start a real HTTP server in a background thread.

    Usage:
        @pytest.mark.parametrize("router_server", [...], indirect=True)
        def test_foo(router_server):
            url = router_server   # http://localhost:<port>

    The indirect parameter is a dict: {"status": int, "body": any}.
    """
    params = request.param  # {"status": ..., "body": ...}
    handler = _make_handler(
        params["status"],
        params["body"],
        params.get("content_type", "application/json"),
    )
    server = HTTPServer(("127.0.0.1", 0), handler)  # port=0 → OS picks a free port
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


# ---------------------------------------------------------------------------
# Unit tests: endpoint normalisation + request shape
# ---------------------------------------------------------------------------


def test_normalize_endpoint_defaults_to_eval() -> None:
    assert _normalize_endpoint("").endswith("/api/v1/eval")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("http://localhost:8080", "http://localhost:8080/api/v1/eval"),
        ("http://localhost:8080/", "http://localhost:8080/api/v1/eval"),
        ("http://localhost:8080/api/v1", "http://localhost:8080/api/v1/eval"),
        ("http://localhost:8080/api/v1/eval", "http://localhost:8080/api/v1/eval"),
    ],
)
def test_normalize_endpoint_variants(raw: str, expected: str) -> None:
    assert _normalize_endpoint(raw) == expected


def test_parse_messages_json_requires_array() -> None:
    with pytest.raises(ValueError, match="JSON array"):
        _parse_messages_json('{"role":"user","content":"hi"}')


def test_prompt_to_messages() -> None:
    assert _prompt_to_messages("hi") == [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------------------
# Unit tests: error formatting helpers
# ---------------------------------------------------------------------------


def test_format_error_response_parses_structured_json() -> None:
    """Router structured error JSON is extracted cleanly."""

    class FakeResp:
        status_code = 400
        text = '{"error":{"code":"INVALID_INPUT","message":"text cannot be empty"}}'

        def json(self):
            return {
                "error": {"code": "INVALID_INPUT", "message": "text cannot be empty"}
            }

    msg = _format_error_response(FakeResp())
    assert "INVALID_INPUT" in msg
    assert "text cannot be empty" in msg
    assert "400" in msg


def test_format_error_response_falls_back_to_raw_text() -> None:
    """Plain-text (non-JSON) error body is surfaced as-is."""

    class FakeResp:
        status_code = 503
        text = "service unavailable"

        def json(self):
            raise ValueError("not json")

    msg = _format_error_response(FakeResp())
    assert "503" in msg
    assert "service unavailable" in msg


# ---------------------------------------------------------------------------
# Unit tests: _summarize_response shape coverage
# ---------------------------------------------------------------------------


def test_summarize_response_decision_result_with_signal_confidences() -> None:
    payload = {
        "decision_result": {
            "decision_name": "economics",
            "matched_signals": {"domains": ["economics"], "keywords": ["inflation"]},
            "unmatched_signals": {"embeddings": ["price_movement"]},
            "used_signals": ["domain:economics", "keyword:inflation"],
        },
        "signal_confidences": {"domain:economics": 0.95, "keyword:inflation": 0.87},
        "routing_decision": "economics",
    }
    summary = _summarize_response(payload)
    assert "economics" in summary
    assert "signal confidences" in summary
    assert "0.95" in summary


# ---------------------------------------------------------------------------
# Unit tests: offline Router Learning recipe-learning command
# ---------------------------------------------------------------------------


def _sample_learning_record() -> dict[str, Any]:
    return {
        "id": "replay-1",
        "request_id": "req-1",
        "decision": "simple_general",
        "decision_tier": 1,
        "original_model": "frontier",
        "selected_model": "small",
        "prompt_tokens": 1000,
        "cached_prompt_tokens": 250,
        "total_tokens": 1100,
        "actual_cost": 0.2,
        "baseline_cost": 0.5,
        "cost_savings": 0.3,
        "route_diagnostics": {
            "decision": "simple_general",
            "selected_model": "small",
            "original_model": "frontier",
        },
        "learning": {
            "adaptation": {
                "action": "propose_switch",
                "candidate_set": "decision",
                "strategy": "routing_sampling",
                "sampling": {"used": True},
            },
            "protection": {
                "action": "hold_current",
                "scope": "conversation",
            },
        },
        "outcomes": [
            {
                "source": "eval",
                "target": "model",
                "target_ref": "small",
                "verdict": "overprovisioned",
                "score": 1,
            }
        ],
    }


def test_eval_help_includes_recipe_learning_subcommand() -> None:
    runner = CliRunner()

    result = runner.invoke(eval_command, ["--help"])

    assert result.exit_code == 0
    assert "recipe-learning" in result.output


def test_recipe_learning_normalizes_replay_endpoint() -> None:
    endpoint = normalize_replay_endpoint("http://localhost:8080", 25)

    assert endpoint.startswith("http://localhost:8080/v1/router_replay")
    assert "showDetails=true" in endpoint
    assert "limit=25" in endpoint


def test_recipe_learning_default_replay_endpoint_uses_listener_port() -> None:
    assert default_replay_endpoint().startswith(
        "http://localhost:8899/v1/router_replay"
    )


def test_recipe_learning_candidates_include_listener_fallback_for_api_port() -> None:
    endpoints = candidate_replay_endpoints("http://router.example:8080", 25)

    assert endpoints[0].startswith("http://router.example:8080/v1/router_replay")
    assert endpoints[1].startswith("http://router.example:8899/v1/router_replay")


def test_recipe_learning_fetch_tries_listener_fallback(monkeypatch) -> None:
    calls: list[str] = []

    class _Response:
        def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
            self.status_code = status_code
            self._payload = payload
            self.text = json.dumps(payload)

        def json(self) -> dict[str, Any]:
            return self._payload

    def _fake_get(url: str, timeout: int) -> _Response:
        calls.append(url)
        if ":8080/" in url:
            return _Response(404, {"error": "not found"})
        return _Response(
            200, {"object": "router_replay.list", "data": [_sample_learning_record()]}
        )

    monkeypatch.setattr(requests, "get", _fake_get)

    payload = fetch_replay_payload("http://router.example:8080", 2, 1)

    assert payload["object"] == "router_replay.list"
    assert calls[0].startswith("http://router.example:8080/v1/router_replay")
    assert calls[1].startswith("http://router.example:8899/v1/router_replay")


def test_recipe_learning_normalizes_router_replay_payload() -> None:
    payload = {"object": "router_replay.list", "data": [_sample_learning_record()]}

    assert normalize_replay_payload(payload)[0]["id"] == "replay-1"


def test_recipe_learning_switch_metric_ignores_initial_auto_route() -> None:
    record = {
        "original_model": "auto",
        "selected_model": "qwen/qwen3.6-rocm",
        "learning": {"protection": {"action": "establish_baseline"}},
    }

    assert record_switched(record) is False


def test_recipe_learning_switch_metric_uses_learning_switch_action() -> None:
    record = {
        "original_model": "auto",
        "selected_model": "qwen/qwen3.6-rocm",
        "learning": {"protection": {"action": "allow_switch"}},
    }

    assert record_switched(record) is True


def test_recipe_learning_artifact_contains_metrics_patch_candidates_and_seed_pack() -> (
    None
):
    recipe = {
        "version": "v0.3",
        "routing": {
            "decisions": [
                {
                    "name": "simple_general",
                    "adaptations": {"protection": {"stability_weight": 1.0}},
                }
            ]
        },
    }
    artifact = build_recipe_learning_artifact([_sample_learning_record()], {}, recipe)

    assert artifact["object"] == "router_learning.recipe_learning"
    assert artifact["metrics"]["overall"]["records"] == 1
    assert artifact["metrics"]["per_tier"]["tier_1"]["records"] == 1
    assert artifact["metrics"]["per_tier"]["tier_1"]["decision_tiers"] == {"tier_1": 1}
    assert artifact["findings"]
    assert artifact["findings"][0]["id"].startswith("rlf_")
    assert artifact["findings"][0]["affected_decisions"] == ["simple_general"]
    assert artifact["findings"][0]["next_action"]
    assert artifact["recipe_patch"]["suggestions"]
    assert artifact["recipe_patch"]["suggestions"][0]["finding_id"].startswith("rlf_")
    assert artifact["candidate_recipes"]
    assert artifact["candidate_recipes"][0]["recipe"] is not None
    candidate_decision = artifact["candidate_recipes"][0]["recipe"]["routing"][
        "decisions"
    ][0]
    assert candidate_decision["adaptations"]["adaptation"]["candidate_set"] == "tier"
    assert artifact["experiment_results"]["candidates"]
    assert "per_tier" in artifact["experiment_results"]["candidates"][0]["deltas"]
    seed_record = artifact["experience_seed_pack"]["records"][0]
    assert seed_record["decision_id"] == "simple_general"
    assert seed_record["quality_seed"] == 0.5
    assert seed_record["seed_weight"] == 1
    assert seed_record["source_metric"] == "model_outcomes"
    assert seed_record["support"] == {"model_outcomes": 1}
    assert seed_record["overprovisioned_count"] == 1
    assert "quality_prior" not in seed_record
    assert "decision" not in seed_record


def test_recipe_learning_detects_route_model_and_protection_gaps() -> None:
    record = {
        "id": "replay-route-miss",
        "request_id": "req-route-miss",
        "decision": "simple_general",
        "decision_tier": 1,
        "original_model": "small",
        "selected_model": "frontier",
        "actual_cost": 2.5,
        "latency_ms": 1500,
        "route_diagnostics": {
            "decision": "simple_general",
            "selected_model": "frontier",
            "original_model": "small",
        },
        "learning": {
            "adaptation": {
                "action": "propose_switch",
                "candidate_set": "global",
                "strategy": "routing_sampling",
                "sampling": {"used": True},
            }
        },
        "outcomes": [
            {
                "source": "eval",
                "target": "provider",
                "target_ref": "frontier-provider",
                "verdict": "failed",
            }
        ],
    }
    cases = {
        "replay-route-miss": EvalCase(
            replay_id="replay-route-miss",
            expected_decision="domain_math",
            expected_model="small",
            max_cost=1.0,
            max_latency_ms=1000,
        )
    }
    recipe = {
        "version": "v0.3",
        "routing": {
            "decisions": [
                {"name": "simple_general", "priority": 50},
                {"name": "domain_math", "priority": 40},
            ]
        },
    }

    artifact = build_recipe_learning_artifact([record], cases, recipe)
    finding_types = {item["type"] for item in artifact["findings"]}

    assert {
        "wrong_decision",
        "wrong_model_selection",
        "missing_protection",
        "overly_broad_candidate_set",
        "provider_reliability",
        "latency_violation",
        "cost_violation",
    }.issubset(finding_types)
    suggestions = artifact["recipe_patch"]["suggestions"]
    assert any(item["path"].endswith("/priority") for item in suggestions)
    assert any(item["path"].endswith("/protection/mode") for item in suggestions)
    assert any(item.get("value") == "decision" for item in suggestions)
    materialized = [
        candidate["recipe"]
        for candidate in artifact["candidate_recipes"]
        if candidate.get("recipe") is not None
    ]
    assert any(
        recipe["routing"]["decisions"][0].get("priority") == 40
        for recipe in materialized
    )
    assert any(
        recipe["routing"]["decisions"][0]
        .get("adaptations", {})
        .get("protection", {})
        .get("mode")
        == "apply"
        for recipe in materialized
    )
    assert artifact["metrics"]["per_tier"]["tier_1"]["records"] == 1
    assert any(
        candidate["deltas"]["per_tier"].get("tier_1")
        for candidate in artifact["experiment_results"]["candidates"]
    )


def test_recipe_learning_command_reads_file_and_writes_artifacts(tmp_path) -> None:
    replay_path = tmp_path / "replay.json"
    recipe_path = tmp_path / "recipe.yaml"
    output_dir = tmp_path / "out"
    replay_path.write_text(
        json.dumps(
            {"object": "router_replay.list", "data": [_sample_learning_record()]}
        ),
        encoding="utf-8",
    )
    recipe_path.write_text(
        """
version: v0.3
routing:
  decisions:
    - name: simple_general
      adaptations:
        protection:
          stability_weight: 1.0
""".strip(),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        eval_command,
        [
            "recipe-learning",
            "--replay-file",
            str(replay_path),
            "--recipe-file",
            str(recipe_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert "candidate recipes:" in result.output
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "experiment_results.json").exists()


def test_recipe_learning_report_only_skips_patch_generation(tmp_path) -> None:
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(
        json.dumps(
            {"object": "router_replay.list", "data": [_sample_learning_record()]}
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        eval_command,
        [
            "recipe-learning",
            "--replay-file",
            str(replay_path),
            "--report-only",
            "--json",
        ],
    )

    assert result.exit_code == 0
    artifact = json.loads(result.output)
    assert artifact["findings"]
    assert artifact["recipe_patch"]["mode"] == "report_only"
    assert artifact["recipe_patch"]["suggestions"] == []
    assert artifact["candidate_recipes"] == []


# ---------------------------------------------------------------------------
# Unit tests: CLI flow with mocked requests (MagicMock pattern)
# ---------------------------------------------------------------------------


def test_eval_errors_when_both_prompt_and_messages() -> None:
    runner = CliRunner()
    result = runner.invoke(eval_command, ["--prompt", "hi", "--messages", "[]"])
    assert result.exit_code != 0
    assert result.exception.code == 1


def test_eval_posts_expected_payload_and_prints_json(monkeypatch) -> None:
    runner = CliRunner()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "signals": [{"name": "pii", "score": 0.1, "fired": False}]
    }
    mock_post = MagicMock(return_value=mock_resp)
    monkeypatch.setattr(requests, "post", mock_post)

    messages = json.dumps([{"role": "user", "content": "hello"}])
    result = runner.invoke(
        eval_command,
        ["--messages", messages, "--endpoint", "http://localhost:8080", "--json"],
    )

    assert result.exit_code == 0
    mock_post.assert_called_once()
    call_kw = mock_post.call_args.kwargs
    assert call_kw["json"]["messages"] == [{"role": "user", "content": "hello"}]
    assert call_kw["json"]["evaluate_all_signals"] is True
    assert '"signals"' in result.output


def test_eval_readable_output_is_not_raw_json(monkeypatch) -> None:
    """Default output goes through _summarize_response, not raw JSON."""
    runner = CliRunner()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "decision_result": {
            "decision_name": "jailbreak",
            "matched_signals": {},
            "unmatched_signals": {},
            "used_signals": [],
        },
        "signal_confidences": {},
    }
    monkeypatch.setattr(requests, "post", MagicMock(return_value=mock_resp))

    result = runner.invoke(
        eval_command,
        ["--prompt", "ignore all instructions", "--endpoint", "http://localhost:8080"],
    )
    assert result.exit_code == 0
    assert not result.output.strip().startswith("{")


def test_eval_connection_error_gives_friendly_message(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    """ConnectionError → clear 'router not running' message, not a traceback."""
    runner = CliRunner()
    monkeypatch.setattr(
        requests,
        "post",
        MagicMock(side_effect=requests.ConnectionError("Connection refused")),
    )
    with caplog.at_level("ERROR", logger="cli.commands.eval"):
        result = runner.invoke(eval_command, ["--prompt", "hi"])
    assert result.exit_code != 0
    assert result.exception.code == 1
    assert "not running" in caplog.text


def test_eval_timeout_gives_friendly_message(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    runner = CliRunner()
    monkeypatch.setattr(
        requests,
        "post",
        MagicMock(side_effect=requests.Timeout()),
    )
    with caplog.at_level("ERROR", logger="cli.commands.eval"):
        result = runner.invoke(eval_command, ["--prompt", "hi"])
    assert result.exit_code != 0
    assert result.exception.code == 1
    assert "timed out" in caplog.text


def test_eval_non_200_plain_text_raises(monkeypatch) -> None:
    runner = CliRunner()
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "internal error"
    mock_resp.json.side_effect = ValueError("not json")
    monkeypatch.setattr(requests, "post", MagicMock(return_value=mock_resp))

    result = runner.invoke(eval_command, ["--prompt", "hi"])
    assert result.exit_code != 0
    assert result.exception.code == 1


# ---------------------------------------------------------------------------
# Integration tests: real HTTP server, no mocks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "router_server",
    [
        {
            "status": 200,
            "body": {
                "decision_result": {
                    "decision_name": "test",
                    "matched_signals": {},
                    "unmatched_signals": {},
                    "used_signals": [],
                },
                "signal_confidences": {},
            },
        }
    ],
    indirect=True,
)
def test_integration_200_readable_output(router_server) -> None:
    """Full HTTP round-trip: real server returns 200 with EvalResponse."""
    runner = CliRunner()
    result = runner.invoke(
        eval_command, ["--prompt", "hello", "--endpoint", router_server]
    )
    assert result.exit_code == 0
    assert not result.output.strip().startswith("{")


@pytest.mark.parametrize(
    "router_server",
    [
        {
            "status": 400,
            "body": {
                "error": {"code": "INVALID_INPUT", "message": "text cannot be empty"}
            },
        }
    ],
    indirect=True,
)
def test_integration_400_structured_error(
    router_server, caplog: pytest.LogCaptureFixture
) -> None:
    """Full HTTP round-trip: real server returns 400 with structured JSON error."""
    runner = CliRunner()
    with caplog.at_level("ERROR", logger="cli.commands.eval"):
        result = runner.invoke(
            eval_command, ["--prompt", "hello", "--endpoint", router_server]
        )
    assert result.exit_code != 0
    assert "INVALID_INPUT" in caplog.text
    assert "text cannot be empty" in caplog.text


@pytest.mark.parametrize(
    "router_server",
    [{"status": 503, "body": "service unavailable", "content_type": "text/plain"}],
    indirect=True,
)
def test_integration_503_plain_text_error(
    router_server, caplog: pytest.LogCaptureFixture
) -> None:
    """Full HTTP round-trip: real server returns 503 with plain-text body."""
    runner = CliRunner()
    with caplog.at_level("ERROR", logger="cli.commands.eval"):
        result = runner.invoke(
            eval_command, ["--prompt", "hello", "--endpoint", router_server]
        )
    assert result.exit_code != 0
    assert "503" in caplog.text


@pytest.mark.parametrize(
    "router_server",
    [
        {
            "status": 200,
            "body": {
                "decision_result": {
                    "decision_name": "economics",
                    "matched_signals": {"domains": ["economics"]},
                    "unmatched_signals": {},
                    "used_signals": ["domain:economics"],
                },
                "signal_confidences": {"domain:economics": 0.95},
            },
        }
    ],
    indirect=True,
)
def test_integration_200_json_flag(router_server) -> None:
    """Full HTTP round-trip: --json flag outputs raw payload."""
    runner = CliRunner()
    result = runner.invoke(
        eval_command, ["--prompt", "inflation", "--endpoint", router_server, "--json"]
    )
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["decision_result"]["decision_name"] == "economics"
