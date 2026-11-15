"""recipes[].decisions[].fallback and recipes[].fallback in the CLI's config models."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

CLI_ROOT = Path(__file__).resolve().parents[1]
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

from cli.models import Decision, FallbackOverride, Recipe  # noqa: E402
from cli.parser import parse_user_config  # noqa: E402

_DOCUMENT = """version: v0.5
providers:
  models:
    - name: small
      backend_refs:
        - endpoint: 10.0.0.1:8000
          provider: vllm
    - name: large
      backend_refs:
        - endpoint: 10.0.0.2:8000
          provider: vllm
recipes:
  - name: default
    fallback:
      enabled: true
      max_attempts: 3
      circuit_breaker: {consecutive_failures: 3, cooldown_period: 30s}
    decisions:
      - name: escalate
        priority: 100
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: small
          - model: large
        fallback:
          max_attempts: 2
          total_timeout: 20s
          per_attempt_timeout: 8s
          retryable_status_codes: [429, 503]
"""


def test_a_config_with_decision_and_recipe_fallback_parses(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_DOCUMENT)
    parsed = parse_user_config(str(config_path), log_summary=False)
    assert parsed.default_recipe.fallback.circuit_breaker.consecutive_failures == 3
    override = parsed.default_recipe.decisions[0].fallback
    assert override.max_attempts == 2
    assert override.total_timeout == "20s"
    assert override.retryable_status_codes == [429, 503]


def test_recipe_routing_takes_a_fallback_policy():
    recipe = Recipe.model_validate(
        {"name": "default", "fallback": {"total_timeout": 20_000_000_000}}
    )
    assert recipe.fallback.total_timeout == 20_000_000_000
    assert (
        Recipe.model_validate(
            {"name": "default", "fallback": {"enabled": False}}
        ).fallback.enabled
        is False
    )


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"circuit_breaker": {"consecutive_failures": 3}}, "circuit_breaker"),
        ({"max_attempts": -1}, "max_attempts"),
        ({"retryable_status_codes": [700]}, "retryable_status_codes"),
        ({"total_timeout": "soon"}, "total_timeout"),
        ({"enabled": "yes"}, "enabled"),
    ],
)
def test_a_decision_fallback_rejects_what_the_router_rejects(override, field):
    with pytest.raises(ValidationError, match=field):
        Decision.model_validate({"name": "d", "priority": 1, "fallback": override})


def test_a_recipe_fallback_forwards_named_request_headers(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        _DOCUMENT.replace(
            "      circuit_breaker: {consecutive_failures: 3, cooldown_period: 30s}\n",
            "      circuit_breaker: {consecutive_failures: 3, cooldown_period: 30s}\n"
            "      forward_request_headers: [x-user-id, X-Request-Id]\n",
        )
    )
    parsed = parse_user_config(str(config_path), log_summary=False)
    assert parsed.default_recipe.fallback.forward_request_headers == [
        "x-user-id",
        "X-Request-Id",
    ]


@pytest.mark.parametrize(
    ("name", "message"),
    [
        (":authority", "cannot include pseudo-header"),
        ("x user", "not an HTTP header name"),
        ("Authorization", "cannot include credential header"),
        ("ocp-apim-subscription-key", "cannot include credential header"),
        ("connection", "framing and body headers"),
        ("proxy-connection", "framing and body headers"),
        ("Host", "framing and body headers"),
        ("x-vsr-selected-model", "belong to the Router and Envoy"),
        ("x-authz-user-id", "cannot include identity header"),
        ("X-Authz-Role", "cannot include identity header"),
    ],
)
def test_a_recipe_fallback_refuses_headers_no_call_may_carry(name, message):
    with pytest.raises(ValidationError, match=message):
        Recipe.model_validate(
            {"name": "default", "fallback": {"forward_request_headers": [name]}}
        )


def test_a_recipe_fallback_refuses_a_forwarded_header_twice():
    with pytest.raises(ValidationError, match="lists 'X-User-Id' twice"):
        Recipe.model_validate(
            {
                "name": "default",
                "fallback": {"forward_request_headers": ["x-user-id", "X-User-Id"]},
            }
        )


def test_a_per_attempt_timeout_cannot_exceed_the_total():
    with pytest.raises(ValidationError, match="per_attempt_timeout cannot exceed"):
        FallbackOverride.model_validate(
            {"total_timeout": "10s", "per_attempt_timeout": "20s"}
        )
