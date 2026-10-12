"""routing.decisions[].reliability: validation and the Envoy rendering it needs."""

import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

CLI_ROOT = Path(__file__).resolve().parents[1]
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

from cli.config_generator import generate_envoy_config_from_user_config  # noqa: E402
from cli.models import DecisionReliability  # noqa: E402
from cli.parser import parse_user_config  # noqa: E402

_PROVIDERS = """
providers:
  models:
    - name: m
      backend_refs:
        - endpoint: 10.0.0.1:8000
          provider: vllm
routing:
  modelCards:
    - name: m
"""

_RELIABILITY_HEADERS = (
    "^x-envoy-(upstream-rq-timeout-ms|upstream-rq-per-try-timeout-ms|"
    "max-retries|retry-on|retriable-status-codes)$"
)


def _decisions(reliability: str = "") -> str:
    return (
        """
  decisions:
    - name: bounded-route
      priority: 100
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: m
"""
        + reliability
    )


def _render(tmp_path, monkeypatch, document: str):
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "envoy.yaml"
    config_path.write_text("version: v0.3\n" + document)
    monkeypatch.setenv("ENVOY_EXTPROC_ADDRESS", "localhost")
    monkeypatch.setenv("ENVOY_ROUTER_API_ADDRESS", "localhost")
    generate_envoy_config_from_user_config(
        parse_user_config(str(config_path)), str(output_path), log_summary=False
    )
    return yaml.safe_load(output_path.read_text())


def _ext_proc(rendered):
    listener = rendered["static_resources"]["listeners"][0]
    hcm = listener["filter_chains"][0]["filters"][0]["typed_config"]
    return next(
        f["typed_config"]
        for f in hcm["http_filters"]
        if f["name"] == "envoy.filters.http.ext_proc"
    )


def test_decision_reliability_lets_ext_proc_set_only_the_reliability_headers(
    tmp_path, monkeypatch
):
    rendered = _render(
        tmp_path,
        monkeypatch,
        _PROVIDERS
        + _decisions(
            """      reliability:
        total_timeout: 90s
        per_try_timeout: 0s
        retry_count: 2
        retry_on: reset
        retriable_status_codes: [503]
"""
        ),
    )
    assert _ext_proc(rendered)["mutation_rules"] == {
        "allow_expression": {"regex": _RELIABILITY_HEADERS}
    }


def test_the_rule_is_rendered_without_any_override(tmp_path, monkeypatch):
    # The Router reloads its configuration without Envoy, so an override added
    # later must already be allowed.
    rendered = _render(tmp_path, monkeypatch, _PROVIDERS + _decisions())
    assert _ext_proc(rendered)["mutation_rules"] == {
        "allow_expression": {"regex": _RELIABILITY_HEADERS}
    }


@pytest.mark.parametrize(
    "field, value",
    [
        ("idle_timeout", "30s"),
        ("first_byte_timeout", "5s"),
        ("retry_back_off_base", "50ms"),
        ("retry_back_off_max", "1s"),
        ("retry_after_max", "10s"),
    ],
)
def test_envoy_render_rejects_what_envoy_cannot_apply_per_request(
    tmp_path, monkeypatch, field, value
):
    document = _PROVIDERS + _decisions(
        f"      reliability:\n        {field}: {value}\n"
    )
    with pytest.raises(ValueError, match=f"reliability.{field} is honored only"):
        _render(tmp_path, monkeypatch, document)


def test_envoy_render_names_the_recipe_of_a_rejected_decision(tmp_path, monkeypatch):
    document = (
        _PROVIDERS
        + _decisions()
        + """recipes:
  - name: support
    routing:
      decisions:
        - name: slow-route
          priority: 10
          rules:
            operator: AND
            conditions: []
          modelRefs:
            - model: m
          reliability:
            idle_timeout: 30s
"""
    )
    with pytest.raises(ValueError, match=r"recipes\['support'\].*'slow-route'"):
        _render(tmp_path, monkeypatch, document)


def test_decision_reliability_validates_like_the_provider_block():
    DecisionReliability(total_timeout="0s", per_try_timeout="0s", retry_count=0)
    for invalid in (
        {"retry_count": 6},
        {"total_timeout": "soon"},
        {"retriable_status_codes": [700]},
        {"retry_back_off_base": "0s"},
        {"retry_back_off_max": "10ms"},
        {"retry_after_max": "0s"},
        {"unknown_field": "1s"},
    ):
        with pytest.raises(ValidationError):
            DecisionReliability(**invalid)


def test_unset_decision_fields_stay_unset():
    reliability = DecisionReliability(total_timeout="90s")
    assert reliability.model_dump(exclude_none=True) == {"total_timeout": "90s"}
    assert reliability.native_only_fields() == []
