"""Envoy rendering of providers.models[].reliability against the shared table."""

import sys
from pathlib import Path

import pytest
import yaml

CLI_ROOT = Path(__file__).resolve().parents[1]
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

from cli.config_generator import generate_envoy_config_from_user_config  # noqa: E402
from cli.durations import envoy_duration, parse_duration  # noqa: E402
from cli.parser import parse_user_config  # noqa: E402

REPO_ROOT = CLI_ROOT.parents[1]
DEFAULTS = yaml.safe_load(
    (
        REPO_ROOT
        / "src/semantic-router/pkg/upstream/testdata/reliability-defaults.yaml"
    ).read_text()
)

_ROUTING = """
routing:
  modelCards:
    - name: m
  decisions:
    - name: default-route
      description: default route
      priority: 100
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: m
"""


def _render(tmp_path, monkeypatch, providers: str, listeners: str = ""):
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "envoy.yaml"
    config_path.write_text("version: v0.3\n" + listeners + providers + _ROUTING)
    monkeypatch.setenv("ENVOY_EXTPROC_ADDRESS", "localhost")
    monkeypatch.setenv("ENVOY_ROUTER_API_ADDRESS", "localhost")
    generate_envoy_config_from_user_config(
        parse_user_config(str(config_path)), str(output_path), log_summary=False
    )
    return yaml.safe_load(output_path.read_text())


def _hcm(rendered):
    listener = rendered["static_resources"]["listeners"][0]
    return listener["filter_chains"][0]["filters"][0]["typed_config"]


def _routes(rendered):
    return _hcm(rendered)["route_config"]["virtual_hosts"][0]["routes"]


def _cluster(rendered):
    return next(
        c
        for c in rendered["static_resources"]["clusters"]
        if c["name"] == "model_m_cluster"
    )


def _seconds(go_duration: str) -> str:
    return envoy_duration(go_duration)


_BARE = """
providers:
  models:
    - name: m
      backend_refs:
        - endpoint: 10.0.0.1:8000
          provider: vllm
"""

_ENABLED = """
providers:
  models:
    - name: m
      reliability:
        lb_policy: least_request
        retry_count: 1
        consecutive_5xx: 3
        health_check_path: /health
        retry_budget_min_concurrency: 5
      backend_refs:
        - endpoint: 10.0.0.1:8000
          provider: vllm
        - endpoint: 10.0.0.2:8000
          provider: vllm
"""


def test_bare_model_renders_the_shared_defaults(tmp_path, monkeypatch):
    rendered = _render(tmp_path, monkeypatch, _BARE)
    listener_timeout = _seconds(DEFAULTS["listener_timeout"])
    assert _hcm(rendered)["stream_idle_timeout"] == listener_timeout
    for route in _routes(rendered):
        assert route["route"]["timeout"] == listener_timeout
        assert "retry_policy" not in route["route"]
    cluster = _cluster(rendered)
    assert cluster["connect_timeout"] == _seconds(DEFAULTS["connect_timeout"])
    assert cluster["lb_policy"] == DEFAULTS["lb_policy"].upper()
    # No circuit_breakers block: Envoy's own defaults are the table's.
    assert "circuit_breakers" not in cluster
    assert "outlier_detection" not in cluster
    assert "health_checks" not in cluster


def test_enabled_reliability_renders_the_shared_defaults(tmp_path, monkeypatch):
    rendered = _render(tmp_path, monkeypatch, _ENABLED)
    for route in _routes(rendered):
        retry = route["route"]["retry_policy"]
        assert retry["retry_on"] == DEFAULTS["retry_on"]
        assert retry["num_retries"] == 1
        assert (
            retry["host_selection_retry_max_attempts"]
            == DEFAULTS["host_selection_retry_max_attempts"]
        )
        # Envoy's own back-off defaults are the table's.
        assert "retry_back_off" not in retry
    cluster = _cluster(rendered)
    assert (
        cluster["least_request_lb_config"]["choice_count"]
        == DEFAULTS["least_request_choice_count"]
    )
    thresholds = cluster["circuit_breakers"]["thresholds"][0]
    assert thresholds["max_connections"] == DEFAULTS["max_connections"]
    assert thresholds["max_pending_requests"] == DEFAULTS["max_pending_requests"]
    assert (
        thresholds["max_requests"] == DEFAULTS["max_requests_with_retries_or_ejection"]
    )
    assert thresholds["retry_budget"] == {
        "budget_percent": {"value": float(DEFAULTS["retry_budget_percent"])},
        "min_retry_concurrency": 5,
    }
    outlier = cluster["outlier_detection"]
    assert outlier["interval"] == _seconds(DEFAULTS["outlier_interval"])
    assert outlier["base_ejection_time"] == _seconds(DEFAULTS["base_ejection_time"])
    assert outlier["max_ejection_percent"] == DEFAULTS["max_ejection_percent"]
    assert "max_ejection_time" not in outlier
    check = cluster["health_checks"][0]
    assert check["interval"] == _seconds(DEFAULTS["health_check_interval"])
    assert check["timeout"] == _seconds(DEFAULTS["health_check_timeout"])
    assert check["unhealthy_threshold"] == DEFAULTS["unhealthy_threshold"]
    assert check["healthy_threshold"] == DEFAULTS["healthy_threshold"]
    assert "no_traffic_interval" not in check
    assert "common_lb_config" not in cluster


def test_timeout_and_retry_fields_render_for_every_route(tmp_path, monkeypatch):
    rendered = _render(
        tmp_path,
        monkeypatch,
        """
providers:
  models:
    - name: m
      reliability:
        retry_count: 2
        retry_on: 5xx,retriable-status-codes
        connect_timeout: 3s
        total_timeout: 2m
        idle_timeout: 45s
        per_try_timeout: 20s
        retriable_status_codes: [429, 503]
        retry_back_off_base: 50ms
        retry_back_off_max: 1s
        retry_after_max: 30s
        retry_budget_percent: 25
      backend_refs:
        - endpoint: 10.0.0.1:8000
          provider: vllm
""",
    )
    routes = _routes(rendered)
    assert len(routes) == 2  # the model's route and the default route
    for route in routes:
        action = route["route"]
        assert action["timeout"] == "120s"
        assert action["idle_timeout"] == "45s"
        retry = action["retry_policy"]
        assert retry["retry_on"] == "5xx,retriable-status-codes"
        assert retry["num_retries"] == 2
        assert retry["per_try_timeout"] == "20s"
        assert retry["retriable_status_codes"] == [429, 503]
        assert retry["retry_back_off"] == {
            "base_interval": "0.05s",
            "max_interval": "1s",
        }
        assert retry["rate_limited_retry_back_off"] == {
            "reset_headers": [{"name": "retry-after", "format": "SECONDS"}],
            "max_interval": "30s",
        }
    cluster = _cluster(rendered)
    assert cluster["connect_timeout"] == "3s"
    assert cluster["circuit_breakers"]["thresholds"][0]["retry_budget"] == {
        "budget_percent": {"value": 25.0},
        "min_retry_concurrency": DEFAULTS["retry_budget_min_concurrency"],
    }


def test_per_try_timeout_alone_renders_a_retry_policy_without_retries(
    tmp_path, monkeypatch
):
    rendered = _render(
        tmp_path,
        monkeypatch,
        """
providers:
  models:
    - name: m
      reliability:
        per_try_timeout: 15s
      backend_refs:
        - endpoint: 10.0.0.1:8000
          provider: vllm
""",
    )
    retry = _routes(rendered)[0]["route"]["retry_policy"]
    assert retry["num_retries"] == 0
    assert retry["per_try_timeout"] == "15s"


def test_first_byte_timeout_is_rejected_for_envoy(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="only in standalone mode"):
        _render(
            tmp_path,
            monkeypatch,
            """
providers:
  models:
    - name: m
      reliability:
        first_byte_timeout: 5s
      backend_refs:
        - endpoint: 10.0.0.1:8000
          provider: vllm
""",
        )


@pytest.mark.parametrize(
    ("go", "envoy"),
    [
        ("30s", "30s"),
        ("25ms", "0.025s"),
        ("1m30s", "90s"),
        ("1.5s", "1.5s"),
        ("2h", "7200s"),
        ("0s", "0s"),
    ],
)
def test_durations_render_in_envoys_seconds_form(go, envoy):
    assert envoy_duration(go) == envoy


@pytest.mark.parametrize("value", ["30", "s", "1x", "1m 30s", ""])
def test_durations_reject_what_go_rejects(value):
    with pytest.raises(ValueError):
        parse_duration(value)
