import re
import sys
from pathlib import Path

import pytest
import yaml

CLI_ROOT = Path(__file__).resolve().parents[1]
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

from cli.config_generator import (  # noqa: E402
    _model_cluster_id,
    generate_envoy_config_from_user_config,
)
from cli.models import ProviderReliability  # noqa: E402
from cli.parser import parse_user_config  # noqa: E402


def _render_envoy_config(
    tmp_path, monkeypatch, config_text, *, extproc_host, router_api_host
):
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "envoy.yaml"
    config_path.write_text(config_text)

    monkeypatch.setenv("ENVOY_EXTPROC_ADDRESS", extproc_host)
    monkeypatch.setenv("ENVOY_ROUTER_API_ADDRESS", router_api_host)

    user_config = parse_user_config(str(config_path))
    generate_envoy_config_from_user_config(user_config, str(output_path))
    return yaml.safe_load(output_path.read_text())


def _cluster_by_name(rendered_config, cluster_name):
    for cluster in rendered_config["static_resources"]["clusters"]:
        if cluster["name"] == cluster_name:
            return cluster
    raise AssertionError(f"cluster {cluster_name!r} not found")


def _model_route(rendered_config, model_name):
    """Find the route entry whose x-selected-model header matches *model_name*."""
    listener = rendered_config["static_resources"]["listeners"][0]
    hcm = listener["filter_chains"][0]["filters"][0]["typed_config"]
    routes = hcm["route_config"]["virtual_hosts"][0]["routes"]
    for route in routes:
        headers = route.get("match", {}).get("headers", [])
        for h in headers:
            if (
                h.get("name") == "x-selected-model"
                and h.get("string_match", {}).get("exact") == model_name
            ):
                return route
    raise AssertionError(f"route for model {model_name!r} not found")


def _default_route(rendered_config):
    """Find the fallback route without an x-selected-model header match."""
    listener = rendered_config["static_resources"]["listeners"][0]
    hcm = listener["filter_chains"][0]["filters"][0]["typed_config"]
    routes = hcm["route_config"]["virtual_hosts"][0]["routes"]
    for route in reversed(routes):
        if not route.get("match", {}).get("headers"):
            return route
    raise AssertionError("default route not found")


def test_provider_reliability_renders_custom_timeouts_for_multiple_models(
    tmp_path, monkeypatch
):
    rendered = _render_envoy_config(
        tmp_path,
        monkeypatch,
        """
version: v0.3
listeners:
  - name: "http-8899"
    address: "0.0.0.0"
    port: 8899
    timeout: "600s"
providers:
  defaults:
    model: "model-fast"
  models:
    - name: "model-fast"
      reliability:
        request_timeout: "10s"
        stream_idle_timeout: "3s"
        connect_timeout: "2s"
      backend_refs:
        - endpoint: "10.0.0.1:8000"
    - name: "model-reasoning"
      reliability:
        request_timeout: "300s"
        stream_idle_timeout: "60s"
        connect_timeout: "5s"
      backend_refs:
        - endpoint: "10.0.0.2:8000"
    - name: "model-default-timeout"
      backend_refs:
        - endpoint: "10.0.0.3:8000"
routing:
  modelCards:
    - name: "model-fast"
    - name: "model-reasoning"
    - name: "model-default-timeout"
  decisions:
    - name: "default-route"
      priority: 100
      rules:
        operator: "AND"
      modelRefs:
        - model: "model-fast"
""",
        extproc_host="localhost",
        router_api_host="localhost",
    )

    listener = rendered["static_resources"]["listeners"][0]
    hcm = listener["filter_chains"][0]["filters"][0]["typed_config"]
    assert hcm["stream_idle_timeout"] == "600s"

    # Fast model route and cluster timeouts
    fast_route = _model_route(rendered, "model-fast")["route"]
    assert fast_route["cluster"] == _model_cluster_id("model-fast") + "_cluster"
    assert fast_route["timeout"] == "10s"
    assert fast_route["idleTimeout"] == "3s"
    fast_cluster = _cluster_by_name(
        rendered, _model_cluster_id("model-fast") + "_cluster"
    )
    assert fast_cluster["connect_timeout"] == "2s"

    # Reasoning model route and cluster timeouts
    reasoning_route = _model_route(rendered, "model-reasoning")["route"]
    assert (
        reasoning_route["cluster"] == _model_cluster_id("model-reasoning") + "_cluster"
    )
    assert reasoning_route["timeout"] == "300s"
    assert reasoning_route["idleTimeout"] == "60s"
    reasoning_cluster = _cluster_by_name(
        rendered, _model_cluster_id("model-reasoning") + "_cluster"
    )
    assert reasoning_cluster["connect_timeout"] == "5s"

    # Default timeout model falls back to listener timeout for total deadline;
    # stream_idle_timeout is omitted on route to inherit HCM stream_idle_timeout.
    default_timeout_route = _model_route(rendered, "model-default-timeout")["route"]
    assert (
        default_timeout_route["cluster"]
        == _model_cluster_id("model-default-timeout") + "_cluster"
    )
    assert default_timeout_route["timeout"] == "600s"
    assert "idleTimeout" not in default_timeout_route
    assert (
        default_timeout_route.get("idleTimeout", hcm.get("stream_idle_timeout"))
        == "600s"
    )
    default_timeout_cluster = _cluster_by_name(
        rendered, _model_cluster_id("model-default-timeout") + "_cluster"
    )
    assert default_timeout_cluster["connect_timeout"] == "10s"


def test_default_route_does_not_borrow_model_reliability(tmp_path, monkeypatch):
    rendered = _render_envoy_config(
        tmp_path,
        monkeypatch,
        """
version: v0.3
listeners:
  - name: "http-8899"
    address: "0.0.0.0"
    port: 8899
    timeout: "300s"
providers:
  defaults:
    model: "model-with-retries"
  models:
    - name: "model-with-retries"
      reliability:
        retry_count: 3
        retry_on: "5xx,connect-failure"
        request_timeout: "15s"
        stream_idle_timeout: "5s"
      backend_refs:
        - endpoint: "10.0.0.1:8000"
routing:
  modelCards:
    - name: "model-with-retries"
  decisions:
    - name: "default-route"
      priority: 100
      rules:
        operator: "AND"
      modelRefs:
        - model: "model-with-retries"
""",
        extproc_host="localhost",
        router_api_host="localhost",
    )

    model_route = _model_route(rendered, "model-with-retries")["route"]
    assert "retry_policy" in model_route
    assert model_route["retry_policy"]["num_retries"] == 3
    assert model_route["timeout"] == "15s"
    assert model_route["idleTimeout"] == "5s"

    def_route = _default_route(rendered)["route"]
    assert "retry_policy" not in def_route
    assert def_route["timeout"] == "300s"
    assert def_route["idleTimeout"] == "300s"


def test_provider_reliability_allows_zero_request_timeout_with_stream_idle_timeout(
    tmp_path, monkeypatch
):
    rendered = _render_envoy_config(
        tmp_path,
        monkeypatch,
        """
version: v0.3
listeners:
  - name: "http-8899"
    address: "0.0.0.0"
    port: 8899
providers:
  defaults:
    model: "model-stream"
  models:
    - name: "model-stream"
      reliability:
        request_timeout: "0s"
        stream_idle_timeout: "30s"
      backend_refs:
        - endpoint: "10.0.0.1:8000"
routing:
  modelCards:
    - name: "model-stream"
  decisions:
    - name: "default-route"
      priority: 100
      rules:
        operator: "AND"
      modelRefs:
        - model: "model-stream"
""",
        extproc_host="localhost",
        router_api_host="localhost",
    )

    route = _model_route(rendered, "model-stream")["route"]
    assert route["timeout"] == "0s"
    assert route["idleTimeout"] == "30s"


def test_provider_reliability_validation():
    # Valid configurations
    rel = ProviderReliability(
        request_timeout="60s",
        stream_idle_timeout="10s",
        connect_timeout="5s",
    )
    assert rel.request_timeout == "60s"
    assert rel.stream_idle_timeout == "10s"
    assert rel.connect_timeout == "5s"

    # Zero request_timeout with positive stream_idle_timeout is valid
    rel_zero = ProviderReliability(request_timeout="0s", stream_idle_timeout="15s")
    assert rel_zero.request_timeout == "0s"

    # Zero request_timeout without stream_idle_timeout is invalid
    with pytest.raises(
        ValueError,
        match="request_timeout cannot be 0 without a positive stream_idle_timeout",
    ):
        ProviderReliability(request_timeout="0s")

    # Zero request_timeout with zero stream_idle_timeout is invalid
    with pytest.raises(
        ValueError,
        match="request_timeout cannot be 0 without a positive stream_idle_timeout",
    ):
        ProviderReliability(request_timeout="0s", stream_idle_timeout="0s")

    # Connect timeout <= 0 is invalid
    with pytest.raises(ValueError, match="connect_timeout must be greater than 0"):
        ProviderReliability(connect_timeout="0s")

    # Invalid duration strings
    with pytest.raises(ValueError, match="invalid duration format"):
        ProviderReliability(request_timeout="invalid")

    # Sub-second and compound duration units normalize to protobuf seconds
    rel_sub = ProviderReliability(
        request_timeout="300ms",
        stream_idle_timeout="100ms",
        connect_timeout="500ms",
    )
    assert rel_sub.request_timeout == "0.3s"
    assert rel_sub.stream_idle_timeout == "0.1s"
    assert rel_sub.connect_timeout == "0.5s"

    rel_units = ProviderReliability(
        request_timeout="2m",
        stream_idle_timeout="1.5s",
        connect_timeout="100us",
    )
    assert rel_units.request_timeout == "120s"
    assert rel_units.stream_idle_timeout == "1.5s"
    assert rel_units.connect_timeout == "0.0001s"


def test_anthropic_model_renders_policy_correct_cluster_timeouts(tmp_path, monkeypatch):
    rendered = _render_envoy_config(
        tmp_path,
        monkeypatch,
        """
version: v0.3
listeners:
  - name: "http-8899"
    address: "0.0.0.0"
    port: 8899
    timeout: "120s"
providers:
  defaults:
    model: "claude-fast"
  models:
    - name: "claude-fast"
      api_format: "anthropic"
      backend_refs:
        - name: "anthropic-fast"
          provider: "anthropic"
      reliability:
        request_timeout: "15s"
        stream_idle_timeout: "3s"
        connect_timeout: "2s"
    - name: "claude-slow"
      api_format: "anthropic"
      backend_refs:
        - name: "anthropic-slow"
          provider: "anthropic"
routing:
  modelCards:
    - name: "claude-fast"
    - name: "claude-slow"
  decisions:
    - name: "default-route"
      priority: 100
      rules:
        operator: "AND"
      modelRefs:
        - model: "claude-fast"
""",
        extproc_host="localhost",
        router_api_host="localhost",
    )

    listener = rendered["static_resources"]["listeners"][0]
    hcm = listener["filter_chains"][0]["filters"][0]["typed_config"]
    assert hcm["stream_idle_timeout"] == "120s"

    fast_route = _model_route(rendered, "claude-fast")["route"]
    assert fast_route["cluster"] == _model_cluster_id("claude-fast") + "_cluster"
    assert fast_route["timeout"] == "15s"
    assert fast_route["idleTimeout"] == "3s"
    assert fast_route["host_rewrite_literal"] == "api.anthropic.com"

    fast_cluster = _cluster_by_name(
        rendered, _model_cluster_id("claude-fast") + "_cluster"
    )
    assert fast_cluster["connect_timeout"] == "2s"
    assert fast_cluster["type"] == "LOGICAL_DNS"

    slow_route = _model_route(rendered, "claude-slow")["route"]
    assert slow_route["cluster"] == _model_cluster_id("claude-slow") + "_cluster"
    assert slow_route["timeout"] == "120s"
    assert "idleTimeout" not in slow_route
    assert slow_route.get("idleTimeout", hcm.get("stream_idle_timeout")) == "120s"

    slow_cluster = _cluster_by_name(
        rendered, _model_cluster_id("claude-slow") + "_cluster"
    )
    assert slow_cluster["connect_timeout"] == "10s"


_PROTOBUF_DURATION_REGEX = re.compile(r"^[0-9]+(?:\.[0-9]{1,9})?s$")


def _assert_valid_protobuf_duration(val: str):
    """Assert that *val* strictly adheres to the protobuf Duration string format.

    google.protobuf.Duration JSON mapping requires a decimal number of seconds ending
    with a single 's', with up to 9 fractional digits (nanosecond resolution).
    """
    assert isinstance(val, str), f"expected duration string, got {type(val)!r}"
    assert _PROTOBUF_DURATION_REGEX.match(
        val
    ), f"duration {val!r} does not conform to protobuf Duration format (must be seconds ending with 's')"
    assert not any(
        val.endswith(u) for u in ("ms", "us", "ns", "m", "h")
    ), f"duration {val!r} contains non-second unit suffix"
    assert float(val[:-1]) >= 0.0


def test_envoy_config_validates_and_normalizes_durations(tmp_path, monkeypatch):
    """Verify that non-seconds/sub-second durations are normalized to protobuf Duration
    format (ending with 's') and pass strict google.protobuf Duration validation."""
    rendered = _render_envoy_config(
        tmp_path,
        monkeypatch,
        """
version: v0.3
listeners:
  - name: "http-8899"
    address: "0.0.0.0"
    port: 8899
    timeout: "5m"
providers:
  defaults:
    model: "model-subsecond"
  models:
    - name: "model-subsecond"
      reliability:
        request_timeout: "300ms"
        stream_idle_timeout: "100ms"
        connect_timeout: "500ms"
      backend_refs:
        - endpoint: "10.0.0.1:8000"
    - name: "model-mixed"
      reliability:
        request_timeout: "1.5s"
        stream_idle_timeout: "250ms"
        connect_timeout: "2m"
      backend_refs:
        - endpoint: "10.0.0.2:8000"
routing:
  modelCards:
    - name: "model-subsecond"
    - name: "model-mixed"
  decisions:
    - name: "default-route"
      priority: 100
      rules:
        operator: "AND"
      modelRefs:
        - model: "model-subsecond"
""",
        extproc_host="localhost",
        router_api_host="localhost",
    )

    listener = rendered["static_resources"]["listeners"][0]
    hcm = listener["filter_chains"][0]["filters"][0]["typed_config"]
    assert hcm["stream_idle_timeout"] == "300s"

    # Verify model-subsecond timeouts are normalized to protobuf Duration format
    sub_route = _model_route(rendered, "model-subsecond")["route"]
    assert sub_route["timeout"] == "0.3s"
    assert sub_route["idleTimeout"] == "0.1s"
    sub_cluster = _cluster_by_name(
        rendered, _model_cluster_id("model-subsecond") + "_cluster"
    )
    assert sub_cluster["connect_timeout"] == "0.5s"

    # Verify model-mixed timeouts
    mixed_route = _model_route(rendered, "model-mixed")["route"]
    assert mixed_route["timeout"] == "1.5s"
    assert mixed_route["idleTimeout"] == "0.25s"
    mixed_cluster = _cluster_by_name(
        rendered, _model_cluster_id("model-mixed") + "_cluster"
    )
    assert mixed_cluster["connect_timeout"] == "120s"

    # Conformance check: all duration fields in Envoy routes and clusters must strictly
    # conform to the protobuf Duration specification without error.
    for route in hcm["route_config"]["virtual_hosts"][0]["routes"]:
        action = route.get("route", {})
        for timeout_key in ("timeout", "idleTimeout", "idle_timeout"):
            if timeout_key in action:
                _assert_valid_protobuf_duration(str(action[timeout_key]))

    for cluster in rendered["static_resources"]["clusters"]:
        if "connect_timeout" in cluster:
            _assert_valid_protobuf_duration(str(cluster["connect_timeout"]))
