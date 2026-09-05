import sys
from pathlib import Path

import pytest
import yaml

CLI_ROOT = Path(__file__).resolve().parents[1]
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

from cli.config_generator import generate_envoy_config_from_user_config  # noqa: E402
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
    default_model: "model-fast"
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

    # Fast model route and cluster timeouts
    fast_route = _model_route(rendered, "model-fast")["route"]
    assert fast_route["timeout"] == "10s"
    assert fast_route["idleTimeout"] == "3s"
    fast_cluster = _cluster_by_name(rendered, "model_fast_cluster")
    assert fast_cluster["connect_timeout"] == "2s"

    # Reasoning model route and cluster timeouts
    reasoning_route = _model_route(rendered, "model-reasoning")["route"]
    assert reasoning_route["timeout"] == "300s"
    assert reasoning_route["idleTimeout"] == "60s"
    reasoning_cluster = _cluster_by_name(rendered, "model_reasoning_cluster")
    assert reasoning_cluster["connect_timeout"] == "5s"

    # Default timeout model falls back to listener timeout for total deadline and idle timeout
    default_timeout_route = _model_route(rendered, "model-default-timeout")["route"]
    assert default_timeout_route["timeout"] == "600s"
    assert default_timeout_route["idleTimeout"] == "600s"
    default_timeout_cluster = _cluster_by_name(
        rendered, "model_default_timeout_cluster"
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
    default_model: "model-with-retries"
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
    default_model: "model-stream"
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
    default_model: "claude-fast"
  models:
    - name: "claude-fast"
      api_format: "anthropic"
      reliability:
        request_timeout: "15s"
        stream_idle_timeout: "3s"
        connect_timeout: "2s"
    - name: "claude-slow"
      api_format: "anthropic"
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

    fast_route = _model_route(rendered, "claude-fast")["route"]
    assert fast_route["cluster"] == "claude_fast_cluster"
    assert fast_route["timeout"] == "15s"
    assert fast_route["idleTimeout"] == "3s"
    assert fast_route["host_rewrite_literal"] == "api.anthropic.com"

    fast_cluster = _cluster_by_name(rendered, "claude_fast_cluster")
    assert fast_cluster["connect_timeout"] == "2s"
    assert fast_cluster["type"] == "LOGICAL_DNS"

    slow_route = _model_route(rendered, "claude-slow")["route"]
    assert slow_route["cluster"] == "claude_slow_cluster"
    assert slow_route["timeout"] == "120s"
    assert slow_route["idleTimeout"] == "120s"

    slow_cluster = _cluster_by_name(rendered, "claude_slow_cluster")
    assert slow_cluster["connect_timeout"] == "10s"
