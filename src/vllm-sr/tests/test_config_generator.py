import re
import sys
from pathlib import Path

import pytest
import yaml

CLI_ROOT = Path(__file__).resolve().parents[1]
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

from cli.config_generator import generate_envoy_config_from_user_config  # noqa: E402
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


def test_generate_envoy_config_uses_logical_dns_for_split_extproc_host(
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
    default_model: "test-model"
  models:
    - name: "test-model"
      backend_refs:
        - name: "primary"
          endpoint: "host.docker.internal:8000"
          protocol: "http"
          weight: 100
routing:
  modelCards:
    - name: "test-model"
  decisions:
    - name: "default-route"
      description: "default route"
      priority: 100
      rules:
        operator: "AND"
        conditions: []
      modelRefs:
        - model: "test-model"
          use_reasoning: false
""",
        extproc_host="vllm-sr-router-container",
        router_api_host="vllm-sr-router-container",
    )

    cluster = _cluster_by_name(rendered, "extproc_service")

    assert cluster["type"] == "LOGICAL_DNS"
    assert cluster["dns_lookup_family"] == "V4_ONLY"
    endpoint = cluster["load_assignment"]["endpoints"][0]["lb_endpoints"][0]["endpoint"]
    assert (
        endpoint["address"]["socket_address"]["address"] == "vllm-sr-router-container"
    )
    assert endpoint["hostname"] == "vllm-sr-router-container"


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


def test_backend_ref_ip_port_path_produces_correct_envoy_cluster_and_route(
    tmp_path, monkeypatch
):
    """Backend ref http://10.0.0.1:8000/v1 should split into address=10.0.0.1,
    port=8000, host_authority=10.0.0.1:8000, path_prefix=/v1, and the route
    should use the segment-anchored regex ^/v1/(.*)$ so the rewrite is
    idempotent and /v1 is never duplicated."""
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
    default_model: "test-model"
  models:
    - name: "test-model"
      backend_refs:
        - name: "primary"
          endpoint: "http://10.0.0.1:8000/v1"
          weight: 100
routing:
  modelCards:
    - name: "test-model"
  decisions:
    - name: "default-route"
      description: "default route"
      priority: 100
      rules:
        operator: "AND"
        conditions: []
      modelRefs:
        - model: "test-model"
          use_reasoning: false
""",
        extproc_host="localhost",
        router_api_host="localhost",
    )

    # --- cluster assertions ---
    cluster = _cluster_by_name(rendered, "test_model_cluster")
    assert cluster["connect_timeout"] == "10s"
    assert cluster["type"] == "STATIC"
    ep = cluster["load_assignment"]["endpoints"][0]["lb_endpoints"][0]["endpoint"]
    assert ep["address"]["socket_address"]["address"] == "10.0.0.1"
    assert ep["address"]["socket_address"]["port_value"] == 8000

    # --- route assertions ---
    route = _model_route(rendered, "test-model")
    route_action = route["route"]
    assert route_action["host_rewrite_literal"] == "10.0.0.1:8000"
    assert route_action["regex_rewrite"]["pattern"]["regex"] == "^/v1/(.*)$"
    assert route_action["regex_rewrite"]["substitution"] == "/v1/\\1"


def test_provider_reliability_renders_retry_outlier_and_least_request(
    tmp_path, monkeypatch
):
    rendered = _render_envoy_config(
        tmp_path,
        monkeypatch,
        """
version: v0.3
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    default_model: test-model
  models:
    - name: test-model
      reliability:
        lb_policy: least_request
        retry_count: 2
        retry_on: connect-failure,refused-stream
        consecutive_5xx: 5
        base_ejection_time: 45s
        max_ejection_percent: 25
        health_check_path: /health
        health_check_interval: 15s
        health_check_timeout: 3s
      backend_refs:
        - endpoint: 10.0.0.1:8000
        - endpoint: 10.0.0.2:8000
routing:
  modelCards:
    - name: test-model
  decisions:
    - name: default-route
      description: default route
      priority: 100
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: test-model
""",
        extproc_host="localhost",
        router_api_host="localhost",
    )

    route = _model_route(rendered, "test-model")["route"]
    assert route["retry_policy"] == {
        "retry_on": "connect-failure,refused-stream",
        "num_retries": 2,
    }
    cluster = _cluster_by_name(rendered, "test_model_cluster")
    assert cluster["lb_policy"] == "LEAST_REQUEST"
    assert cluster["least_request_lb_config"]["choice_count"] == 2
    assert cluster["outlier_detection"]["consecutive_5xx"] == 5
    assert cluster["outlier_detection"]["base_ejection_time"] == "45s"
    assert cluster["outlier_detection"]["max_ejection_percent"] == 25
    assert cluster["health_checks"][0]["http_health_check"]["path"] == "/health"
    assert cluster["health_checks"][0]["interval"] == "15s"
    assert cluster["health_checks"][0]["timeout"] == "3s"
    assert cluster["circuit_breakers"]["thresholds"][0]["max_requests"] == 4096


def test_backend_ref_domain_with_path_produces_correct_envoy_cluster_and_route(
    tmp_path, monkeypatch
):
    """Backend ref https://api.example.com/compatible-mode/v1 should produce
    address=api.example.com, port=443, host_authority=api.example.com (standard
    port omitted), LOGICAL_DNS cluster, and regex_rewrite for path prefix."""
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
    default_model: "test-model"
  models:
    - name: "test-model"
      backend_refs:
        - name: "primary"
          endpoint: "https://api.example.com/compatible-mode/v1/"
          weight: 100
routing:
  modelCards:
    - name: "test-model"
  decisions:
    - name: "default-route"
      description: "default route"
      priority: 100
      rules:
        operator: "AND"
        conditions: []
      modelRefs:
        - model: "test-model"
          use_reasoning: false
""",
        extproc_host="localhost",
        router_api_host="localhost",
    )

    # --- cluster assertions ---
    cluster = _cluster_by_name(rendered, "test_model_cluster")
    assert cluster["type"] == "LOGICAL_DNS"
    assert cluster["dns_lookup_family"] == "V4_ONLY"
    ep = cluster["load_assignment"]["endpoints"][0]["lb_endpoints"][0]["endpoint"]
    assert ep["address"]["socket_address"]["address"] == "api.example.com"
    assert ep["address"]["socket_address"]["port_value"] == 443
    assert ep["hostname"] == "api.example.com"

    # --- route assertions ---
    route = _model_route(rendered, "test-model")
    route_action = route["route"]
    # standard port 443 → host_authority should omit port
    assert route_action["host_rewrite_literal"] == "api.example.com"
    assert route_action["regex_rewrite"]["pattern"]["regex"] == "^/v1/(.*)$"
    assert route_action["regex_rewrite"]["substitution"] == "/compatible-mode/v1/\\1"


def test_backend_ref_https_base_url_uses_tls_and_explicit_extra_headers(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-openrouter")
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
    default_model: "test-model"
  models:
    - name: "test-model"
      provider_model_id: "openai/gpt-4o-mini"
      backend_refs:
        - name: "openrouter"
          base_url: "https://openrouter.ai/api/v1"
          provider: "openai"
          auth_header: "Authorization"
          auth_prefix: "Bearer"
          api_key_env: "OPENROUTER_API_KEY"
          extra_headers:
            X-Test-Trace: "router-flow"
            X-Test-Tenant: "eval"
          weight: 1
routing:
  modelCards:
    - name: "test-model"
  decisions:
    - name: "default-route"
      description: "default route"
      priority: 100
      rules:
        operator: "AND"
        conditions: []
      modelRefs:
        - model: "test-model"
          use_reasoning: false
""",
        extproc_host="localhost",
        router_api_host="localhost",
    )

    cluster = _cluster_by_name(rendered, "test_model_cluster")
    assert cluster["type"] == "LOGICAL_DNS"
    assert cluster["transport_socket"]["name"] == "envoy.transport_sockets.tls"

    route = _model_route(rendered, "test-model")
    route_action = route["route"]
    assert route_action["host_rewrite_literal"] == "openrouter.ai"
    assert route_action["regex_rewrite"]["substitution"] == "/api/v1/\\1"

    headers = {
        item["header"]["key"]: item["header"]["value"]
        for item in route["request_headers_to_add"]
    }
    assert headers["Authorization"] == "Bearer sk-test-openrouter"
    assert headers["X-Test-Trace"] == "router-flow"
    assert headers["X-Test-Tenant"] == "eval"


def test_generate_envoy_config_custom_anthropic_upstream_rewrites_host(
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
    default_model: "claude-sonnet-4.6"
  models:
    - name: "claude-sonnet-4.6"
      api_format: "anthropic"
      backend_refs:
        - name: "anthropic-primary"
          endpoint: "domain.com:443"
          protocol: "https"
          weight: 100
          base_url: "https://domain.com/Anthropic"
          type: "anthropic"
          provider: "anthropic"
routing:
  modelCards:
    - name: "claude-sonnet-4.6"
  decisions:
    - name: "default-route"
      description: "default route"
      priority: 100
      rules:
        operator: "AND"
        conditions: []
      modelRefs:
        - model: "claude-sonnet-4.6"
          use_reasoning: false
""",
        extproc_host="vllm-sr-router-container",
        router_api_host="vllm-sr-router-container",
    )

    route = _model_route(rendered, "claude-sonnet-4.6")
    assert route["route"]["cluster"] == "claude_sonnet_4.6_cluster"
    assert route["route"]["host_rewrite_literal"] == "domain.com"

    with pytest.raises(AssertionError):
        _cluster_by_name(rendered, "anthropic_api_cluster")


def test_generate_envoy_config_uses_logical_dns_for_api_only_router_fallback(
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
    default_model: "claude-test"
  models:
    - name: "claude-test"
      api_format: "anthropic"
routing:
  modelCards:
    - name: "claude-test"
  decisions:
    - name: "default-route"
      description: "default route"
      priority: 100
      rules:
        operator: "AND"
        conditions: []
      modelRefs:
        - model: "claude-test"
          use_reasoning: false
""",
        extproc_host="vllm-sr-router-container",
        router_api_host="vllm-sr-router-container",
    )

    cluster = _cluster_by_name(rendered, "vllm_static_cluster")

    assert cluster["type"] == "LOGICAL_DNS"
    assert cluster["dns_lookup_family"] == "V4_ONLY"
    endpoint = cluster["load_assignment"]["endpoints"][0]["lb_endpoints"][0]["endpoint"]
    assert (
        endpoint["address"]["socket_address"]["address"] == "vllm-sr-router-container"
    )
    assert endpoint["hostname"] == "vllm-sr-router-container"


def test_backend_ref_v1_prefixed_base_url_rewrite_is_idempotent(tmp_path, monkeypatch):
    """A backend whose base_url path starts with /v1 (e.g. Gemini's
    https://generativelanguage.googleapis.com/v1beta/openai) must generate a
    rewrite that is safe to apply twice: when ext_proc mutates the headers and
    Envoy re-runs route selection, the already-rewritten path must not match
    the pattern again and get the prefix doubled."""
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
    default_model: "gemini-3.5-flash"
  models:
    - name: "gemini-3.5-flash"
      backend_refs:
        - name: "primary"
          base_url: "https://generativelanguage.googleapis.com/v1beta/openai"
          provider: "openai"
          weight: 100
routing:
  modelCards:
    - name: "gemini-3.5-flash"
  decisions:
    - name: "default-route"
      description: "default route"
      priority: 100
      rules:
        operator: "AND"
        conditions: []
      modelRefs:
        - model: "gemini-3.5-flash"
          use_reasoning: false
""",
        extproc_host="localhost",
        router_api_host="localhost",
    )

    model_route = _model_route(rendered, "gemini-3.5-flash")["route"]
    rewrite = model_route["regex_rewrite"]
    regex = rewrite["pattern"]["regex"]
    substitution = rewrite["substitution"]
    assert regex == "^/v1/(.*)$"
    assert substitution == "/v1beta/openai/\\1"

    # default_model points at the same backend, so the default route carries an
    # identical rewrite block; it must be anchored the same way.
    routes = rendered["static_resources"]["listeners"][0]["filter_chains"][0][
        "filters"
    ][0]["typed_config"]["route_config"]["virtual_hosts"][0]["routes"]
    default_route = routes[-1]["route"]
    assert default_route["regex_rewrite"]["pattern"]["regex"] == "^/v1/(.*)$"
    assert default_route["regex_rewrite"]["substitution"] == "/v1beta/openai/\\1"

    # Simulate Envoy applying the rewrite a second time after route recompute:
    # the second pass must be a no-op, not a doubled prefix.
    first_pass = re.sub(regex, substitution, "/v1/chat/completions")
    assert first_pass == "/v1beta/openai/chat/completions"
    assert re.sub(regex, substitution, first_pass) == first_pass


def test_backend_ref_non_v1_prefix_rewrite_not_applied_twice(tmp_path, monkeypatch):
    """Prefixes that do not start with /v1 (e.g. Azure openai/v1) must keep
    the old anchored behavior: the rewritten path can never match the pattern
    again, so the prefix is applied exactly once even after route recompute."""
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
    default_model: "azure-openai-gpt-4o"
  models:
    - name: "azure-openai-gpt-4o"
      backend_refs:
        - name: "primary"
          base_url: "https://my-azure.openai.azure.com/openai/v1"
          provider: "openai"
          weight: 100
routing:
  modelCards:
    - name: "azure-openai-gpt-4o"
  decisions:
    - name: "default-route"
      description: "default route"
      priority: 100
      rules:
        operator: "AND"
        conditions: []
      modelRefs:
        - model: "azure-openai-gpt-4o"
          use_reasoning: false
""",
        extproc_host="localhost",
        router_api_host="localhost",
    )

    model_route = _model_route(rendered, "azure-openai-gpt-4o")["route"]
    rewrite = model_route["regex_rewrite"]
    regex = rewrite["pattern"]["regex"]
    substitution = rewrite["substitution"]
    assert regex == "^/v1/(.*)$"
    assert substitution == "/openai/v1/\\1"

    # A second application of the anchored pattern must not double the prefix.
    first_pass = re.sub(regex, substitution, "/v1/chat/completions")
    assert first_pass == "/openai/v1/chat/completions"
    assert re.sub(regex, substitution, first_pass) == first_pass
