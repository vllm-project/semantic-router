import sys
from pathlib import Path

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
            if h.get("name") == "x-selected-model":
                if h.get("string_match", {}).get("exact") == model_name:
                    return route
    raise AssertionError(f"route for model {model_name!r} not found")


def test_backend_ref_ip_port_path_produces_correct_envoy_cluster_and_route(
    tmp_path, monkeypatch
):
    """Backend ref http://10.0.0.1:8000/v1 should split into address=10.0.0.1,
    port=8000, host_authority=10.0.0.1:8000, path_prefix=/v1, and the route
    should use regex ^/v1(.*)$ to avoid duplicating /v1."""
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
    assert cluster["type"] == "STATIC"
    ep = cluster["load_assignment"]["endpoints"][0]["lb_endpoints"][0]["endpoint"]
    assert ep["address"]["socket_address"]["address"] == "10.0.0.1"
    assert ep["address"]["socket_address"]["port_value"] == 8000

    # --- route assertions ---
    route = _model_route(rendered, "test-model")
    route_action = route["route"]
    assert route_action["host_rewrite_literal"] == "10.0.0.1:8000"
    assert route_action["regex_rewrite"]["pattern"]["regex"] == "^/v1(.*)$"
    assert route_action["regex_rewrite"]["substitution"] == "/v1\\1"


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
    assert route_action["regex_rewrite"]["pattern"]["regex"] == "^/v1(.*)$"
    assert route_action["regex_rewrite"]["substitution"] == "/compatible-mode/v1\\1"


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
