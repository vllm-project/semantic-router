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
