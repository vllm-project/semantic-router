import os
import re
import sys
from pathlib import Path

import pytest
import yaml

CLI_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

from cli.config_generator import generate_envoy_config_from_user_config  # noqa: E402
from cli.container_start import _render_split_envoy_config  # noqa: E402
from cli.envoy_dispatch_contract import (  # noqa: E402
    BACKEND_DISPATCH_ADDRESS_ENV,
    INTERNAL_REQUEST_HEADERS,
)
from cli.parser import parse_user_config  # noqa: E402
from cli.runtime_stack import resolve_runtime_stack  # noqa: E402


def _durable_config(*, bind_address="0.0.0.0", port=8187):
    return f"""
version: v0.3
listeners:
  - name: public
    address: 0.0.0.0
    port: 8899
providers:
  models: []
routing:
  modelCards: []
recipes: []
entrypoints: []
global:
  stores:
    management:
      postgres:
        dsn_env: VLLM_SR_ACCESS_DATABASE_URL
    runtime:
      redis:
        url_env: VLLM_SR_ACCESS_RUNTIME_URL
  services:
    agent:
      public_inference_endpoint: http://vllm-sr-envoy-container:8899/v1/chat/completions
    access:
      enabled: false
    backend_dispatch:
      bind_address: {bind_address}
      port: {port}
      audience: vllm-sr.backend-dispatch
      capability_ttl: 30s
      max_request_body_bytes: 67108864
    backend_egress:
      policy_file: /app/config/backend-egress-policy.yaml
    management_api:
      enabled: true
      bind_address: 0.0.0.0
      port: 8080
      auth:
        mode: router
"""


def _render(tmp_path, monkeypatch, config_text, *, dispatch_address=None):
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "envoy.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    monkeypatch.setenv("ENVOY_EXTPROC_ADDRESS", "router.internal")
    if dispatch_address is None:
        monkeypatch.delenv(BACKEND_DISPATCH_ADDRESS_ENV, raising=False)
    else:
        monkeypatch.setenv(BACKEND_DISPATCH_ADDRESS_ENV, dispatch_address)
    config = parse_user_config(str(config_path))
    generate_envoy_config_from_user_config(config, str(output_path))
    return yaml.safe_load(output_path.read_text(encoding="utf-8")), output_path


def _http_connection_manager(rendered):
    listener = rendered["static_resources"]["listeners"][0]
    return listener["filter_chains"][0]["filters"][0]["typed_config"]


def _cluster(rendered, name):
    return next(
        cluster
        for cluster in rendered["static_resources"]["clusters"]
        if cluster["name"] == name
    )


def test_envoy_uses_only_the_stable_backend_dispatch_hop(tmp_path, monkeypatch):
    rendered, output_path = _render(
        tmp_path,
        monkeypatch,
        _durable_config(),
        dispatch_address="router.internal",
    )

    clusters = rendered["static_resources"]["clusters"]
    assert {cluster["name"] for cluster in clusters} == {
        "extproc_service",
        "backend_dispatch_cluster",
    }
    dispatch = _cluster(rendered, "backend_dispatch_cluster")
    assert dispatch["type"] == "LOGICAL_DNS"
    endpoint = dispatch["load_assignment"]["endpoints"][0]["lb_endpoints"][0][
        "endpoint"
    ]
    assert endpoint["address"]["socket_address"] == {
        "address": "router.internal",
        "port_value": 8187,
    }

    hcm = _http_connection_manager(rendered)
    assert hcm["route_config"]["virtual_hosts"][0]["routes"] == [
        {
            "match": {"prefix": "/"},
            "route": {"cluster": "backend_dispatch_cluster", "timeout": "0s"},
        }
    ]
    filters = hcm["http_filters"]
    assert [item["name"] for item in filters] == [
        "envoy.filters.http.lua",
        "envoy.filters.http.ext_proc",
        "envoy.filters.http.router",
    ]
    sanitizer = filters[0]["typed_config"]["default_source_code"]["inline_string"]
    assert set(re.findall(r'"(x-[a-z0-9-]+)"', sanitizer)) == set(
        INTERNAL_REQUEST_HEADERS
    )
    assert filters[1]["typed_config"]["failure_mode_allow"] is False
    assert "retry_policy" not in output_path.read_text(encoding="utf-8")


def test_file_config_can_use_loopback_dispatch_without_override(tmp_path, monkeypatch):
    rendered, _ = _render(
        tmp_path,
        monkeypatch,
        _durable_config(bind_address="127.0.0.1", port=8190),
    )

    dispatch = _cluster(rendered, "backend_dispatch_cluster")
    assert dispatch["type"] == "STATIC"
    socket = dispatch["load_assignment"]["endpoints"][0]["lb_endpoints"][0]["endpoint"][
        "address"
    ]["socket_address"]
    assert socket == {"address": "127.0.0.1", "port_value": 8190}


def test_bind_only_dispatch_requires_a_network_override(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_durable_config(), encoding="utf-8")
    monkeypatch.delenv(BACKEND_DISPATCH_ADDRESS_ENV, raising=False)

    with pytest.raises(ValueError, match="bind-only address"):
        generate_envoy_config_from_user_config(
            parse_user_config(str(config_path)),
            str(tmp_path / "envoy.yaml"),
        )


def test_split_docker_routes_dispatch_to_private_router_hostname(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "envoy.yaml"
    config_path.write_text(_durable_config(), encoding="utf-8")
    monkeypatch.setenv(BACKEND_DISPATCH_ADDRESS_ENV, "original.internal")
    stack = resolve_runtime_stack()

    _render_split_envoy_config(str(config_path), str(output_path), stack)

    rendered = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    endpoint = _cluster(rendered, "backend_dispatch_cluster")["load_assignment"][
        "endpoints"
    ][0]["lb_endpoints"][0]["endpoint"]
    assert (
        endpoint["address"]["socket_address"]["address"] == stack.router_container_name
    )
    assert endpoint["address"]["socket_address"]["port_value"] == 8187
    assert os.environ[BACKEND_DISPATCH_ADDRESS_ENV] == "original.internal"


def test_split_docker_rejects_loopback_dispatch_listener(tmp_path):
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "envoy.yaml"
    config_path.write_text(_durable_config(bind_address="127.0.0.1"), encoding="utf-8")

    with pytest.raises(ValueError, match="wildcard address"):
        _render_split_envoy_config(
            str(config_path), str(output_path), resolve_runtime_stack()
        )


def test_static_local_envoy_uses_the_same_dispatch_contract():
    rendered = yaml.safe_load(
        (REPO_ROOT / "deploy/local/envoy.yaml").read_text(encoding="utf-8")
    )
    assert {
        cluster["name"] for cluster in rendered["static_resources"]["clusters"]
    } == {"extproc_service", "backend_dispatch_cluster"}
