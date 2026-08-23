import re
import sys
from pathlib import Path

import yaml

CLI_ROOT = Path(__file__).resolve().parents[1]
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

from cli.config_generator import generate_envoy_config_from_user_config  # noqa: E402
from cli.managed_envoy_contract import INTERNAL_REQUEST_HEADERS  # noqa: E402
from cli.parser import parse_user_config  # noqa: E402


def _standalone_config(*, dispatch_address="127.0.0.1", dispatch_port=8187):
    return f"""
version: v0.4
listeners:
  - name: public
    address: 0.0.0.0
    port: 8899
models:
  - name: private/model
    card:
      description: Private model used by the standalone dispatch test.
      capabilities: [chat]
    runtime:
      max_retries: 3
      request_timeout: 45s
      stream_timeout: 10m
    connections:
      - provider: openai-compatible
        endpoint: https://secret-model-origin.example/v1
        model: upstream-secret-model
        credential: private-token
        weight: "1"
recipes:
  - name: balance
    document:
      decisions:
        - name: Default
entrypoints:
  - name: vllm-sr/blend
    aliases: [blend]
    recipe: balance
    assignments:
      Default:
        models:
          - model: private/model
            weight: "1"
global:
  control_plane:
    mode: standalone
  services:
    backend_dispatch:
      bind_address: {dispatch_address}
      port: {dispatch_port}
      audience: vllm-sr.backend-dispatch
      capability_ttl: 30s
      max_request_body_bytes: 67108864
    backend_egress:
      policy_file: /app/config/backend-egress-policy.yaml
    backend_credentials:
      private-token:
        credential_adapter_id: bearer
        secret_env: PRIVATE_PROVIDER_TOKEN
"""


def _render(
    tmp_path,
    monkeypatch,
    *,
    dispatch_address="127.0.0.1",
    dispatch_upstream=None,
    extproc="127.0.0.1",
):
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "envoy.yaml"
    config_path.write_text(
        _standalone_config(dispatch_address=dispatch_address), encoding="utf-8"
    )
    monkeypatch.setenv("ENVOY_EXTPROC_ADDRESS", extproc)
    if dispatch_upstream is None:
        monkeypatch.delenv("ENVOY_BACKEND_DISPATCH_ADDRESS", raising=False)
    else:
        monkeypatch.setenv("ENVOY_BACKEND_DISPATCH_ADDRESS", dispatch_upstream)
    config = parse_user_config(str(config_path))
    generate_envoy_config_from_user_config(config, str(output_path))
    return yaml.safe_load(output_path.read_text(encoding="utf-8")), output_path


def _cluster(rendered, name):
    return next(
        cluster
        for cluster in rendered["static_resources"]["clusters"]
        if cluster["name"] == name
    )


def _connection_manager(rendered):
    listener = rendered["static_resources"]["listeners"][0]
    return listener["filter_chains"][0]["filters"][0]["typed_config"]


def test_standalone_envoy_has_only_stable_router_owned_upstreams(tmp_path, monkeypatch):
    rendered, output_path = _render(tmp_path, monkeypatch)

    assert {
        cluster["name"] for cluster in rendered["static_resources"]["clusters"]
    } == {
        "extproc_service",
        "backend_dispatch_cluster",
    }
    hcm = _connection_manager(rendered)
    assert hcm["route_config"]["virtual_hosts"][0]["routes"] == [
        {
            "match": {"prefix": "/"},
            "route": {"cluster": "backend_dispatch_cluster", "timeout": "0s"},
        }
    ]

    rendered_text = output_path.read_text(encoding="utf-8")
    for private_value in (
        "secret-model-origin.example",
        "upstream-secret-model",
        "private-token",
        "PRIVATE_PROVIDER_TOKEN",
        "X-Provider-Version",
        "retry_policy",
        "health_checks",
        "fallback:",
    ):
        assert private_value not in rendered_text


def test_standalone_envoy_uses_static_clusters_for_ip_endpoints(tmp_path, monkeypatch):
    rendered, _ = _render(tmp_path, monkeypatch)
    assert _cluster(rendered, "extproc_service")["type"] == "STATIC"
    dispatch = _cluster(rendered, "backend_dispatch_cluster")
    assert dispatch["type"] == "STATIC"
    endpoint = dispatch["load_assignment"]["endpoints"][0]["lb_endpoints"][0][
        "endpoint"
    ]
    assert endpoint["address"]["socket_address"] == {
        "address": "127.0.0.1",
        "port_value": 8187,
    }
    assert "hostname" not in endpoint


def test_standalone_envoy_uses_logical_dns_for_named_services(tmp_path, monkeypatch):
    rendered, _ = _render(
        tmp_path,
        monkeypatch,
        dispatch_address="0.0.0.0",
        dispatch_upstream="router.internal",
        extproc="router.internal",
    )
    for name in ("extproc_service", "backend_dispatch_cluster"):
        cluster = _cluster(rendered, name)
        assert cluster["type"] == "LOGICAL_DNS"
        assert cluster["dns_lookup_family"] == "V4_ONLY"
        endpoint = cluster["load_assignment"]["endpoints"][0]["lb_endpoints"][0][
            "endpoint"
        ]
        assert endpoint["hostname"] == "router.internal"


def test_public_internal_headers_are_removed_before_extproc(tmp_path, monkeypatch):
    rendered, _ = _render(tmp_path, monkeypatch)
    filters = _connection_manager(rendered)["http_filters"]
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


def test_multiple_public_listeners_share_one_dispatch_cluster(tmp_path, monkeypatch):
    config = yaml.safe_load(_standalone_config())
    config["listeners"].append(
        {"name": "secondary", "address": "127.0.0.1", "port": 8900}
    )
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "envoy.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("ENVOY_EXTPROC_ADDRESS", "127.0.0.1")
    monkeypatch.delenv("ENVOY_BACKEND_DISPATCH_ADDRESS", raising=False)

    generate_envoy_config_from_user_config(
        parse_user_config(str(config_path)), str(output_path)
    )
    rendered = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    assert len(rendered["static_resources"]["listeners"]) == 2
    assert [
        cluster["name"] for cluster in rendered["static_resources"]["clusters"]
    ].count("backend_dispatch_cluster") == 1
