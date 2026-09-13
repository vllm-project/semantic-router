"""Exercise the parser, Envoy renderer, and local container wiring together."""

import pytest
import yaml
from cli import container_start
from cli.config_generator import (
    ENVOY_CONTAINER_LISTENER_ADDRESS_ENV,
    generate_envoy_config_from_user_config,
)
from cli.parser import parse_user_config
from cli.runtime_stack import resolve_runtime_stack
from cli.validator import validate_user_config


def _write_config(tmp_path, aliases, address="127.0.0.1"):
    document = {
        "version": "v0.3",
        "listeners": [{"name": "local", "address": address, "port": 8899}],
        "providers": {
            "defaults": {"model": aliases[0]},
            "models": [
                {
                    "name": alias,
                    "backend_refs": [
                        {"provider": "vllm", "endpoint": f"127.0.0.1:{8000 + index}"}
                    ],
                }
                for index, alias in enumerate(aliases)
            ],
        },
        "routing": {},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document))
    return path


def _routes(rendered):
    listener = rendered["static_resources"]["listeners"][0]
    hcm = listener["filter_chains"][0]["filters"][0]["typed_config"]
    return hcm["route_config"]["virtual_hosts"][0]["routes"]


def _model_routes(rendered):
    return {
        header["string_match"]["exact"]: route["route"]["cluster"]
        for route in _routes(rendered)
        for header in route["match"].get("headers", [])
        if header["name"] == "x-selected-model"
    }


def test_distinct_aliases_retain_distinct_clusters_and_backend_routes(tmp_path):
    aliases = [
        "model-a",
        "model_a",
        "model/a",
        "model_2fa",
        "model:a",
        "模型",
        'model"quoted',
        "model\\path",
    ]
    path = _write_config(tmp_path, aliases)
    config = parse_user_config(str(path))
    assert not validate_user_config(config, log_summary=False)
    output = tmp_path / "envoy.yaml"
    generate_envoy_config_from_user_config(config, str(output))
    rendered = yaml.safe_load(output.read_text())
    clusters = rendered["static_resources"]["clusters"]
    names = [cluster["name"] for cluster in clusters]
    assert len(names) == len(set(names))
    routes = _model_routes(rendered)
    assert set(routes) == set(aliases)
    assert len(set(routes.values())) == len(aliases)
    for index, alias in enumerate(aliases):
        cluster = next(
            cluster for cluster in clusters if cluster["name"] == routes[alias]
        )
        assert cluster["load_assignment"]["cluster_name"] == routes[alias]
        address = cluster["load_assignment"]["endpoints"][0]["lb_endpoints"][0][
            "endpoint"
        ]["address"]["socket_address"]
        assert address["port_value"] == 8000 + index

    # Adding/reordering aliases must not change an existing resource's identity.
    config.providers.models.reverse()
    generate_envoy_config_from_user_config(config, str(output))
    assert _model_routes(yaml.safe_load(output.read_text())) == routes


@pytest.mark.parametrize("address", ["127.0.0.1", "::1", "0.0.0.0", "::"])
def test_bridge_listener_is_reachable_without_widening_host_publication(
    tmp_path, monkeypatch, address
):
    monkeypatch.delenv(ENVOY_CONTAINER_LISTENER_ADDRESS_ENV, raising=False)
    path = _write_config(tmp_path, ["test-model"], address)
    original = path.read_text()
    config = parse_user_config(str(path))
    layout = resolve_runtime_stack()
    output = tmp_path / "envoy.yaml"
    container_start._render_split_envoy_config(str(path), str(output), layout)
    rendered = yaml.safe_load(output.read_text())
    socket = rendered["static_resources"]["listeners"][0]["address"]["socket_address"]
    assert socket == {"address": "0.0.0.0", "port_value": 8899}
    assert path.read_text() == original

    # Build the real Envoy argv, including the unchanged host address and offset.
    command = container_start._build_envoy_runtime_command(
        runtime="docker",
        envoy_image="envoyproxy/envoy:v1.34-latest",
        nofile_limit=65536,
        runtime_network_name=layout.network_name,
        common_env={},
        listeners=[listener.model_dump() for listener in config.listeners],
        runtime_paths={
            "envoy_config_path": str(output),
            "log_spool_envoy_mount": "/tmp/envoy-logs:/var/log/envoy",
            "log_spool_gid": "1000",
        },
        setup_mode=False,
        stack_layout=layout,
        envoy_log_level="info",
    )
    host_address = f"[{address}]" if ":" in address else address
    assert f"{host_address}:{8899 + layout.port_offset}:8899" in command

    # A normal/native renderer still follows the canonical listener address.
    generate_envoy_config_from_user_config(config, str(output))
    native = yaml.safe_load(output.read_text())
    assert (
        native["static_resources"]["listeners"][0]["address"]["socket_address"][
            "address"
        ]
        == address
    )

    # Dashboard's Python subprocess inherits this environment on later saves.
    dashboard_env = container_start._build_dashboard_runtime_env(
        common_env={}, listener_port=8899, stack_layout=layout
    )
    monkeypatch.setenv(
        ENVOY_CONTAINER_LISTENER_ADDRESS_ENV,
        dashboard_env[ENVOY_CONTAINER_LISTENER_ADDRESS_ENV],
    )
    generate_envoy_config_from_user_config(config, str(output))
    regenerated = yaml.safe_load(output.read_text())
    assert (
        regenerated["static_resources"]["listeners"][0]["address"]["socket_address"][
            "address"
        ]
        == "0.0.0.0"
    )
