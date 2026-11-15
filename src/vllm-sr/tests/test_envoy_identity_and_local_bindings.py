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
        "version": "v0.5",
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
        "recipes": [{"name": "default"}],
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
        if header["name"] == "x-vsr-route-model"
    }


# The Router reads a listener timeout as a Go duration; Envoy takes seconds.
@pytest.mark.parametrize(
    ("configured", "timeout"),
    [
        ("0s", "0s"),
        ("300s", "300s"),
        ("0", "0s"),
        ("5m", "300s"),
        ("1m30s", "90s"),
        ("500ms", "0.5s"),
    ],
)
def test_chat_duration_is_independent_of_extproc_message_timeout(
    tmp_path, configured, timeout
):
    path = _write_config(tmp_path, ["model-a", "model-b"])
    document = yaml.safe_load(path.read_text())
    document["listeners"][0]["timeout"] = configured
    path.write_text(yaml.safe_dump(document))
    config = parse_user_config(str(path))
    assert not validate_user_config(config, log_summary=False)
    output = tmp_path / "envoy.yaml"
    generate_envoy_config_from_user_config(config, str(output))
    rendered = yaml.safe_load(output.read_text())
    listener = rendered["static_resources"]["listeners"][0]
    hcm = listener["filter_chains"][0]["filters"][0]["typed_config"]
    assert hcm["stream_idle_timeout"] == timeout
    assert all(route["route"]["timeout"] == timeout for route in _routes(rendered))
    for route in _routes(rendered):
        action = route["route"]
        assert (
            action.get(
                "idle_timeout", action.get("idleTimeout", hcm["stream_idle_timeout"])
            )
            == timeout
        )
    extproc = next(
        item["typed_config"]
        for item in hcm["http_filters"]
        if item["name"] == "envoy.filters.http.ext_proc"
    )
    # An unlimited chat must still give the processor time to answer each
    # message, and its gRPC stream must not add a hidden whole-chat deadline.
    assert extproc["message_timeout"] == "1200s"
    assert "timeout" not in extproc["grpc_service"]


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
        envoy_image="envoyproxy/envoy:v1.35.3",
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


@pytest.mark.parametrize(
    ("configured", "stripped"),
    [
        (None, ["x-vsr-delegated-credential"]),
        (
            " X-Gateway-Token ",
            ["x-vsr-delegated-credential", "x-gateway-token"],
        ),
    ],
)
def test_delegated_credential_is_stripped_ahead_of_extproc(
    tmp_path, configured, stripped
):
    path = _write_config(tmp_path, ["model-a"])
    document = yaml.safe_load(path.read_text())
    document["listeners"][0]["api_keys"] = ["listener-key"]
    if configured is not None:
        document["global"] = {
            "services": {"authz": {"delegated_credential": {"header": configured}}}
        }
    path.write_text(yaml.safe_dump(document))
    config = parse_user_config(str(path))
    output = tmp_path / "envoy.yaml"
    generate_envoy_config_from_user_config(config, str(output))
    rendered = yaml.safe_load(output.read_text())
    hcm = rendered["static_resources"]["listeners"][0]["filter_chains"][0]["filters"][
        0
    ]["typed_config"]
    filters = hcm["http_filters"]
    extproc = next(
        index
        for index, item in enumerate(filters)
        if item["name"] == "envoy.filters.http.ext_proc"
    )
    # A filter ahead of ext_proc removes every client copy, so the Router never
    # reads one; the route-level removal runs only at the router filter.
    strip = filters[extproc - 1]["typed_config"]["inline_code"]
    for header in stripped:
        assert f'request_handle:headers():remove("{header}")' in strip
    removed = hcm["route_config"]["virtual_hosts"][0]["request_headers_to_remove"]
    assert set(stripped) <= set(removed)


def test_delegated_credential_header_must_be_a_plain_name(tmp_path):
    path = _write_config(tmp_path, ["model-a"])
    document = yaml.safe_load(path.read_text())
    document["global"] = {
        "services": {"authz": {"delegated_credential": {"header": 'x-a")..evil'}}}
    }
    path.write_text(yaml.safe_dump(document))
    config = parse_user_config(str(path))
    with pytest.raises(ValueError, match="plain header name"):
        generate_envoy_config_from_user_config(config, str(tmp_path / "envoy.yaml"))


IDENTITY_HEADERS = [
    "x-authz-user-id",
    "x-authz-user-groups",
    "x-authz-team-id",
    "x-authz-tenant-id",
]


def _render(tmp_path, document):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document))
    config = parse_user_config(str(path))
    assert not validate_user_config(config, log_summary=False)
    output = tmp_path / "envoy.yaml"
    generate_envoy_config_from_user_config(config, str(output))
    return yaml.safe_load(output.read_text())


def _connection_manager(rendered, index):
    listener = rendered["static_resources"]["listeners"][index]
    return listener["filter_chains"][0]["filters"][0]["typed_config"]


def test_listener_removes_client_identity_before_any_filter(tmp_path):
    document = yaml.safe_load(_write_config(tmp_path, ["model-a"]).read_text())
    document["listeners"].append(
        {
            "name": "behind-proxy",
            "address": "127.0.0.1",
            "port": 8900,
            "identity": {"trust_headers": True},
        }
    )
    document["global"] = {
        "services": {
            "authz": {
                "identity": {
                    "user_id_header": "X-JWT-Sub",
                    "user_groups_header": "x-jwt-groups",
                }
            }
        }
    }
    rendered = _render(tmp_path, document)
    want = [*IDENTITY_HEADERS, "x-jwt-sub", "x-jwt-groups"]

    untrusted = _connection_manager(rendered, 0)["http_filters"]
    assert untrusted[0]["name"] == "strip-client-identity"
    mutations = untrusted[0]["typed_config"]["mutations"]["request_mutations"]
    assert [mutation["remove"] for mutation in mutations] == want
    names = [item["name"] for item in untrusted]
    assert names.index("strip-client-identity") < names.index(
        "envoy.filters.http.ext_proc"
    )

    trusted = _connection_manager(rendered, 1)["http_filters"]
    assert "strip-client-identity" not in [item["name"] for item in trusted]

    # No provider receives them, whichever listener the request came through.
    for index in (0, 1):
        virtual_host = _connection_manager(rendered, index)["route_config"][
            "virtual_hosts"
        ][0]
        assert set(want) <= set(virtual_host["request_headers_to_remove"])


def test_extproc_listener_refuses_trusted_peers(tmp_path):
    document = yaml.safe_load(_write_config(tmp_path, ["model-a"]).read_text())
    document["listeners"][0]["identity"] = {
        "trust_headers": True,
        "trusted_peers": ["10.0.0.0/8"],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document))
    config = parse_user_config(str(path))
    with pytest.raises(ValueError, match=r"identity\.trusted_peers"):
        generate_envoy_config_from_user_config(config, str(tmp_path / "envoy.yaml"))


def test_identity_header_names_must_be_removable(tmp_path):
    document = yaml.safe_load(_write_config(tmp_path, ["model-a"]).read_text())
    document["global"] = {
        "services": {"authz": {"identity": {"user_id_header": "x user"}}}
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document))
    config = parse_user_config(str(path))
    with pytest.raises(ValueError, match="user_id_header"):
        generate_envoy_config_from_user_config(config, str(tmp_path / "envoy.yaml"))
