from types import SimpleNamespace

import pytest
from cli import container_cli, container_start, core, runtime_lifecycle
from cli.consts import (
    DEFAULT_API_PORT,
    DEFAULT_DASHBOARD_PORT,
    DEFAULT_FLEET_SIM_PORT,
    DEFAULT_METRICS_PORT,
    DEFAULT_ROUTER_PORT,
)
from cli.runtime_stack import resolve_runtime_stack


@pytest.fixture(autouse=True)
def _split_runtime_topology(monkeypatch):
    monkeypatch.setenv("VLLM_SR_TOPOLOGY", "split")


def _capture_run_commands(monkeypatch):
    captured = []

    def fake_run(cmd, capture_output, text, check):
        captured.append(cmd)
        return SimpleNamespace(stdout="container-id\n", stderr="")

    monkeypatch.setattr(container_start.subprocess, "run", fake_run)
    monkeypatch.setattr(
        container_start, "_render_split_envoy_config", lambda *args, **kwargs: None
    )
    return captured


def _find_container_run_cmd(commands, container_name):
    for cmd in commands:
        if "--name" not in cmd:
            continue
        if cmd[cmd.index("--name") + 1] == container_name:
            return cmd
    raise AssertionError(
        f"container command for {container_name} not found: {commands!r}"
    )


def _option_values(command, option):
    return [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == option
    ]


def _stub_valid_container_cli(monkeypatch, tmp_path):
    docker_bin = tmp_path / "docker"
    docker_bin.write_text("")
    monkeypatch.setattr(
        container_start,
        "resolve_container_cli_path",
        lambda preferred_path=None: str(docker_bin),
    )
    return docker_bin


def test_core_resolves_management_port_for_startup_health_check():
    assert (
        core._configured_management_port(
            {"global": {"services": {"management_api": {"port": 9090}}}}
        )
        == 9090
    )
    assert core._configured_management_port({}) == 8080


def test_container_start_vllm_sr_sets_split_service_urls_for_dashboard(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\n"
    )

    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **kwargs: {
            "router": "test-image",
            "envoy": "test-image",
            "dashboard": "test-image",
        },
    )
    captured = _capture_run_commands(monkeypatch)
    _stub_valid_container_cli(monkeypatch, tmp_path)

    rc, _, _ = container_cli.container_start_vllm_sr(
        str(config_path),
        {"TARGET_ROUTER_API_URL": "http://stale-router:8080"},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        network_name="vllm-sr-network",
        openclaw_network_name="vllm-sr-network",
        minimal=False,
    )

    assert rc == 0
    dashboard_cmd = _find_container_run_cmd(captured, "vllm-sr-dashboard-container")
    assert "TARGET_ROUTER_API_URL=http://vllm-sr-router-container:8080" in dashboard_cmd
    assert (
        "TARGET_ROUTER_METRICS_URL=http://vllm-sr-router-container:9190/metrics"
        in dashboard_cmd
    )
    assert "TARGET_ENVOY_URL=http://vllm-sr-envoy-container:8899" in dashboard_cmd
    assert "VLLM_SR_ENVOY_CONFIG_PATH=/app/.vllm-sr/envoy.yaml" in dashboard_cmd
    assert "ENVOY_EXTPROC_ADDRESS=vllm-sr-router-container" in dashboard_cmd
    assert "ENVOY_ROUTER_API_ADDRESS=vllm-sr-router-container" in dashboard_cmd
    assert "ROUTER_CONFIG_PATH=/app/.vllm-sr/runtime-config.yaml" in dashboard_cmd
    router_cmd = _find_container_run_cmd(captured, "vllm-sr-router-container")
    assert "127.0.0.1:8080:8080" in router_cmd
    assert "127.0.0.1:50051:50051" in router_cmd
    assert "127.0.0.1:9190:9190" in router_cmd
    envoy_cmd = _find_container_run_cmd(captured, "vllm-sr-envoy-container")
    assert "0.0.0.0:8899:8899" in envoy_cmd

    for component, command in (
        ("router", router_cmd),
        ("envoy", envoy_cmd),
        ("dashboard", dashboard_cmd),
    ):
        assert command[command.index("--entrypoint") + 1] == "/bin/sh"
        producer_mounts = [
            mount
            for mount in _option_values(command, "-v")
            if mount.endswith(":/var/log/vllm-sr-producer/current.log:z")
        ]
        assert len(producer_mounts) == 1
        assert producer_mounts[0].split(":", 1)[0].endswith(f"/{component}.log")

    assert any(
        mount.endswith(":/app/.vllm-sr/logs:ro,z")
        for mount in _option_values(router_cmd, "-v")
    )
    assert any(
        mount.endswith(":/var/log/vllm-sr:ro,z")
        for mount in _option_values(dashboard_cmd, "-v")
    )
    assert "VLLM_SR_LOG_SPOOL_DIR=/var/log/vllm-sr" in dashboard_cmd
    assert any(
        value.startswith("VLLM_SR_LOG_SPOOL_GID=")
        for value in _option_values(dashboard_cmd, "-e")
    )
    assert "--group-add" in envoy_cmd


def test_split_runtime_honors_management_listener_port(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.3\n"
        "listeners:\n  - name: public\n    address: 0.0.0.0\n    port: 8899\n"
        "global:\n  services:\n    management_api:\n"
        "      bind_address: 0.0.0.0\n      port: 9090\n"
    )
    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **_kwargs: {
            "router": "test-image",
            "envoy": "test-image",
            "dashboard": "test-image",
        },
    )
    captured = _capture_run_commands(monkeypatch)
    _stub_valid_container_cli(monkeypatch, tmp_path)

    rc, _, _ = container_cli.container_start_vllm_sr(
        str(config_path),
        {"TARGET_ROUTER_API_URL": "http://stale-router:8080"},
        [{"name": "public", "address": "0.0.0.0", "port": 8899}],
        network_name="vllm-sr-network",
        minimal=False,
    )

    assert rc == 0
    router_cmd = _find_container_run_cmd(captured, "vllm-sr-router-container")
    dashboard_cmd = _find_container_run_cmd(captured, "vllm-sr-dashboard-container")
    assert "127.0.0.1:9090:9090" in router_cmd
    assert "127.0.0.1:8080:8080" not in router_cmd
    assert "TARGET_ROUTER_API_URL=http://vllm-sr-router-container:9090" in dashboard_cmd
    assert "TARGET_ROUTER_API_URL=http://stale-router:8080" not in dashboard_cmd


@pytest.mark.parametrize("bind_address", ["127.0.0.1", "localhost", "::1"])
def test_split_runtime_rejects_management_listener_unreachable_from_dashboard(
    tmp_path, monkeypatch, bind_address
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.3\n"
        "listeners:\n  - name: public\n    address: 0.0.0.0\n    port: 8899\n"
        "global:\n  services:\n    management_api:\n"
        f"      bind_address: {bind_address}\n      port: 8080\n"
    )
    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **_kwargs: {
            "router": "test-image",
            "envoy": "test-image",
            "dashboard": "test-image",
        },
    )
    _stub_valid_container_cli(monkeypatch, tmp_path)

    with pytest.raises(
        ValueError, match=r"requires management_api\.bind_address 0\.0\.0\.0"
    ):
        container_cli.container_start_vllm_sr(
            str(config_path),
            {},
            [{"name": "public", "address": "0.0.0.0", "port": 8899}],
            network_name="vllm-sr-network",
            minimal=False,
        )


def test_split_runtime_rejects_invalid_management_auth_exposure(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.3\n"
        "listeners:\n  - name: public\n    address: 0.0.0.0\n    port: 8899\n"
        "global:\n  services:\n    management_api:\n"
        "      bind_address: 0.0.0.0\n      port: 8080\n"
        "      remote_exposure: true\n      auth:\n        mode: disabled\n"
    )
    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **_kwargs: {
            "router": "test-image",
            "envoy": "test-image",
            "dashboard": "test-image",
        },
    )
    _stub_valid_container_cli(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="requires bearer auth tokens"):
        container_cli.container_start_vllm_sr(
            str(config_path),
            {},
            [{"name": "public", "address": "0.0.0.0", "port": 8899}],
            network_name="vllm-sr-network",
            minimal=False,
        )


def test_envoy_host_publish_preserves_loopback_listener_address(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n  - name: local-http\n    address: 127.0.0.1\n    port: 8899\n"
    )
    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **_kwargs: {
            "router": "test-image",
            "envoy": "test-image",
            "dashboard": "test-image",
        },
    )
    captured = _capture_run_commands(monkeypatch)
    _stub_valid_container_cli(monkeypatch, tmp_path)

    rc, _, _ = container_cli.container_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "local-http", "address": "127.0.0.1", "port": 8899}],
        network_name="vllm-sr-network",
        openclaw_network_name="vllm-sr-network",
        minimal=False,
    )

    assert rc == 0
    envoy_cmd = _find_container_run_cmd(captured, "vllm-sr-envoy-container")
    assert "127.0.0.1:8899:8899" in envoy_cmd


def test_envoy_host_publish_brackets_ipv6_listener_address(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n  - name: local-v6\n    address: ::1\n    port: 8899\n"
    )
    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **_kwargs: {
            "router": "test-image",
            "envoy": "test-image",
            "dashboard": "test-image",
        },
    )
    captured = _capture_run_commands(monkeypatch)
    _stub_valid_container_cli(monkeypatch, tmp_path)

    rc, _, _ = container_cli.container_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "local-v6", "address": "::1", "port": 8899}],
        network_name="vllm-sr-network",
        openclaw_network_name="vllm-sr-network",
        minimal=False,
    )

    assert rc == 0
    envoy_cmd = _find_container_run_cmd(captured, "vllm-sr-envoy-container")
    assert "[::1]:8899:8899" in envoy_cmd


def test_container_start_vllm_sr_uses_role_specific_runtime_images(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\n"
    )

    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **kwargs: {
            "router": "router-image:latest",
            "envoy": "envoy-image:latest",
            "dashboard": "dashboard-image:latest",
        },
    )
    captured = _capture_run_commands(monkeypatch)
    _stub_valid_container_cli(monkeypatch, tmp_path)

    rc, _, _ = container_cli.container_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        network_name="vllm-sr-network",
        openclaw_network_name="vllm-sr-network",
        minimal=False,
    )

    assert rc == 0
    router_cmd = _find_container_run_cmd(captured, "vllm-sr-router-container")
    envoy_cmd = _find_container_run_cmd(captured, "vllm-sr-envoy-container")
    dashboard_cmd = _find_container_run_cmd(captured, "vllm-sr-dashboard-container")
    assert "router-image:latest" in router_cmd
    assert "envoy-image:latest" in envoy_cmd
    assert "dashboard-image:latest" in dashboard_cmd
    assert "/usr/local/bin/envoy" in envoy_cmd
    assert "/etc/envoy/envoy.yaml" in " ".join(envoy_cmd)


def test_container_start_vllm_sr_skips_dashboard_image_resolution_in_minimal_mode(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\n"
    )
    captured_kwargs = {}

    def fake_get_runtime_images(**kwargs):
        captured_kwargs.update(kwargs)
        return {
            "router": "router-image:latest",
            "envoy": "envoy-image:latest",
        }

    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(container_start, "get_runtime_images", fake_get_runtime_images)
    captured = _capture_run_commands(monkeypatch)
    _stub_valid_container_cli(monkeypatch, tmp_path)

    rc, _, _ = container_cli.container_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        network_name="vllm-sr-network",
        openclaw_network_name="vllm-sr-network",
        minimal=True,
    )

    assert rc == 0
    assert captured_kwargs["include_dashboard"] is False
    _find_container_run_cmd(captured, "vllm-sr-router-container")
    _find_container_run_cmd(captured, "vllm-sr-envoy-container")
    with pytest.raises(AssertionError):
        _find_container_run_cmd(captured, "vllm-sr-dashboard-container")


def test_container_start_vllm_sr_creates_router_and_envoy_standby_in_setup_mode(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.3\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\nsetup:\n  mode: true\n"
    )

    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **kwargs: {
            "router": "router-image:latest",
            "envoy": "envoy-image:latest",
            "dashboard": "dashboard-image:latest",
        },
    )
    captured = _capture_run_commands(monkeypatch)
    _stub_valid_container_cli(monkeypatch, tmp_path)

    rc, _, _ = container_cli.container_start_vllm_sr(
        str(config_path),
        {"VLLM_SR_SETUP_MODE": "true"},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        network_name="vllm-sr-network",
        openclaw_network_name="vllm-sr-network",
        minimal=False,
    )

    assert rc == 0
    router_cmd = _find_container_run_cmd(captured, "vllm-sr-router-container")
    envoy_cmd = _find_container_run_cmd(captured, "vllm-sr-envoy-container")
    dashboard_cmd = _find_container_run_cmd(captured, "vllm-sr-dashboard-container")
    assert router_cmd[1] == "create"
    assert envoy_cmd[1] == "create"
    assert dashboard_cmd[1:3] == ["run", "-d"]


def test_start_vllm_sr_creates_and_connects_shared_network_without_observability(
    monkeypatch,
):
    calls = []

    def record(name, ret=(0, "", "")):
        def _fn(*args, **kwargs):
            calls.append((name, args, kwargs))
            return ret

        return _fn

    monkeypatch.setattr(core, "print_vllm_logo", lambda: None)
    monkeypatch.setattr(core, "ensure_clean_runtime_container", lambda _name: None)
    monkeypatch.setattr(
        core,
        "load_config",
        lambda path: {
            "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}]
        },
    )
    monkeypatch.setattr(
        core, "provision_storage_backends", lambda *args, **kwargs: set()
    )
    monkeypatch.setattr(
        runtime_lifecycle,
        "container_status",
        lambda _name: "running",
    )
    monkeypatch.setattr(
        runtime_lifecycle,
        "container_create_network",
        record("container_create_network"),
    )
    monkeypatch.setattr(
        core, "start_fleet_sim_sidecar", record("start_fleet_sim_sidecar", True)
    )
    monkeypatch.setattr(
        core, "container_start_vllm_sr", record("container_start_vllm_sr")
    )
    monkeypatch.setattr(
        runtime_lifecycle,
        "container_network_connect",
        record("container_network_connect"),
    )
    monkeypatch.setattr(
        runtime_lifecycle, "container_logs_since", lambda *args, **kwargs: (0, "", "")
    )
    monkeypatch.setattr(
        runtime_lifecycle, "container_exec", lambda *args, **kwargs: (0, "ok", "")
    )
    monkeypatch.setattr(
        runtime_lifecycle, "load_openclaw_registry", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        runtime_lifecycle, "container_logs", lambda *args, **kwargs: None
    )

    core.start_vllm_sr("/tmp/config.yaml", env_vars={}, enable_observability=False)

    create_calls = [c for c in calls if c[0] == "container_create_network"]
    fleet_sim_calls = [c for c in calls if c[0] == "start_fleet_sim_sidecar"]
    start_calls = [c for c in calls if c[0] == "container_start_vllm_sr"]
    connect_calls = [c for c in calls if c[0] == "container_network_connect"]

    assert create_calls[0][1] == ("vllm-sr-network",)
    assert fleet_sim_calls[0][1][0] == "/tmp"
    assert fleet_sim_calls[0][1][2].fleet_sim_container_name == "vllm-sr-sim-container"
    assert start_calls[0][2]["network_name"] == "vllm-sr-network"
    assert start_calls[0][2]["openclaw_network_name"] == "vllm-sr-network"
    assert start_calls[0][2]["runtime_config_file"] == "/tmp/config.yaml"
    assert (
        start_calls[0][2]["env_vars"]["TARGET_FLEET_SIM_URL"]
        == "http://vllm-sr-sim-container:8000"
    )
    assert [call[1] for call in connect_calls] == [
        ("vllm-sr-network", "vllm-sr-router-container"),
        ("vllm-sr-network", "vllm-sr-envoy-container"),
        ("vllm-sr-network", "vllm-sr-dashboard-container"),
    ]


def test_resolve_runtime_stack_supports_custom_stack_name_and_port_offset():
    stack_layout = resolve_runtime_stack(stack_name="audit-a", port_offset=200)

    assert stack_layout.router_container_name == "audit-a-vllm-sr-router-container"
    assert stack_layout.envoy_container_name == "audit-a-vllm-sr-envoy-container"
    assert (
        stack_layout.dashboard_container_name == "audit-a-vllm-sr-dashboard-container"
    )
    assert stack_layout.fleet_sim_container_name == "audit-a-vllm-sr-sim"
    assert stack_layout.network_name == "audit-a-vllm-sr-network"
    assert stack_layout.jaeger_container_name == "audit-a-vllm-sr-jaeger"
    assert stack_layout.prometheus_container_name == "audit-a-vllm-sr-prometheus"
    assert stack_layout.grafana_container_name == "audit-a-vllm-sr-grafana"
    assert stack_layout.router_port == DEFAULT_ROUTER_PORT + 200
    assert stack_layout.metrics_port == DEFAULT_METRICS_PORT + 200
    assert stack_layout.dashboard_port == DEFAULT_DASHBOARD_PORT + 200
    assert stack_layout.api_port == DEFAULT_API_PORT + 200
    assert stack_layout.fleet_sim_port == DEFAULT_FLEET_SIM_PORT + 200
    assert (
        stack_layout.dashboard_service_url
        == "http://audit-a-vllm-sr-dashboard-container:8700"
    )
    assert (
        stack_layout.router_api_service_url
        == "http://audit-a-vllm-sr-router-container:8080"
    )
    assert (
        stack_layout.router_metrics_service_url
        == "http://audit-a-vllm-sr-router-container:9190/metrics"
    )
    assert (
        stack_layout.envoy_admin_service_url
        == "http://audit-a-vllm-sr-envoy-container:9901"
    )
    assert (
        stack_layout.envoy_listener_service_url(8899)
        == "http://audit-a-vllm-sr-envoy-container:8899"
    )


def test_start_vllm_sr_uses_state_root_override(monkeypatch, tmp_path):
    calls = []
    state_root = tmp_path / "workspace-root"
    state_root.mkdir()

    def record(name, ret=(0, "", "")):
        def _fn(*args, **kwargs):
            calls.append((name, args, kwargs))
            return ret

        return _fn

    monkeypatch.setenv("VLLM_SR_STATE_ROOT_DIR", str(state_root))
    monkeypatch.setattr(core, "ensure_clean_runtime_container", lambda _name: None)
    monkeypatch.setattr(
        core,
        "load_config",
        lambda _path: {
            "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}]
        },
    )
    monkeypatch.setattr(
        core, "provision_storage_backends", lambda *args, **kwargs: set()
    )
    monkeypatch.setattr(
        runtime_lifecycle,
        "container_status",
        lambda _name: "running",
    )
    monkeypatch.setattr(
        runtime_lifecycle,
        "container_create_network",
        record("container_create_network"),
    )
    monkeypatch.setattr(
        core, "start_fleet_sim_sidecar", record("start_fleet_sim_sidecar", True)
    )
    monkeypatch.setattr(
        core, "container_start_vllm_sr", record("container_start_vllm_sr")
    )
    monkeypatch.setattr(
        runtime_lifecycle,
        "container_network_connect",
        record("container_network_connect"),
    )
    monkeypatch.setattr(
        runtime_lifecycle, "container_logs_since", lambda *args, **kwargs: (0, "", "")
    )
    monkeypatch.setattr(
        runtime_lifecycle, "container_exec", lambda *args, **kwargs: (0, "ok", "")
    )
    monkeypatch.setattr(
        runtime_lifecycle, "load_openclaw_registry", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        runtime_lifecycle, "container_logs", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        core, "recover_openclaw_containers", record("recover_openclaw_containers")
    )

    core.start_vllm_sr("/tmp/config.yaml", env_vars={}, enable_observability=False)

    fleet_sim_calls = [c for c in calls if c[0] == "start_fleet_sim_sidecar"]
    start_calls = [c for c in calls if c[0] == "container_start_vllm_sr"]
    recover_calls = [c for c in calls if c[0] == "recover_openclaw_containers"]

    assert fleet_sim_calls[0][1][0] == str(state_root)
    assert start_calls[0][2]["state_root_dir"] == str(state_root)
    assert recover_calls[0][1][0] == str(state_root)


def test_resolve_runtime_stack_supports_default_role_container_names():
    stack_layout = resolve_runtime_stack()

    assert stack_layout.router_container_name == "vllm-sr-router-container"
    assert stack_layout.envoy_container_name == "vllm-sr-envoy-container"
    assert stack_layout.dashboard_container_name == "vllm-sr-dashboard-container"
    assert (
        stack_layout.dashboard_service_url == "http://vllm-sr-dashboard-container:8700"
    )
    assert stack_layout.router_api_service_url == "http://vllm-sr-router-container:8080"
    assert (
        stack_layout.router_metrics_service_url
        == "http://vllm-sr-router-container:9190/metrics"
    )
    assert stack_layout.envoy_admin_service_url == "http://vllm-sr-envoy-container:9901"
    assert (
        stack_layout.envoy_listener_service_url(8899)
        == "http://vllm-sr-envoy-container:8899"
    )


def test_container_start_vllm_sr_applies_custom_stack_name_and_port_offset(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\n"
    )

    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **kwargs: {
            "router": "test-image",
            "envoy": "test-image",
            "dashboard": "test-image",
        },
    )
    captured = _capture_run_commands(monkeypatch)
    _stub_valid_container_cli(monkeypatch, tmp_path)
    monkeypatch.setenv("VLLM_SR_STACK_NAME", "audit-a")
    monkeypatch.setenv("VLLM_SR_PORT_OFFSET", "200")

    rc, _, _ = container_cli.container_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        network_name=None,
        openclaw_network_name=None,
        minimal=False,
    )

    assert rc == 0
    router_cmd = _find_container_run_cmd(captured, "audit-a-vllm-sr-router-container")
    envoy_cmd = _find_container_run_cmd(captured, "audit-a-vllm-sr-envoy-container")
    dashboard_cmd = _find_container_run_cmd(
        captured, "audit-a-vllm-sr-dashboard-container"
    )
    assert "OPENCLAW_DEFAULT_NETWORK_MODE=audit-a-vllm-sr-network" in dashboard_cmd
    assert "VLLM_SR_PORT_OFFSET=200" in dashboard_cmd
    assert "0.0.0.0:9099:8899" in envoy_cmd
    assert "127.0.0.1:50251:50051" in router_cmd
    assert "127.0.0.1:9390:9190" in router_cmd
    assert "8900:8700" in dashboard_cmd
    assert "127.0.0.1:8280:8080" in router_cmd


def test_container_start_vllm_sr_propagates_stack_name_to_dashboard(
    tmp_path, monkeypatch
):
    """Dashboard's runtime-config sync resolves the per-stack filename via
    VLLM_SR_STACK_NAME. Without the env propagated into the dashboard container,
    sync writes to runtime-config.yaml while the CLI wrote
    runtime-config.<stack>.yaml; router stays pinned to the stale path and
    setup-mode never disengages.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\n"
    )

    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **kwargs: {
            "router": "test-image",
            "envoy": "test-image",
            "dashboard": "test-image",
        },
    )
    captured = _capture_run_commands(monkeypatch)
    _stub_valid_container_cli(monkeypatch, tmp_path)
    monkeypatch.setenv("VLLM_SR_STACK_NAME", "audit-a")

    rc, _, _ = container_cli.container_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        network_name=None,
        openclaw_network_name=None,
        minimal=False,
    )

    assert rc == 0
    router_cmd = _find_container_run_cmd(captured, "audit-a-vllm-sr-router-container")
    dashboard_cmd = _find_container_run_cmd(
        captured, "audit-a-vllm-sr-dashboard-container"
    )
    # Dashboard runs the runtime-config sync, router consumes the resolved
    # path; both need stack-name visibility.
    assert "VLLM_SR_STACK_NAME=audit-a" in dashboard_cmd
    assert "VLLM_SR_STACK_NAME=audit-a" in router_cmd
    assert (
        "VLLM_SR_RECIPE_STORE_DIR=/app/.vllm-sr/recipe-store/audit-a" in dashboard_cmd
    )
    assert "/app/.vllm-sr/runtime-config.audit-a.yaml" in dashboard_cmd
    assert "/app/.vllm-sr/runtime-config.audit-a.yaml" in router_cmd


def test_container_start_vllm_sr_omits_stack_name_env_for_default_stack(
    tmp_path, monkeypatch
):
    """Default stack uses the unsuffixed runtime-config.yaml on both ends, so
    we do not need to inject VLLM_SR_STACK_NAME and should not start doing so
    accidentally.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\n"
    )

    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **kwargs: {
            "router": "test-image",
            "envoy": "test-image",
            "dashboard": "test-image",
        },
    )
    captured = _capture_run_commands(monkeypatch)
    _stub_valid_container_cli(monkeypatch, tmp_path)
    monkeypatch.delenv("VLLM_SR_STACK_NAME", raising=False)

    rc, _, _ = container_cli.container_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        network_name="vllm-sr-network",
        openclaw_network_name="vllm-sr-network",
        minimal=False,
    )

    assert rc == 0
    dashboard_cmd = _find_container_run_cmd(captured, "vllm-sr-dashboard-container")
    router_cmd = _find_container_run_cmd(captured, "vllm-sr-router-container")
    for token in dashboard_cmd + router_cmd:
        message = f"unexpected stack-name env on default stack: {token}"
        assert not token.startswith("VLLM_SR_STACK_NAME="), message
