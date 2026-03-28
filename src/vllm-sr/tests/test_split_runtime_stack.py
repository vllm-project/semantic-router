from types import SimpleNamespace

import pytest
from cli import core, docker_cli, docker_start, runtime_lifecycle
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

    monkeypatch.setattr(docker_start.subprocess, "run", fake_run)
    monkeypatch.setattr(
        docker_start, "_render_split_envoy_config", lambda *args, **kwargs: None
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


def _stub_valid_docker_cli(monkeypatch, tmp_path):
    docker_bin = tmp_path / "docker"
    docker_bin.write_text("")
    monkeypatch.setattr(
        docker_start,
        "resolve_docker_cli_path",
        lambda preferred_path=None: str(docker_bin),
    )
    return docker_bin


def test_docker_start_vllm_sr_sets_split_service_urls_for_dashboard(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\n"
    )

    monkeypatch.setattr(docker_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        docker_start,
        "get_runtime_images",
        lambda **kwargs: {
            "router": "test-image",
            "envoy": "test-image",
            "dashboard": "test-image",
        },
    )
    captured = _capture_run_commands(monkeypatch)
    _stub_valid_docker_cli(monkeypatch, tmp_path)

    rc, _, _ = docker_cli.docker_start_vllm_sr(
        str(config_path),
        {},
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


def test_docker_start_vllm_sr_uses_role_specific_runtime_images(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\n"
    )

    monkeypatch.setattr(docker_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        docker_start,
        "get_runtime_images",
        lambda **kwargs: {
            "router": "router-image:latest",
            "envoy": "envoy-image:latest",
            "dashboard": "dashboard-image:latest",
        },
    )
    captured = _capture_run_commands(monkeypatch)
    _stub_valid_docker_cli(monkeypatch, tmp_path)

    rc, _, _ = docker_cli.docker_start_vllm_sr(
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


def test_docker_start_vllm_sr_creates_router_and_envoy_standby_in_setup_mode(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.3\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\nsetup:\n  mode: true\n"
    )

    monkeypatch.setattr(docker_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        docker_start,
        "get_runtime_images",
        lambda **kwargs: {
            "router": "router-image:latest",
            "envoy": "envoy-image:latest",
            "dashboard": "dashboard-image:latest",
        },
    )
    captured = _capture_run_commands(monkeypatch)
    _stub_valid_docker_cli(monkeypatch, tmp_path)

    rc, _, _ = docker_cli.docker_start_vllm_sr(
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
        runtime_lifecycle,
        "docker_container_status",
        lambda name: "not found" if name == "vllm-sr-container" else "running",
    )
    monkeypatch.setattr(
        runtime_lifecycle, "docker_create_network", record("docker_create_network")
    )
    monkeypatch.setattr(
        core, "start_fleet_sim_sidecar", record("start_fleet_sim_sidecar", True)
    )
    monkeypatch.setattr(core, "docker_start_vllm_sr", record("docker_start_vllm_sr"))
    monkeypatch.setattr(
        runtime_lifecycle, "docker_network_connect", record("docker_network_connect")
    )
    monkeypatch.setattr(
        runtime_lifecycle, "docker_logs_since", lambda *args, **kwargs: (0, "", "")
    )
    monkeypatch.setattr(
        runtime_lifecycle, "docker_exec", lambda *args, **kwargs: (0, "ok", "")
    )
    monkeypatch.setattr(
        runtime_lifecycle, "load_openclaw_registry", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(runtime_lifecycle, "docker_logs", lambda *args, **kwargs: None)

    core.start_vllm_sr("/tmp/config.yaml", env_vars={}, enable_observability=False)

    create_calls = [c for c in calls if c[0] == "docker_create_network"]
    fleet_sim_calls = [c for c in calls if c[0] == "start_fleet_sim_sidecar"]
    start_calls = [c for c in calls if c[0] == "docker_start_vllm_sr"]
    connect_calls = [c for c in calls if c[0] == "docker_network_connect"]

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

    assert stack_layout.container_name == "audit-a-vllm-sr-container"
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


def test_start_vllm_sr_uses_state_root_override(monkeypatch):
    calls = []

    def record(name, ret=(0, "", "")):
        def _fn(*args, **kwargs):
            calls.append((name, args, kwargs))
            return ret

        return _fn

    monkeypatch.setenv("VLLM_SR_STATE_ROOT_DIR", "/workspace-root")
    monkeypatch.setattr(core, "ensure_clean_runtime_container", lambda _name: None)
    monkeypatch.setattr(
        core,
        "load_config",
        lambda _path: {
            "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}]
        },
    )
    monkeypatch.setattr(
        runtime_lifecycle,
        "docker_container_status",
        lambda name: "not found" if name == "vllm-sr-container" else "running",
    )
    monkeypatch.setattr(
        runtime_lifecycle, "docker_create_network", record("docker_create_network")
    )
    monkeypatch.setattr(
        core, "start_fleet_sim_sidecar", record("start_fleet_sim_sidecar", True)
    )
    monkeypatch.setattr(core, "docker_start_vllm_sr", record("docker_start_vllm_sr"))
    monkeypatch.setattr(
        runtime_lifecycle, "docker_network_connect", record("docker_network_connect")
    )
    monkeypatch.setattr(
        runtime_lifecycle, "docker_logs_since", lambda *args, **kwargs: (0, "", "")
    )
    monkeypatch.setattr(
        runtime_lifecycle, "docker_exec", lambda *args, **kwargs: (0, "ok", "")
    )
    monkeypatch.setattr(
        runtime_lifecycle, "load_openclaw_registry", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(runtime_lifecycle, "docker_logs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        core, "recover_openclaw_containers", record("recover_openclaw_containers")
    )

    core.start_vllm_sr("/tmp/config.yaml", env_vars={}, enable_observability=False)

    fleet_sim_calls = [c for c in calls if c[0] == "start_fleet_sim_sidecar"]
    start_calls = [c for c in calls if c[0] == "docker_start_vllm_sr"]
    recover_calls = [c for c in calls if c[0] == "recover_openclaw_containers"]

    assert fleet_sim_calls[0][1][0] == "/workspace-root"
    assert start_calls[0][2]["state_root_dir"] == "/workspace-root"
    assert recover_calls[0][1][0] == "/workspace-root"


def test_resolve_runtime_stack_supports_default_role_container_names():
    stack_layout = resolve_runtime_stack()

    assert stack_layout.container_name == "vllm-sr-container"
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


def test_docker_start_vllm_sr_applies_custom_stack_name_and_port_offset(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\n"
    )

    monkeypatch.setattr(docker_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        docker_start,
        "get_runtime_images",
        lambda **kwargs: {
            "router": "test-image",
            "envoy": "test-image",
            "dashboard": "test-image",
        },
    )
    captured = _capture_run_commands(monkeypatch)
    _stub_valid_docker_cli(monkeypatch, tmp_path)
    monkeypatch.setenv("VLLM_SR_STACK_NAME", "audit-a")
    monkeypatch.setenv("VLLM_SR_PORT_OFFSET", "200")

    rc, _, _ = docker_cli.docker_start_vllm_sr(
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
    assert "9099:8899" in envoy_cmd
    assert "50251:50051" in router_cmd
    assert "9390:9190" in router_cmd
    assert "8900:8700" in dashboard_cmd
    assert "8280:8080" in router_cmd
