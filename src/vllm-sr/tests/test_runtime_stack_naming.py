"""Stack naming and port offsets, from layout resolution to the run command.

Split out of ``test_split_runtime_stack``: these tests are about one thing --
that a named stack with a port offset renames every container, network, and
published port consistently, and that a state-root override relocates the whole
stack -- while the split-runtime tests next door are about how the three
services address each other.
"""

import subprocess
from types import SimpleNamespace

import pytest
from cli import container_cli, container_start, core, runtime_lifecycle
from cli.consts import (
    DEFAULT_API_PORT,
    DEFAULT_DASHBOARD_PORT,
    DEFAULT_METRICS_PORT,
    DEFAULT_ROUTER_PORT,
)
from cli.runtime_stack import resolve_runtime_stack


@pytest.fixture(autouse=True)
def _split_runtime_topology(monkeypatch):
    monkeypatch.setenv("VLLM_SR_TOPOLOGY", "split")


def _capture_run_commands(monkeypatch):
    captured = []

    def fake_run(cmd, capture_output, text, check, env=None):
        captured.append(cmd)
        return SimpleNamespace(stdout="container-id\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
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


def test_resolve_runtime_stack_supports_custom_stack_name_and_port_offset():
    stack_layout = resolve_runtime_stack(stack_name="audit-a", port_offset=200)

    assert stack_layout.router_container_name == "audit-a-vllm-sr-router-container"
    assert stack_layout.envoy_container_name == "audit-a-vllm-sr-envoy-container"
    assert (
        stack_layout.dashboard_container_name == "audit-a-vllm-sr-dashboard-container"
    )
    assert stack_layout.network_name == "audit-a-vllm-sr-network"
    assert stack_layout.jaeger_container_name == "audit-a-vllm-sr-jaeger"
    assert stack_layout.prometheus_container_name == "audit-a-vllm-sr-prometheus"
    assert stack_layout.grafana_container_name == "audit-a-vllm-sr-grafana"
    assert stack_layout.router_port == DEFAULT_ROUTER_PORT + 200
    assert stack_layout.metrics_port == DEFAULT_METRICS_PORT + 200
    assert stack_layout.dashboard_port == DEFAULT_DASHBOARD_PORT + 200
    assert stack_layout.api_port == DEFAULT_API_PORT + 200
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

    start_calls = [c for c in calls if c[0] == "container_start_vllm_sr"]
    recover_calls = [c for c in calls if c[0] == "recover_openclaw_containers"]

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
