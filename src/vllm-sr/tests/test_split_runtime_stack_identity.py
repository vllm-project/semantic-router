"""Stack identity and port-isolation contracts for the split runtime."""

import pytest
from cli import container_cli, container_start
from cli.runtime_stack import resolve_runtime_stack
from split_runtime_test_support import CONFIG_BODY as _CONFIG_BODY
from split_runtime_test_support import capture_run_commands as _capture_run_commands
from split_runtime_test_support import find_container_run_cmd as _find_container_run_cmd
from split_runtime_test_support import option_values as _option_values
from split_runtime_test_support import (
    stub_valid_container_cli as _stub_valid_container_cli,
)


@pytest.fixture(autouse=True)
def _split_runtime_topology(monkeypatch):
    monkeypatch.setenv("VLLM_SR_TOPOLOGY", "split")


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
    config_path.write_text(_CONFIG_BODY)

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
    """Every role resolves the same stack-scoped compiled bootstrap."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_CONFIG_BODY)

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
    assert "VLLM_SR_STACK_NAME=audit-a" in dashboard_cmd
    assert "VLLM_SR_STACK_NAME=audit-a" in router_cmd
    assert not any("VLLM_SR_RECIPE_STORE_DIR=" in arg for arg in dashboard_cmd)
    assert not any("VLLM_SR_ACTIVE_RECIPE_DIR=" in arg for arg in dashboard_cmd)
    assert not any("/app/recipe" in arg for arg in dashboard_cmd)
    assert "/app/.vllm-sr/compiled-bootstrap.audit-a.yaml" in dashboard_cmd
    assert "/app/.vllm-sr/compiled-bootstrap.audit-a.yaml" in router_cmd
    assert any(
        mount.endswith(":/app/.vllm-sr/compiled-bootstrap.audit-a.yaml:ro,z")
        for mount in _option_values(dashboard_cmd, "-v")
    )


def test_container_start_vllm_sr_omits_stack_name_env_for_default_stack(
    tmp_path, monkeypatch
):
    """Default stack does not need an explicit stack-name environment value."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_CONFIG_BODY)

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
