from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from cli import container_cli, container_start
from cli.config_generator import generate_envoy_config_from_user_config
from cli.container_host_ports import build_host_port_publication_plan
from cli.parser import parse_user_config
from cli.runtime_stack import resolve_runtime_stack


@pytest.fixture(autouse=True)
def _split_runtime_topology(monkeypatch):
    monkeypatch.setenv("VLLM_SR_TOPOLOGY", "split")


def _prepare_runtime(monkeypatch, tmp_path, *, render_split_config=False):
    commands = []

    def fake_run(cmd, capture_output, text, check, **kwargs):
        commands.append(cmd)
        return SimpleNamespace(stdout="container-id\n", stderr="")

    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **kwargs: {
            "router": "router:test",
            "envoy": "envoy:test",
            "dashboard": "dashboard:test",
        },
    )
    monkeypatch.setattr(container_start.subprocess, "run", fake_run)
    if not render_split_config:
        monkeypatch.setattr(
            container_start, "_render_split_envoy_config", lambda *args, **kwargs: None
        )
    docker_bin = tmp_path / "docker"
    docker_bin.write_text("")
    monkeypatch.setattr(
        container_start,
        "resolve_container_cli_path",
        lambda preferred_path=None: str(docker_bin),
    )
    return commands


def _start_runtime(tmp_path, listener_address, *, minimal, host_port_plan=None):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n"
        "  - name: http-8899\n"
        f"    address: {listener_address}\n"
        "    port: 8899\n"
    )
    listener = {
        "name": "http-8899",
        "address": listener_address,
        "port": 8899,
    }
    result = container_cli.container_start_vllm_sr(
        str(config_path),
        {},
        [listener],
        minimal=minimal,
        host_port_plan=host_port_plan,
    )
    return result, config_path


def _find_container_command(commands, container_name):
    for command in commands:
        if (
            "--name" in command
            and command[command.index("--name") + 1] == container_name
        ):
            return command
    raise AssertionError(f"container command not found: {container_name}")


def _mounted_envoy_config(envoy_command):
    for index, token in enumerate(envoy_command):
        if token != "-v":
            continue
        mount_spec = envoy_command[index + 1]
        suffix = ":/etc/envoy/envoy.yaml:z"
        if mount_spec.endswith(suffix):
            config_path = Path(mount_spec.removesuffix(suffix))
            return yaml.safe_load(config_path.read_text())
    raise AssertionError("mounted Envoy config not found")


def test_envoy_host_publish_keeps_loopback_listener_local(tmp_path, monkeypatch):
    commands = _prepare_runtime(monkeypatch, tmp_path, render_split_config=True)

    (return_code, _, _), config_path = _start_runtime(
        tmp_path, "127.0.0.1", minimal=True
    )

    assert return_code == 0
    envoy_command = _find_container_command(commands, "vllm-sr-envoy-container")
    assert "127.0.0.1:8899:8899" in envoy_command
    rendered = _mounted_envoy_config(envoy_command)
    socket_address = rendered["static_resources"]["listeners"][0]["address"][
        "socket_address"
    ]
    assert socket_address["address"] == "0.0.0.0"
    assert "address: 127.0.0.1" in config_path.read_text()


def test_standard_config_generator_preserves_listener_bind_address(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "envoy.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n"
        "  - name: http-8899\n"
        "    address: 127.0.0.1\n"
        "    port: 8899\n"
    )
    monkeypatch.setenv("ENVOY_EXTPROC_ADDRESS", "localhost")
    monkeypatch.setenv("ENVOY_ROUTER_API_ADDRESS", "localhost")

    generate_envoy_config_from_user_config(
        parse_user_config(str(config_path)), str(output_path)
    )

    rendered = yaml.safe_load(output_path.read_text())
    socket_address = rendered["static_resources"]["listeners"][0]["address"][
        "socket_address"
    ]
    assert socket_address["address"] == "127.0.0.1"


def test_internal_host_publish_allows_explicit_valid_override(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv("VLLM_SR_INTERNAL_BIND_ADDRESS", "192.0.2.10")
    commands = _prepare_runtime(monkeypatch, tmp_path)

    with caplog.at_level("WARNING"):
        (return_code, _, _), _ = _start_runtime(tmp_path, "0.0.0.0", minimal=False)

    assert return_code == 0
    router_command = _find_container_command(commands, "vllm-sr-router-container")
    envoy_command = _find_container_command(commands, "vllm-sr-envoy-container")
    dashboard_command = _find_container_command(commands, "vllm-sr-dashboard-container")
    assert "192.0.2.10:50051:50051" in router_command
    assert "192.0.2.10:8700:8700" in dashboard_command
    assert "0.0.0.0:8899:8899" in envoy_command
    assert "publishes internal, data, and observability ports" in caplog.text
    assert all(
        "VLLM_SR_INTERNAL_BIND_ADDRESS" not in token
        for command in commands
        for token in command
    )


def test_preflight_plan_prevents_late_environment_drift(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_SR_INTERNAL_BIND_ADDRESS", raising=False)
    monkeypatch.delenv("VLLM_SR_PORT_OFFSET", raising=False)
    listener = {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
    plan = build_host_port_publication_plan(resolve_runtime_stack(), [listener])
    monkeypatch.setenv("VLLM_SR_INTERNAL_BIND_ADDRESS", "invalid-after-preflight")
    commands = _prepare_runtime(monkeypatch, tmp_path)

    (return_code, _, _), _ = _start_runtime(
        tmp_path,
        "0.0.0.0",
        minimal=True,
        host_port_plan=plan,
    )

    assert return_code == 0
    router_command = _find_container_command(commands, "vllm-sr-router-container")
    assert "127.0.0.1:50051:50051" in router_command


def test_preflight_rejects_duplicate_split_listener_container_ports():
    listeners = [
        {"name": "loopback", "address": "127.0.0.1", "port": 8899},
        {"name": "private", "address": "192.0.2.10", "port": 8899},
    ]

    with pytest.raises(
        ValueError,
        match=r"listener\[0\].*listener\[1\].*port 8899",
    ):
        build_host_port_publication_plan(resolve_runtime_stack(), listeners)


def test_preflight_rejects_listener_colliding_with_envoy_admin_container_port():
    listener = {"name": "admin-collision", "address": "127.0.0.1", "port": 9901}

    with pytest.raises(
        ValueError,
        match=r"Envoy admin.*listener\[0\].*port 9901",
    ):
        build_host_port_publication_plan(resolve_runtime_stack(), [listener])


def test_preflight_rejects_wildcard_listener_overlapping_internal_dashboard():
    listener = {"name": "public", "address": "0.0.0.0", "port": 8700}

    with pytest.raises(
        ValueError,
        match=r"dashboard \(127\.0\.0\.1:8700\).*listener\[0\] \(0\.0\.0\.0:8700\)",
    ):
        build_host_port_publication_plan(resolve_runtime_stack(), [listener])


def test_preflight_rejects_duplicate_internal_fixed_publications():
    layout = resolve_runtime_stack()
    conflicting_layout = replace(layout, dashboard_port=layout.router_port)
    listener = {"name": "public", "address": "0.0.0.0", "port": 8899}

    with pytest.raises(
        ValueError,
        match=r"router extproc \(127\.0\.0\.1:50051\).*dashboard \(127\.0\.0\.1:50051\)",
    ):
        build_host_port_publication_plan(conflicting_layout, [listener])


def test_preflight_allows_same_host_port_on_distinct_concrete_addresses():
    listener = {"name": "secondary-loopback", "address": "127.0.0.2", "port": 8700}

    plan = build_host_port_publication_plan(resolve_runtime_stack(), [listener])

    assert plan.listeners[0].bind_address == "127.0.0.2"
    assert plan.listeners[0].host_port == 8700


def test_invalid_internal_bind_fails_before_filesystem_or_container_mutation(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("VLLM_SR_INTERNAL_BIND_ADDRESS", "localhost:8080")
    commands = _prepare_runtime(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="VLLM_SR_INTERNAL_BIND_ADDRESS"):
        _start_runtime(tmp_path, "0.0.0.0", minimal=False)

    assert commands == []
    assert not (tmp_path / ".vllm-sr").exists()
