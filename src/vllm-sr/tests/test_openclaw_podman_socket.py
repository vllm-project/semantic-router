"""Podman-specific dashboard socket mount tests.

These tests cover the docker-or-podman runtime flag in
`cli.container_openclaw_support`: when runtime is podman the daemon socket
is mounted at the canonical `/var/run/docker.sock`, the host docker CLI
is NOT mounted (the in-image CLI is sufficient), and a missing podman
socket degrades to a warning rather than a fatal exit.

They live in a separate module so the docker-default coverage in
`test_openclaw_shared_network.py` stays under the project's 800-line
file-length limit.
"""

import os

import pytest
from cli import container_cli, container_start
from tests.test_openclaw_shared_network import (
    _capture_run_commands,
    _find_container_run_cmd,
)


@pytest.fixture(autouse=True)
def _split_runtime_topology(monkeypatch):
    monkeypatch.setenv("VLLM_SR_TOPOLOGY", "split")


_LISTENERS = [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}]
_CONFIG_BODY = (
    "version: v0.1\n"
    "listeners:\n"
    "  - name: http-8899\n"
    "    address: 0.0.0.0\n"
    "    port: 8899\n"
)


def _stub_runtime_images(monkeypatch):
    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "podman")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **kwargs: {
            "router": "test-image",
            "envoy": "test-image",
            "dashboard": "test-image",
        },
    )


def _start_vllm_sr(config_path):
    return container_cli.container_start_vllm_sr(
        str(config_path),
        {},
        _LISTENERS,
        network_name="vllm-sr-network",
        openclaw_network_name="vllm-sr-network",
        minimal=False,
    )


def test_container_start_vllm_sr_mounts_podman_socket_at_canonical_path(
    tmp_path, monkeypatch
):
    """Podman runtime should mount podman.sock at /var/run/docker.sock and skip docker-CLI mount."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_CONFIG_BODY)

    podman_socket = tmp_path / "podman.sock"
    podman_socket.write_text("")
    monkeypatch.setenv("VLLM_SR_CONTAINER_SOCKET", str(podman_socket))

    _stub_runtime_images(monkeypatch)
    captured = _capture_run_commands(monkeypatch)

    rc, _, _ = _start_vllm_sr(config_path)

    assert rc == 0
    dashboard_cmd = _find_container_run_cmd(captured, "vllm-sr-dashboard-container")
    # podman.sock mounted at the canonical /var/run/docker.sock so the in-image
    # docker CLI in the dashboard transparently drives podman.
    assert f"{podman_socket}:/var/run/docker.sock" in dashboard_cmd
    # Dashboard backend keeps exec'ing "docker"; the in-image CLI is what runs.
    assert "OPENCLAW_CONTAINER_RUNTIME=docker" in dashboard_cmd
    # Host docker CLI should NOT be mounted in podman mode — the runtime is
    # podman, not docker, so a host docker bin is irrelevant.
    assert not any(
        ":/usr/local/bin/docker:ro" in part for part in dashboard_cmd
    ), f"unexpected host CLI mount in podman mode: {dashboard_cmd!r}"


def test_container_start_vllm_sr_resolves_default_podman_socket(tmp_path, monkeypatch):
    """Without explicit override, podman mode should look up /run/podman/podman.sock."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_CONFIG_BODY)

    monkeypatch.delenv("VLLM_SR_CONTAINER_SOCKET", raising=False)
    real_exists = os.path.exists

    def fake_exists(path: str) -> bool:
        if path == "/run/podman/podman.sock":
            return True
        if path in ("/var/run/docker.sock",):
            return False
        return real_exists(path)

    monkeypatch.setattr(os.path, "exists", fake_exists)
    _stub_runtime_images(monkeypatch)
    captured = _capture_run_commands(monkeypatch)

    rc, _, _ = _start_vllm_sr(config_path)

    assert rc == 0
    dashboard_cmd = _find_container_run_cmd(captured, "vllm-sr-dashboard-container")
    assert "/run/podman/podman.sock:/var/run/docker.sock" in dashboard_cmd


def test_container_start_vllm_sr_warns_when_podman_socket_missing(
    tmp_path, monkeypatch
):
    """Podman mode should not fail if no podman socket is found — just warn."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_CONFIG_BODY)

    monkeypatch.delenv("VLLM_SR_CONTAINER_SOCKET", raising=False)
    real_exists = os.path.exists

    def fake_exists(path: str) -> bool:
        if "podman.sock" in path or "docker.sock" in path:
            return False
        return real_exists(path)

    monkeypatch.setattr(os.path, "exists", fake_exists)
    _stub_runtime_images(monkeypatch)
    captured = _capture_run_commands(monkeypatch)

    rc, _, _ = _start_vllm_sr(config_path)

    assert rc == 0
    dashboard_cmd = _find_container_run_cmd(captured, "vllm-sr-dashboard-container")
    assert not any(
        "docker.sock" in part for part in dashboard_cmd
    ), f"unexpected socket mount: {dashboard_cmd!r}"
