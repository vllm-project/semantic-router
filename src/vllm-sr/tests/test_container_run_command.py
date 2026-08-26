import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli import container_run_command  # noqa: E402


def test_append_env_vars_hides_inherited_secret_values():
    cmd = []
    container_run_command.append_env_vars(
        cmd,
        {
            "DASHBOARD_ADMIN_EMAIL": "core@vllm-sr.ai",
            "DASHBOARD_ADMIN_PASSWORD": "secret-value",
        },
        {"DASHBOARD_ADMIN_PASSWORD"},
    )

    assert cmd == [
        "-e",
        "DASHBOARD_ADMIN_EMAIL=core@vllm-sr.ai",
        "-e",
        "DASHBOARD_ADMIN_PASSWORD",
    ]
    assert "secret-value" not in " ".join(cmd)


def test_append_custom_dns_noop_when_unset(monkeypatch):
    monkeypatch.delenv("VLLM_SR_DNS", raising=False)
    cmd = []
    container_run_command.append_custom_dns(cmd)
    assert cmd == []


def test_append_custom_dns_noop_when_empty(monkeypatch):
    monkeypatch.setenv("VLLM_SR_DNS", "")
    cmd = []
    container_run_command.append_custom_dns(cmd)
    assert cmd == []


def test_append_custom_dns_single(monkeypatch):
    monkeypatch.setenv("VLLM_SR_DNS", "10.0.0.53")
    cmd = []
    container_run_command.append_custom_dns(cmd)
    assert cmd == ["--dns", "10.0.0.53"]


def test_append_custom_dns_multiple(monkeypatch):
    monkeypatch.setenv("VLLM_SR_DNS", "10.0.0.53,10.0.0.54")
    cmd = []
    container_run_command.append_custom_dns(cmd)
    assert cmd == ["--dns", "10.0.0.53", "--dns", "10.0.0.54"]


def test_append_custom_dns_trims_whitespace_and_blanks(monkeypatch):
    monkeypatch.setenv("VLLM_SR_DNS", " 10.0.0.53 , ,10.0.0.54 ")
    cmd = []
    container_run_command.append_custom_dns(cmd)
    assert cmd == ["--dns", "10.0.0.53", "--dns", "10.0.0.54"]


def test_append_custom_dns_preserves_existing_cmd(monkeypatch):
    monkeypatch.setenv("VLLM_SR_DNS", "10.0.0.53")
    cmd = ["docker", "run", "-d"]
    container_run_command.append_custom_dns(cmd)
    assert cmd == ["docker", "run", "-d", "--dns", "10.0.0.53"]


def test_append_custom_dns_does_not_interfere_with_host_gateway(monkeypatch):
    monkeypatch.setenv("VLLM_SR_DNS", "10.0.0.53")
    cmd = []
    container_run_command.append_host_gateway(cmd, "docker")
    container_run_command.append_custom_dns(cmd)
    assert cmd == [
        "--add-host=host.docker.internal:host-gateway",
        "--dns",
        "10.0.0.53",
    ]


def test_append_host_gateway_for_podman(monkeypatch):
    cmd = []
    container_run_command.append_host_gateway(cmd, "podman")
    assert cmd == ["--add-host=host.docker.internal:host-gateway"]


def test_append_nvidia_gpu_passthrough_uses_gpus_flag_for_docker(monkeypatch):
    monkeypatch.delenv("VLLM_SR_NVIDIA_GPU_PASSTHROUGH", raising=False)
    cmd = []
    container_run_command.append_nvidia_gpu_passthrough(cmd, "docker")
    assert cmd == ["--gpus", "all", "--runtime", "nvidia"]


def test_append_nvidia_gpu_passthrough_uses_cdi_for_podman(monkeypatch):
    monkeypatch.delenv("VLLM_SR_NVIDIA_GPU_PASSTHROUGH", raising=False)
    cmd = []
    container_run_command.append_nvidia_gpu_passthrough(cmd, "podman")
    assert cmd == ["--device", "nvidia.com/gpu=all"]


def test_append_nvidia_gpu_passthrough_disabled_via_env(monkeypatch):
    monkeypatch.setenv("VLLM_SR_NVIDIA_GPU_PASSTHROUGH", "0")
    cmd = []
    container_run_command.append_nvidia_gpu_passthrough(cmd, "podman")
    assert cmd == []


def test_maybe_append_nvidia_gpu_passthrough_skips_when_disabled():
    cmd = []
    container_run_command.maybe_append_nvidia_gpu_passthrough(
        cmd, enable_nvidia_gpu=False, runtime="docker"
    )
    assert cmd == []


def test_append_supplemental_gids_uses_keep_groups_for_rootless_podman_on_linux(
    monkeypatch,
):
    monkeypatch.setattr(container_run_command.sys, "platform", "linux")
    monkeypatch.setattr(container_run_command.os, "geteuid", lambda: 1000)
    cmd = []
    container_run_command.append_supplemental_gids(cmd, [100, 200], "podman")
    assert cmd == ["--group-add", "keep-groups"]


def test_append_supplemental_gids_uses_explicit_gids_for_podman_on_macos(
    monkeypatch,
):
    """macOS Podman machine runs in remote mode and rejects keep-groups (#2954)."""
    monkeypatch.setattr(container_run_command.sys, "platform", "darwin")
    monkeypatch.setattr(container_run_command.os, "geteuid", lambda: 1000)
    cmd = []
    container_run_command.append_supplemental_gids(cmd, [100, 200], "podman")
    assert cmd == ["--group-add", "100", "--group-add", "200"]


def test_append_supplemental_gids_uses_explicit_gids_for_root_podman_on_linux(
    monkeypatch,
):
    monkeypatch.setattr(container_run_command.sys, "platform", "linux")
    monkeypatch.setattr(container_run_command.os, "geteuid", lambda: 0)
    cmd = []
    container_run_command.append_supplemental_gids(cmd, [100], "podman")
    assert cmd == ["--group-add", "100"]


def test_append_supplemental_gids_uses_explicit_gids_for_docker_on_linux(
    monkeypatch,
):
    monkeypatch.setattr(container_run_command.sys, "platform", "linux")
    monkeypatch.setattr(container_run_command.os, "geteuid", lambda: 1000)
    cmd = []
    container_run_command.append_supplemental_gids(cmd, [100], "docker")
    assert cmd == ["--group-add", "100"]


def test_append_supplemental_gids_dedupes_and_preserves_order(monkeypatch):
    monkeypatch.setattr(container_run_command.sys, "platform", "linux")
    monkeypatch.setattr(container_run_command.os, "geteuid", lambda: 1000)
    cmd = []
    container_run_command.append_supplemental_gids(cmd, [100, 200, 100], "docker")
    assert cmd == ["--group-add", "100", "--group-add", "200"]


def test_append_supplemental_gids_noop_when_empty(monkeypatch):
    monkeypatch.setattr(container_run_command.sys, "platform", "linux")
    monkeypatch.setattr(container_run_command.os, "geteuid", lambda: 1000)
    cmd = []
    container_run_command.append_supplemental_gids(cmd, [], "podman")
    assert cmd == []


def test_append_supplemental_gids_rejects_non_positive_gid(monkeypatch):
    monkeypatch.setattr(container_run_command.sys, "platform", "darwin")
    monkeypatch.setattr(container_run_command.os, "geteuid", lambda: 1000)
    with pytest.raises(ValueError):
        container_run_command.append_supplemental_gids([], [0], "podman")
