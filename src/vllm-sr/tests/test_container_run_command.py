import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli import container_host_ports, container_run_command  # noqa: E402


def test_resolve_internal_bind_address_defaults_to_loopback(monkeypatch):
    monkeypatch.delenv("VLLM_SR_INTERNAL_BIND_ADDRESS", raising=False)

    assert container_host_ports.resolve_internal_bind_address() == "127.0.0.1"


def test_resolve_internal_bind_address_accepts_explicit_loopback(monkeypatch):
    monkeypatch.setenv("VLLM_SR_INTERNAL_BIND_ADDRESS", " ::1 ")

    assert container_host_ports.resolve_internal_bind_address() == "::1"


def test_resolve_internal_bind_address_warns_for_non_loopback(monkeypatch, caplog):
    monkeypatch.setenv("VLLM_SR_INTERNAL_BIND_ADDRESS", "0.0.0.0")

    with caplog.at_level("WARNING"):
        assert container_host_ports.resolve_internal_bind_address() == "0.0.0.0"

    assert "publishes internal, data, and observability ports" in caplog.text
    assert "host firewall controls" in caplog.text


@pytest.mark.parametrize(
    "value",
    [
        "",
        "localhost",
        "0.0.0.0:8080",
        "10.0.0.0/24",
        "fe80::1%eth0",
        "224.0.0.1",
    ],
)
def test_resolve_internal_bind_address_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("VLLM_SR_INTERNAL_BIND_ADDRESS", value)

    with pytest.raises(ValueError, match="VLLM_SR_INTERNAL_BIND_ADDRESS"):
        container_host_ports.resolve_internal_bind_address()


def test_format_port_mapping_supports_ipv4_and_ipv6():
    assert (
        container_host_ports.format_port_mapping("127.0.0.1", 8080, 80)
        == "127.0.0.1:8080:80"
    )
    assert container_host_ports.format_port_mapping("::1", 8080, 80) == "[::1]:8080:80"


def test_append_port_mappings_requires_explicit_bind_addresses():
    cmd = []

    container_run_command.append_port_mappings(
        cmd,
        [
            ("127.0.0.1", 8080, 80),
            ("0.0.0.0", 8899, 8899),
        ],
    )

    assert cmd == [
        "-p",
        "127.0.0.1:8080:80",
        "-p",
        "0.0.0.0:8899:8899",
    ]


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
