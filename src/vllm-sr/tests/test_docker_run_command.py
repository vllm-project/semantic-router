import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli import docker_run_command  # noqa: E402


def test_append_custom_dns_noop_when_unset(monkeypatch):
    monkeypatch.delenv("VLLM_SR_DNS", raising=False)
    cmd = []
    docker_run_command.append_custom_dns(cmd)
    assert cmd == []


def test_append_custom_dns_noop_when_empty(monkeypatch):
    monkeypatch.setenv("VLLM_SR_DNS", "")
    cmd = []
    docker_run_command.append_custom_dns(cmd)
    assert cmd == []


def test_append_custom_dns_single(monkeypatch):
    monkeypatch.setenv("VLLM_SR_DNS", "10.0.0.53")
    cmd = []
    docker_run_command.append_custom_dns(cmd)
    assert cmd == ["--dns", "10.0.0.53"]


def test_append_custom_dns_multiple(monkeypatch):
    monkeypatch.setenv("VLLM_SR_DNS", "10.0.0.53,10.0.0.54")
    cmd = []
    docker_run_command.append_custom_dns(cmd)
    assert cmd == ["--dns", "10.0.0.53", "--dns", "10.0.0.54"]


def test_append_custom_dns_trims_whitespace_and_blanks(monkeypatch):
    monkeypatch.setenv("VLLM_SR_DNS", " 10.0.0.53 , ,10.0.0.54 ")
    cmd = []
    docker_run_command.append_custom_dns(cmd)
    assert cmd == ["--dns", "10.0.0.53", "--dns", "10.0.0.54"]


def test_append_custom_dns_preserves_existing_cmd(monkeypatch):
    monkeypatch.setenv("VLLM_SR_DNS", "10.0.0.53")
    cmd = ["docker", "run", "-d"]
    docker_run_command.append_custom_dns(cmd)
    assert cmd == ["docker", "run", "-d", "--dns", "10.0.0.53"]


def test_append_custom_dns_does_not_interfere_with_host_gateway(monkeypatch):
    monkeypatch.setenv("VLLM_SR_DNS", "10.0.0.53")
    cmd = []
    docker_run_command.append_host_gateway(cmd, "docker")
    docker_run_command.append_custom_dns(cmd)
    assert cmd == [
        "--add-host=host.docker.internal:host-gateway",
        "--dns",
        "10.0.0.53",
    ]


def test_append_host_gateway_for_podman(monkeypatch):
    cmd = []
    docker_run_command.append_host_gateway(cmd, "podman")
    assert cmd == ["--add-host=host.docker.internal:host-gateway"]


def test_append_nvidia_gpu_passthrough_uses_gpus_flag_for_docker(monkeypatch):
    monkeypatch.delenv("VLLM_SR_NVIDIA_GPU_PASSTHROUGH", raising=False)
    cmd = []
    docker_run_command.append_nvidia_gpu_passthrough(cmd, "docker")
    assert cmd == ["--gpus", "all", "--runtime", "nvidia"]


def test_append_nvidia_gpu_passthrough_uses_cdi_for_podman(monkeypatch):
    monkeypatch.delenv("VLLM_SR_NVIDIA_GPU_PASSTHROUGH", raising=False)
    cmd = []
    docker_run_command.append_nvidia_gpu_passthrough(cmd, "podman")
    assert cmd == ["--device", "nvidia.com/gpu=all"]


def test_append_nvidia_gpu_passthrough_disabled_via_env(monkeypatch):
    monkeypatch.setenv("VLLM_SR_NVIDIA_GPU_PASSTHROUGH", "0")
    cmd = []
    docker_run_command.append_nvidia_gpu_passthrough(cmd, "podman")
    assert cmd == []


def test_maybe_append_nvidia_gpu_passthrough_skips_when_disabled():
    cmd = []
    docker_run_command.maybe_append_nvidia_gpu_passthrough(
        cmd, enable_nvidia_gpu=False, runtime="docker"
    )
    assert cmd == []

