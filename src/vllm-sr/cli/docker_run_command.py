"""Helpers for assembling Docker run/create commands."""

import os

from cli.consts import PLATFORM_AMD, PLATFORM_NVIDIA
from cli.docker_images import _normalize_platform
from cli.utils import get_logger

log = get_logger(__name__)


def build_base_run_command(
    runtime, nofile_limit, network_name, container_name, *, start_immediately
):
    cmd = [runtime]
    if start_immediately:
        cmd.extend(["run", "-d"])
    else:
        cmd.append("create")
    cmd.extend(
        [
            "--name",
            container_name,
            "--ulimit",
            f"nofile={nofile_limit}:{nofile_limit}",
        ]
    )
    if network_name:
        cmd.extend(["--network", network_name])
    return cmd


def append_amd_gpu_passthrough(cmd, normalized_platform):
    if normalized_platform != PLATFORM_AMD:
        return

    passthrough_enabled = os.getenv("VLLM_SR_AMD_GPU_PASSTHROUGH", "1").lower()
    if passthrough_enabled in ["0", "false", "no", "off"]:
        log.info(
            "AMD GPU passthrough disabled by VLLM_SR_AMD_GPU_PASSTHROUGH environment variable"
        )
        return

    required_devices = ["/dev/kfd", "/dev/dri"]
    mounted_devices = []
    missing_devices = []
    for device in required_devices:
        if os.path.exists(device):
            cmd.extend(["--device", device])
            mounted_devices.append(device)
        else:
            missing_devices.append(device)

    if mounted_devices:
        cmd.extend(["--group-add", "video"])
        cmd.extend(["--cap-add", "SYS_PTRACE"])
        cmd.extend(["--security-opt", "seccomp=unconfined"])
        log.info(
            f"AMD GPU passthrough enabled with devices: {', '.join(mounted_devices)}"
        )

    if missing_devices:
        log.warning(
            "Platform 'amd' selected but missing AMD GPU devices on host: "
            f"{', '.join(missing_devices)}. Container may fall back to CPU."
        )


def append_host_gateway(cmd, runtime):
    if runtime == "docker":
        cmd.append("--add-host=host.docker.internal:host-gateway")


def append_mount_specs(cmd, mount_specs: list[str]):
    for mount_spec in mount_specs:
        cmd.extend(["-v", mount_spec])


def append_port_mappings(cmd, port_mappings: list[tuple[int, int]]):
    for host_port, container_port in port_mappings:
        cmd.extend(["-p", f"{host_port}:{container_port}"])


def append_env_vars(cmd, env_vars: dict[str, str]):
    for key, value in env_vars.items():
        cmd.extend(["-e", f"{key}={value}"])


def maybe_append_amd_gpu_passthrough(cmd, enable_amd_gpu: bool):
    if enable_amd_gpu:
        append_amd_gpu_passthrough(cmd, _normalize_platform(PLATFORM_AMD))


def append_nvidia_gpu_passthrough(cmd):
    passthrough_enabled = os.getenv("VLLM_SR_NVIDIA_GPU_PASSTHROUGH", "1").lower()
    if passthrough_enabled in ["0", "false", "no", "off"]:
        log.info(
            "NVIDIA GPU passthrough disabled by VLLM_SR_NVIDIA_GPU_PASSTHROUGH environment variable"
        )
        return

    cmd.extend(["--gpus", "all"])
    cmd.extend(["--runtime", "nvidia"])
    log.info("NVIDIA GPU passthrough enabled (--gpus all --runtime nvidia)")


def maybe_append_nvidia_gpu_passthrough(cmd, enable_nvidia_gpu: bool):
    if enable_nvidia_gpu:
        append_nvidia_gpu_passthrough(cmd)
