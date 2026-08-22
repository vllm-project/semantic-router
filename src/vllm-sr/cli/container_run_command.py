"""Helpers for assembling Docker run/create commands."""

import os
import sys

from cli.consts import PLATFORM_AMD
from cli.container_images import _normalize_platform
from cli.utils import get_logger

log = get_logger(__name__)
ORDINARY_PORT_MAPPING_LENGTH = 2


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
    """Map host.docker.internal to the container host.

    Docker has supported ``--add-host=...:host-gateway`` since 20.10. Podman
    has supported the same magic value since 4.7. Both runtimes use the same
    flag spelling, so we add it unconditionally for the supported runtimes.
    """
    if runtime in ("docker", "podman"):
        cmd.append("--add-host=host.docker.internal:host-gateway")


def append_custom_dns(cmd):
    """Inject --dns servers from the comma-separated VLLM_SR_DNS env var."""
    for raw_dns in os.getenv("VLLM_SR_DNS", "").split(","):
        dns = raw_dns.strip()
        if dns:
            cmd.extend(["--dns", dns])


def append_mount_specs(cmd, mount_specs: list[str]):
    for mount_spec in mount_specs:
        cmd.extend(["-v", mount_spec])


def append_supplemental_gids(cmd, gids: list[int], runtime: str):
    """Grant only the resolved host groups needed by a service."""

    if (
        gids
        and runtime == "podman"
        and os.geteuid() != 0
        and sys.platform.startswith("linux")
    ):
        # Rootless Podman remaps numeric container GIDs. keep-groups asks crun
        # to retain the invoking user's real supplementary groups, including
        # the private spool GID, without making the host file world-writable.
        # macOS Podman machine runs in remote mode and rejects keep-groups.
        cmd.extend(["--group-add", "keep-groups"])
        return
    for gid in dict.fromkeys(gids):
        if gid <= 0:
            raise ValueError("supplemental container GIDs must be positive")
        cmd.extend(["--group-add", str(gid)])


def append_port_mappings(
    cmd,
    port_mappings: list[tuple[int, int] | tuple[str, int, int]],
):
    """Append ordinary or explicitly host-bound published ports."""
    for mapping in port_mappings:
        if len(mapping) == ORDINARY_PORT_MAPPING_LENGTH:
            host_port, container_port = mapping
            rendered = f"{host_port}:{container_port}"
        else:
            host_address, host_port, container_port = mapping
            if ":" in host_address and not host_address.startswith("["):
                host_address = f"[{host_address}]"
            rendered = f"{host_address}:{host_port}:{container_port}"
        cmd.extend(["-p", rendered])


def append_env_vars(
    cmd,
    env_vars: dict[str, str],
    inherited_env_keys: set[str] | None = None,
):
    inherited_env_keys = inherited_env_keys or set()
    for key, value in env_vars.items():
        cmd.extend(["-e", key if key in inherited_env_keys else f"{key}={value}"])


def maybe_append_amd_gpu_passthrough(cmd, enable_amd_gpu: bool):
    if enable_amd_gpu:
        append_amd_gpu_passthrough(cmd, _normalize_platform(PLATFORM_AMD))


def append_nvidia_gpu_passthrough(cmd, runtime: str):
    passthrough_enabled = os.getenv("VLLM_SR_NVIDIA_GPU_PASSTHROUGH", "1").lower()
    if passthrough_enabled in ["0", "false", "no", "off"]:
        log.info(
            "NVIDIA GPU passthrough disabled by VLLM_SR_NVIDIA_GPU_PASSTHROUGH environment variable"
        )
        return

    if runtime == "podman":
        # Podman uses CDI (`--device nvidia.com/gpu=all`); fall back to passing
        # `--gpus all` only for Docker, which understands the nvidia container
        # toolkit shim natively.
        cmd.extend(["--device", "nvidia.com/gpu=all"])
        log.info("NVIDIA GPU passthrough enabled via CDI (--device nvidia.com/gpu=all)")
        return

    cmd.extend(["--gpus", "all"])
    cmd.extend(["--runtime", "nvidia"])
    log.info("NVIDIA GPU passthrough enabled (--gpus all --runtime nvidia)")


def maybe_append_nvidia_gpu_passthrough(cmd, enable_nvidia_gpu: bool, runtime: str):
    if enable_nvidia_gpu:
        append_nvidia_gpu_passthrough(cmd, runtime)
