"""Container startup orchestration for vLLM Semantic Router."""

import os
import shutil
import socket
import subprocess
from contextlib import suppress

from cli.consts import (
    DEFAULT_NOFILE_LIMIT,
    MIN_NOFILE_LIMIT,
    PLATFORM_AMD,
    VLLM_SR_DOCKER_NAME,
)
from cli.docker_images import _normalize_platform, get_docker_image
from cli.docker_runtime import get_container_runtime
from cli.utils import getLogger

log = getLogger(__name__)


def docker_start_vllm_sr(
    config_file,
    env_vars,
    listeners,
    image=None,
    pull_policy=None,
    network_name=None,
    openclaw_network_name=None,
    minimal=False,
):
    """
    Start vLLM Semantic Router container.

    Returns:
        (return_code, stdout, stderr)
    """
    runtime = get_container_runtime()
    env_vars = dict(env_vars or {})

    normalized_platform = _resolve_platform(env_vars)
    image = get_docker_image(
        image=image, pull_policy=pull_policy, platform=normalized_platform
    )
    nofile_limit = _resolve_nofile_limit()

    cmd = _build_base_run_command(runtime, nofile_limit, network_name)
    _append_amd_gpu_passthrough(cmd, normalized_platform)
    _append_host_gateway(cmd, runtime)
    _append_listener_and_service_ports(cmd, listeners, minimal)

    config_dir = _mount_config_and_state_dirs(cmd, config_file)
    _configure_openclaw_support(
        cmd,
        env_vars,
        config_dir,
        openclaw_network_name,
        runtime,
    )

    for key, value in env_vars.items():
        cmd.extend(["-e", f"{key}={value}"])
    cmd.append(image)

    log.info(f"Starting vLLM Semantic Router container with {runtime}...")
    log.debug(f"Container command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return (0, result.stdout, result.stderr)
    except subprocess.CalledProcessError as exc:
        return (exc.returncode, exc.stdout, exc.stderr)


def _resolve_platform(env_vars):
    platform = (
        env_vars.get("DASHBOARD_PLATFORM")
        or env_vars.get("VLLM_SR_PLATFORM")
        or os.getenv("VLLM_SR_PLATFORM")
    )
    return _normalize_platform(platform)


def _resolve_nofile_limit():
    nofile_limit = int(os.getenv("VLLM_SR_NOFILE_LIMIT", DEFAULT_NOFILE_LIMIT))
    if nofile_limit < MIN_NOFILE_LIMIT:
        log.warning(
            f"File descriptor limit {nofile_limit} is below minimum {MIN_NOFILE_LIMIT}. "
            "Using minimum value."
        )
        return MIN_NOFILE_LIMIT
    if nofile_limit != DEFAULT_NOFILE_LIMIT:
        log.info(f"Using custom file descriptor limit: {nofile_limit}")
    return nofile_limit


def _build_base_run_command(runtime, nofile_limit, network_name):
    cmd = [
        runtime,
        "run",
        "-d",
        "--name",
        VLLM_SR_DOCKER_NAME,
        "--ulimit",
        f"nofile={nofile_limit}:{nofile_limit}",
    ]
    if network_name:
        cmd.extend(["--network", network_name])
    return cmd


def _append_amd_gpu_passthrough(cmd, normalized_platform):
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


def _append_host_gateway(cmd, runtime):
    if runtime == "docker":
        cmd.append("--add-host=host.docker.internal:host-gateway")
        return

    try:
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.connect(("8.8.8.8", 80))
        host_ip = udp_socket.getsockname()[0]
        udp_socket.close()
        cmd.append(f"--add-host=host.docker.internal:{host_ip}")
        log.info(f"Using host IP for Podman: {host_ip}")
    except Exception as exc:
        log.warning(f"Could not detect host IP for Podman: {exc}")
        log.info("Podman will use host.containers.internal by default")


def _append_listener_and_service_ports(cmd, listeners, minimal):
    for listener in listeners:
        port = listener.get("port")
        if port:
            cmd.extend(["-p", f"{port}:{port}"])

    cmd.extend(["-p", "50051:50051"])
    cmd.extend(["-p", "9190:9190"])
    if not minimal:
        cmd.extend(["-p", "8700:8700"])
    cmd.extend(["-p", "8080:8080"])


def _mount_config_and_state_dirs(cmd, config_file):
    config_path = os.path.abspath(config_file)
    config_dir = os.path.dirname(config_path)

    cmd.extend(["-v", f"{config_path}:/app/config.yaml:z"])

    vllm_sr_dir = os.path.join(config_dir, ".vllm-sr")
    if os.path.exists(vllm_sr_dir):
        cmd.extend(["-v", f"{vllm_sr_dir}:/app/.vllm-sr:z"])
        log.info(f"Mounting .vllm-sr directory: {vllm_sr_dir}")

    models_dir = os.path.join(config_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    cmd.extend(["-v", f"{models_dir}:/app/models:z"])
    return config_dir


def _configure_openclaw_support(
    cmd,
    env_vars,
    config_dir,
    openclaw_network_name,
    runtime,
):
    default_openclaw_data_dir = os.path.join(config_dir, ".vllm-sr", "openclaw-data")
    openclaw_data_dir = (
        env_vars.get("OPENCLAW_DATA_DIR")
        or os.getenv("OPENCLAW_DATA_DIR")
        or default_openclaw_data_dir
    )
    openclaw_data_dir = os.path.abspath(openclaw_data_dir)
    os.makedirs(openclaw_data_dir, exist_ok=True)
    cmd.extend(["-v", f"{openclaw_data_dir}:{openclaw_data_dir}:z"])
    env_vars["OPENCLAW_DATA_DIR"] = openclaw_data_dir
    log.info(f"Mounting OpenClaw data directory: {openclaw_data_dir}")

    env_vars.setdefault(
        "OPENCLAW_BASE_IMAGE",
        os.getenv("OPENCLAW_BASE_IMAGE", "ghcr.io/openclaw/openclaw:latest"),
    )
    env_vars.setdefault(
        "OPENCLAW_DEFAULT_NETWORK_MODE",
        openclaw_network_name or "vllm-sr-network",
    )

    if runtime == "docker":
        _attach_docker_socket(cmd)
        _attach_docker_cli(cmd)


def _attach_docker_socket(cmd):
    docker_socket = os.getenv("VLLM_SR_DOCKER_SOCKET")
    socket_candidates = []
    if docker_socket:
        socket_candidates.append(docker_socket)
    else:
        socket_candidates.append("/var/run/docker.sock")
        xdg_runtime_dir = os.getenv("XDG_RUNTIME_DIR")
        if xdg_runtime_dir:
            socket_candidates.append(os.path.join(xdg_runtime_dir, "docker.sock"))
        with suppress(Exception):
            socket_candidates.append(f"/run/user/{os.getuid()}/docker.sock")

    resolved_socket = next(
        (
            candidate
            for candidate in socket_candidates
            if candidate and os.path.exists(candidate)
        ),
        None,
    )
    if resolved_socket:
        cmd.extend(["-v", f"{resolved_socket}:/var/run/docker.sock"])
        log.info(f"Mounting Docker socket for dashboard OpenClaw: {resolved_socket}")
    else:
        log.warning(
            "Docker socket not found (checked: "
            f"{', '.join(socket_candidates)}); dashboard OpenClaw create/start/stop may be unavailable"
        )


def _attach_docker_cli(cmd):
    docker_bin = os.getenv("VLLM_SR_DOCKER_BIN") or shutil.which("docker")
    if not docker_bin:
        for candidate in ["/usr/local/bin/docker", "/usr/bin/docker", "/bin/docker"]:
            if os.path.exists(candidate):
                docker_bin = candidate
                break

    if docker_bin and os.path.exists(docker_bin):
        container_docker_bin = "/usr/local/bin/docker"
        cmd.extend(["-v", f"{docker_bin}:{container_docker_bin}:ro"])
        cmd.extend(["-e", f"OPENCLAW_CONTAINER_RUNTIME={container_docker_bin}"])
        log.info(f"Mounting Docker CLI for dashboard OpenClaw: {docker_bin}")
    else:
        cmd.extend(["-e", "OPENCLAW_CONTAINER_RUNTIME=docker"])
