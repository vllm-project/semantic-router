"""Container startup orchestration for vLLM Semantic Router."""

import os
import shutil
import subprocess
from contextlib import suppress

from cli.consts import (
    DEFAULT_NOFILE_LIMIT,
    MIN_NOFILE_LIMIT,
    PLATFORM_AMD,
)
from cli.docker_images import _normalize_platform, get_docker_image
from cli.docker_runtime import get_container_runtime
from cli.runtime_stack import RuntimeStackLayout, resolve_runtime_stack
from cli.utils import get_logger

log = get_logger(__name__)


def docker_start_vllm_sr(
    config_file,
    env_vars,
    listeners,
    image=None,
    pull_policy=None,
    network_name=None,
    openclaw_network_name=None,
    minimal=False,
    stack_layout: RuntimeStackLayout | None = None,
    state_root_dir: str | None = None,
    runtime_config_file: str | None = None,
):
    """
    Start vLLM Semantic Router container.

    Returns:
        (return_code, stdout, stderr)
    """
    runtime = get_container_runtime()
    env_vars = dict(env_vars or {})
    stack_layout = stack_layout or resolve_runtime_stack()

    normalized_platform = _resolve_platform(env_vars)
    image = get_docker_image(
        image=image, pull_policy=pull_policy, platform=normalized_platform
    )
    nofile_limit = _resolve_nofile_limit()

    cmd = _build_base_run_command(runtime, nofile_limit, network_name, stack_layout)
    _append_amd_gpu_passthrough(cmd, normalized_platform)
    _append_host_gateway(cmd)
    _append_listener_and_service_ports(cmd, listeners, minimal, stack_layout)

    config_dir, runtime_container_config = _mount_config_and_state_dirs(
        cmd,
        config_file,
        runtime_config_file=runtime_config_file,
        state_root_dir=state_root_dir,
    )
    if runtime_container_config:
        env_vars.setdefault("VLLM_SR_RUNTIME_CONFIG_PATH", runtime_container_config)
    _configure_openclaw_support(
        cmd,
        env_vars,
        config_dir,
        openclaw_network_name,
        stack_layout,
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


def _build_base_run_command(runtime, nofile_limit, network_name, stack_layout):
    cmd = [
        runtime,
        "run",
        "-d",
        "--name",
        stack_layout.container_name,
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


def _append_host_gateway(cmd):
    cmd.append("--add-host=host.docker.internal:host-gateway")


def _append_listener_and_service_ports(cmd, listeners, minimal, stack_layout):
    for listener in listeners:
        port = listener.get("port")
        if port:
            cmd.extend(["-p", f"{port + stack_layout.port_offset}:{port}"])

    cmd.extend(["-p", f"{stack_layout.router_port}:50051"])
    cmd.extend(["-p", f"{stack_layout.metrics_port}:9190"])
    if not minimal:
        cmd.extend(["-p", f"{stack_layout.dashboard_port}:8700"])
    cmd.extend(["-p", f"{stack_layout.api_port}:8080"])


def _mount_config_and_state_dirs(
    cmd,
    config_file,
    runtime_config_file=None,
    state_root_dir=None,
):
    source_config_path = os.path.abspath(config_file)
    runtime_config_path = os.path.abspath(runtime_config_file or config_file)
    config_dir = (
        os.path.abspath(state_root_dir)
        if state_root_dir
        else os.path.dirname(source_config_path)
    )

    cmd.extend(["-v", f"{source_config_path}:/app/config.yaml:z"])

    vllm_sr_dir = os.path.join(config_dir, ".vllm-sr")
    if os.path.exists(vllm_sr_dir):
        cmd.extend(["-v", f"{vllm_sr_dir}:/app/.vllm-sr:z"])
        log.info(f"Mounting .vllm-sr directory: {vllm_sr_dir}")

    models_dir = os.path.join(config_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    cmd.extend(["-v", f"{models_dir}:/app/models:z"])

    dashboard_data_dir = os.path.join(config_dir, ".vllm-sr", "dashboard-data")
    os.makedirs(dashboard_data_dir, exist_ok=True)
    cmd.extend(["-v", f"{dashboard_data_dir}:/app/data:z"])
    log.info(f"Mounting dashboard data directory: {dashboard_data_dir}")

    runtime_container_config = None
    if runtime_config_path != source_config_path:
        runtime_container_config = (
            f"/app/.vllm-sr/{os.path.basename(runtime_config_path)}"
        )
        log.info(
            "Using source config %s with runtime override %s",
            source_config_path,
            runtime_container_config,
        )

    return config_dir, runtime_container_config


def _configure_openclaw_support(
    cmd,
    env_vars,
    config_dir,
    openclaw_network_name,
    stack_layout,
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
        openclaw_network_name or stack_layout.network_name,
    )

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
