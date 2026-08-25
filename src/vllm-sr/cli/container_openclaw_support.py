"""OpenClaw runtime support for docker-start orchestration."""

from __future__ import annotations

import os
import stat

from cli.utils import get_logger

log = get_logger(__name__)

OPENCLAW_CONTAINER_RUNTIME_DISABLED_ENV = "OPENCLAW_CONTAINER_RUNTIME_DISABLED"
CONTAINER_SOCKET_ENV = "VLLM_SR_CONTAINER_SOCKET"
_CONTAINER_SOCKET_MODE = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP


def configure_openclaw_support(
    mount_specs,
    env_vars,
    config_dir,
    openclaw_network_name,
    runtime,
    stack_layout,
    *,
    resolve_container_cli,
):
    default_openclaw_data_dir = os.path.join(config_dir, ".vllm-sr", "openclaw-data")
    openclaw_data_dir = (
        env_vars.get("OPENCLAW_DATA_DIR")
        or os.getenv("OPENCLAW_DATA_DIR")
        or default_openclaw_data_dir
    )
    openclaw_data_dir = os.path.abspath(openclaw_data_dir)
    os.makedirs(openclaw_data_dir, exist_ok=True)
    mount_specs.append(f"{openclaw_data_dir}:{openclaw_data_dir}:z")
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

    env_vars[OPENCLAW_CONTAINER_RUNTIME_DISABLED_ENV] = "true"
    socket_path = _explicit_container_socket_path()
    if socket_path is None:
        log.info(
            "Dashboard OpenClaw container management is disabled by default. "
            f"Set {CONTAINER_SOCKET_ENV} to an absolute daemon socket path to opt "
            "in; daemon access grants host-equivalent privilege."
        )
        return

    if runtime == "docker":
        if _attach_container_socket(mount_specs, runtime, socket_path):
            env_vars[OPENCLAW_CONTAINER_RUNTIME_DISABLED_ENV] = "false"
            _attach_container_cli(
                mount_specs,
                env_vars,
                resolve_container_cli=resolve_container_cli,
            )
    elif runtime == "podman" and _attach_container_socket(
        mount_specs, runtime, socket_path
    ):
        # Podman exposes a Docker-Engine-API-compatible socket. The dashboard
        # image already ships a real `docker` CLI; mounting podman.sock at the
        # canonical /var/run/docker.sock path lets the in-image docker CLI
        # drive container lifecycle (start/stop/inspect/logs) through podman
        # transparently — no Go-side changes needed.
        env_vars[OPENCLAW_CONTAINER_RUNTIME_DISABLED_ENV] = "false"
        env_vars["OPENCLAW_CONTAINER_RUNTIME"] = "docker"
        log.info(
            "Podman runtime: dashboard will use the in-image Docker CLI against "
            "the mounted podman.sock for container lifecycle"
        )


def _explicit_container_socket_path() -> str | None:
    """Return the explicitly opted-in daemon socket, or disable the feature."""

    raw_path = os.getenv(CONTAINER_SOCKET_ENV)
    if raw_path is None or raw_path == "":
        return None
    if raw_path != raw_path.strip():
        raise ValueError(f"{CONTAINER_SOCKET_ENV} must not contain whitespace padding")
    if not os.path.isabs(raw_path) or os.path.normpath(raw_path) != raw_path:
        raise ValueError(
            f"{CONTAINER_SOCKET_ENV} must be an absolute canonical socket path"
        )
    return raw_path


def _attach_container_socket(mount_specs, runtime: str, socket_path: str) -> bool:
    """Mount one explicitly selected daemon socket at /var/run/docker.sock.

    Both Docker and Podman expose a Docker-Engine-API-compatible UNIX socket.
    We always mount it at the canonical container path /var/run/docker.sock so
    the dashboard's in-image docker CLI works without runtime-specific config.
    This opt-in grants the Dashboard host-equivalent daemon privileges; the
    filesystem checks below constrain socket sharing, not daemon capabilities.
    """
    if not os.path.exists(socket_path):
        log.warning(
            f"Configured {runtime} socket does not exist: {socket_path}; Dashboard "
            "OpenClaw container management remains disabled"
        )
        return False
    if not _runtime_socket_is_group_safe(socket_path):
        log.warning(
            "Dashboard OpenClaw container management is unavailable because "
            f"the explicitly configured {runtime} socket has an unsafe owner, "
            "type, group, or mode; Router and Dashboard startup will continue "
            "without the socket"
        )
        return False
    mount_specs.append(f"{socket_path}:/var/run/docker.sock")
    log.warning(
        f"Explicitly mounting {runtime} socket {socket_path} for Dashboard OpenClaw. "
        "Daemon access grants host-equivalent privilege."
    )
    return True


def _runtime_socket_is_group_safe(path: str, *, lstat_path=os.lstat) -> bool:
    """Validate socket identity and sharing mode before the privileged opt-in.

    Passing this check does not reduce the daemon's host-equivalent capability.
    """

    try:
        info = lstat_path(path)
    except OSError:
        return False
    current_uid = None
    get_effective_uid = getattr(os, "geteuid", None)
    if get_effective_uid is not None:
        current_uid = get_effective_uid()
    allowed_owners = {0}
    if current_uid is not None:
        allowed_owners.add(current_uid)
    return (
        stat.S_ISSOCK(info.st_mode)
        and info.st_uid in allowed_owners
        and info.st_gid != 0
        and stat.S_IMODE(info.st_mode) == _CONTAINER_SOCKET_MODE
    )


def _attach_container_cli(mount_specs, env_vars, *, resolve_container_cli):
    mount_host_cli = _should_mount_host_container_cli()
    if not mount_host_cli:
        env_vars["OPENCLAW_CONTAINER_RUNTIME"] = "docker"
        log.info("Using in-image Docker CLI for dashboard container management")
        return

    docker_bin = resolve_container_cli(os.getenv("VLLM_SR_CONTAINER_BIN"))
    if not docker_bin:
        for candidate in ["/usr/local/bin/docker", "/usr/bin/docker", "/bin/docker"]:
            docker_bin = resolve_container_cli(candidate)
            if docker_bin:
                break

    if docker_bin and os.path.exists(docker_bin):
        container_docker_bin = "/usr/local/bin/docker"
        mount_specs.append(f"{docker_bin}:{container_docker_bin}:ro")
        env_vars["OPENCLAW_CONTAINER_RUNTIME"] = container_docker_bin
        log.info(
            f"Mounting host Docker CLI for dashboard container management: {docker_bin}"
        )
        return

    env_vars["OPENCLAW_CONTAINER_RUNTIME"] = "docker"
    requested_mount = os.getenv("VLLM_SR_MOUNT_CONTAINER_CLI")
    if requested_mount:
        log.warning(
            "VLLM_SR_MOUNT_CONTAINER_CLI requested a host Docker CLI mount, "
            "but no supported Docker CLI was found; falling back to in-image Docker CLI"
        )
        return

    log.warning(
        "Host Docker CLI was not found; falling back to the in-image Docker CLI for "
        "dashboard container management"
    )


def _should_mount_host_container_cli() -> bool:
    raw = (os.getenv("VLLM_SR_MOUNT_CONTAINER_CLI") or "").strip().lower()
    if raw == "":
        return True
    return raw in {"1", "true", "yes", "on"}
