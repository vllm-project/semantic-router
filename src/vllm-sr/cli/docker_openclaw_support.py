"""OpenClaw runtime support for docker-start orchestration."""

from __future__ import annotations

import os
from contextlib import suppress

from cli.utils import get_logger

log = get_logger(__name__)


def configure_openclaw_support(
    mount_specs,
    env_vars,
    config_dir,
    openclaw_network_name,
    runtime,
    stack_layout,
    *,
    resolve_docker_cli,
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

    if runtime == "docker":
        _attach_docker_socket(mount_specs)
        _attach_docker_cli(
            mount_specs,
            env_vars,
            resolve_docker_cli=resolve_docker_cli,
        )


def _attach_docker_socket(mount_specs):
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
        mount_specs.append(f"{resolved_socket}:/var/run/docker.sock")
        log.info(f"Mounting Docker socket for dashboard OpenClaw: {resolved_socket}")
        return

    log.warning(
        "Docker socket not found (checked: "
        f"{', '.join(socket_candidates)}); dashboard OpenClaw create/start/stop may be unavailable"
    )


def _attach_docker_cli(mount_specs, env_vars, *, resolve_docker_cli):
    mount_host_cli = _should_mount_host_docker_cli()
    if not mount_host_cli:
        env_vars["OPENCLAW_CONTAINER_RUNTIME"] = "docker"
        log.info("Using in-image Docker CLI for dashboard container management")
        return

    docker_bin = resolve_docker_cli(os.getenv("VLLM_SR_DOCKER_BIN"))
    if not docker_bin:
        for candidate in ["/usr/local/bin/docker", "/usr/bin/docker", "/bin/docker"]:
            docker_bin = resolve_docker_cli(candidate)
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
    requested_mount = os.getenv("VLLM_SR_MOUNT_DOCKER_CLI")
    if requested_mount:
        log.warning(
            "VLLM_SR_MOUNT_DOCKER_CLI requested a host Docker CLI mount, "
            "but no supported Docker CLI was found; falling back to in-image Docker CLI"
        )
        return

    log.warning(
        "Host Docker CLI was not found; falling back to the in-image Docker CLI for "
        "dashboard container management"
    )


def _should_mount_host_docker_cli() -> bool:
    raw = (os.getenv("VLLM_SR_MOUNT_DOCKER_CLI") or "").strip().lower()
    if raw == "":
        return True
    return raw in {"1", "true", "yes", "on"}
