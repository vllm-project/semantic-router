"""Low-level container runtime helpers for vLLM Semantic Router.

Both Docker and Podman are supported. Selection happens via:

1. The `CONTAINER_RUNTIME` environment variable (``docker`` or ``podman``).
2. Auto-detection: prefer ``docker`` if present, fall back to ``podman``.

Once resolved the choice is cached for the lifetime of the process and used
verbatim by every helper that shells out (``docker run`` / ``podman run`` etc.).
Docker and Podman expose the same CLI surface for the operations this CLI uses,
so the only places that need to know about the difference are runtime detection
and any flag that one runtime supports but the other does not.
"""

import json
import os
import shutil
import subprocess
import sys
from functools import lru_cache

from cli.consts import (
    CONTAINER_RUNTIME_DOCKER,
    CONTAINER_RUNTIME_ENV,
    CONTAINER_RUNTIME_PODMAN,
    SUPPORTED_CONTAINER_RUNTIMES,
)
from cli.utils import get_logger

log = get_logger(__name__)


def get_container_runtime():
    """Detect and return the active container runtime (``docker`` or ``podman``)."""
    return _detect_container_runtime()


def reset_container_runtime_cache() -> None:
    """Reset the cached runtime detection. Tests and the --runtime flag use this."""
    _detect_container_runtime.cache_clear()


def resolve_runtime_cli_path(runtime: str) -> str | None:
    """Return the resolved path to a runtime CLI on $PATH (or ``None``)."""
    return shutil.which(runtime)


def resolve_container_cli_path(preferred_path: str | None = None) -> str | None:
    """Resolve the docker-compatible CLI used for OpenClaw provisioning.

    When the active runtime is Podman, this still has to return a usable
    *Docker* binary because the dashboard's OpenClaw subsystem talks to a real
    Docker daemon. Returns ``None`` if no Docker CLI is found.
    """
    docker_path = preferred_path or shutil.which("docker")
    if not docker_path:
        return None
    if _docker_path_looks_like_podman(docker_path):
        # Real docker is required for OpenClaw — a Podman shim cannot satisfy it.
        return None
    return docker_path


@lru_cache(maxsize=1)
def _detect_container_runtime():
    """Detect a supported runtime, honoring the explicit ``CONTAINER_RUNTIME`` env."""
    if sys.platform.startswith("win"):
        _exit_unsupported_runtime(
            "native Windows",
            "Native Windows is not supported for local `vllm-sr serve` workflows. "
            "Run `vllm-sr serve` from WSL2 or another Linux environment with a "
            "supported container runtime.",
        )

    explicit = (os.getenv(CONTAINER_RUNTIME_ENV) or "").strip().lower()
    if explicit:
        if explicit not in SUPPORTED_CONTAINER_RUNTIMES:
            _exit_unsupported_runtime(
                f"{CONTAINER_RUNTIME_ENV}={explicit}",
                "vLLM Semantic Router local deployment requires Docker or Podman.",
            )
        runtime_path = resolve_runtime_cli_path(explicit)
        if not runtime_path:
            log.error(
                f"{CONTAINER_RUNTIME_ENV}={explicit} but `{explicit}` was not "
                "found in PATH"
            )
            sys.exit(1)
        _ensure_supported_runtime_daemon(explicit, runtime_path)
        log.info(f"Using container runtime from {CONTAINER_RUNTIME_ENV}: {explicit}")
        return explicit

    docker_path = resolve_runtime_cli_path(CONTAINER_RUNTIME_DOCKER)
    if docker_path and not _docker_path_looks_like_podman(docker_path):
        _ensure_supported_runtime_daemon(CONTAINER_RUNTIME_DOCKER, docker_path)
        log.info("Detected container runtime: docker")
        return CONTAINER_RUNTIME_DOCKER

    podman_path = resolve_runtime_cli_path(CONTAINER_RUNTIME_PODMAN)
    if podman_path:
        _ensure_supported_runtime_daemon(CONTAINER_RUNTIME_PODMAN, podman_path)
        log.info("Detected container runtime: podman")
        return CONTAINER_RUNTIME_PODMAN

    log.error("Neither docker nor podman was found in PATH")
    log.error("Please install Docker or Podman to use this tool")
    log.error("")
    log.error("Installation instructions:")
    log.error("  Docker: https://docs.docker.com/get-docker/")
    log.error("  Podman: https://podman.io/getting-started/installation")
    sys.exit(1)


def _ensure_supported_runtime_daemon(runtime: str, runtime_path: str) -> None:
    """Verify the runtime CLI can talk to its daemon/service."""
    if runtime == CONTAINER_RUNTIME_DOCKER:
        version_info = _load_docker_version_info(runtime_path)
        if _docker_daemon_looks_like_podman(version_info):
            context_name = version_info.get("Client", {}).get("Context")
            runtime_hint = (
                runtime_path
                if not context_name
                else f"{runtime_path} (context={context_name})"
            )
            log.warning(
                "The `docker` CLI is connected to a Podman-compatible daemon at "
                f"{runtime_hint}. Continuing with the `docker` CLI; set "
                f"{CONTAINER_RUNTIME_ENV}=podman to invoke `podman` directly."
            )
        return

    # Podman: a working `podman info` proves the rootful or rootless service
    # can satisfy container operations.
    try:
        result = subprocess.run(
            [runtime_path, "info", "--format", "{{json .Host.OS}}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        _exit_unavailable_runtime("podman", f"{runtime_path}: {exc}")

    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        _exit_unavailable_runtime("podman", details or runtime_path)


def _docker_path_looks_like_podman(docker_path: str) -> bool:
    resolved_path = os.path.realpath(docker_path)
    if os.path.basename(resolved_path).lower().startswith("podman"):
        return True

    try:
        result = subprocess.run(
            [docker_path, "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False

    version_output = f"{result.stdout}\n{result.stderr}".lower()
    return "podman" in version_output


def _load_docker_version_info(docker_path: str) -> dict:
    try:
        result = subprocess.run(
            [docker_path, "version", "--format", "{{json .}}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        _exit_unavailable_runtime("docker", f"{docker_path}: {exc}")

    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        _exit_unavailable_runtime("docker", details or docker_path)

    try:
        version_info = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        _exit_unavailable_runtime(
            "docker", f"unable to parse `docker version` output: {exc}"
        )

    if not version_info.get("Server"):
        _exit_unavailable_runtime(
            "docker", "Docker server details are unavailable for the active context."
        )

    return version_info


def _docker_daemon_looks_like_podman(version_info: dict) -> bool:
    server_info = version_info.get("Server") or {}
    components = server_info.get("Components") or []
    for component in components:
        name = str(component.get("Name", "")).lower()
        version = str(component.get("Version", "")).lower()
        if "podman" in name or "podman" in version:
            return True
    return False


def _exit_unsupported_runtime(runtime_hint: str, message: str) -> None:
    log.error(message)
    log.error(f"Unsupported container runtime: {runtime_hint}")
    log.error(f"Set {CONTAINER_RUNTIME_ENV}=docker or {CONTAINER_RUNTIME_ENV}=podman.")
    sys.exit(1)


def _exit_unavailable_runtime(runtime: str, details: str) -> None:
    log.error(
        f"{runtime.capitalize()} runtime is not reachable for local vLLM "
        "Semantic Router deployment."
    )
    log.error(f"{runtime.capitalize()} availability check failed: {details}")
    if runtime == CONTAINER_RUNTIME_DOCKER:
        log.error(
            "Start Docker Desktop or Docker Engine and ensure the active Docker "
            "context points to a real Docker daemon."
        )
    else:
        log.error(
            "Make sure podman is installed and `podman info` succeeds. On Linux "
            "you may need to enable the rootless service via "
            "`systemctl --user start podman.socket`, or run as root."
        )
    sys.exit(1)


def container_image_exists(image_name):
    """Check if a container image exists locally."""
    runtime = get_container_runtime()
    try:
        result = subprocess.run(
            [runtime, "images", "-q", image_name],
            capture_output=True,
            text=True,
            check=False,
        )
        return bool(result.stdout.strip())
    except Exception as exc:
        log.warning(f"Failed to check container image: {exc}")
        return False


def container_pull_image(image_name):
    """
    Pull a container image.

    Args:
        image_name: Name of the image to pull

    Returns:
        True if successful, False otherwise
    """
    runtime = get_container_runtime()
    try:
        log.info(f"Pulling container image: {image_name}")
        log.info("This may take a few minutes...")

        subprocess.run(
            [runtime, "pull", image_name],
            capture_output=False,
            text=True,
            check=True,
        )

        log.info(f"Successfully pulled: {image_name}")
        return True
    except subprocess.CalledProcessError as exc:
        log.error(f"Failed to pull image: {exc}")
        return False
    except Exception as exc:
        log.error(f"Error pulling image: {exc}")
        return False
