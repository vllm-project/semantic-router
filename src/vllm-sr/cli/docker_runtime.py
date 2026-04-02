"""Low-level container runtime helpers for vLLM Semantic Router."""

import json
import os
import shutil
import subprocess
import sys
from functools import lru_cache

from cli.utils import get_logger

log = get_logger(__name__)


def get_container_runtime():
    """Detect and return the active container runtime."""
    return _detect_container_runtime()


def resolve_docker_cli_path(preferred_path: str | None = None) -> str | None:
    """Resolve a Docker CLI path and reject Podman shims masquerading as docker."""
    docker_path = preferred_path or shutil.which("docker")
    if not docker_path:
        return None

    _ensure_supported_docker_cli(docker_path)
    return docker_path


@lru_cache(maxsize=1)
def _detect_container_runtime():
    """
    Detect and return the supported container runtime.

    Returns:
        str: 'docker'

    Raises:
        SystemExit: If Docker is unavailable or resolves to Podman
    """
    env_runtime = (os.getenv("CONTAINER_RUNTIME") or "").strip()
    if env_runtime:
        normalized_runtime = env_runtime.lower()
        if normalized_runtime != "docker":
            _exit_unsupported_runtime(
                f"CONTAINER_RUNTIME={env_runtime}",
                "vLLM Semantic Router local deployment requires Docker.",
            )

        docker_path = resolve_docker_cli_path()
        if docker_path:
            _ensure_supported_docker_daemon(docker_path)
            log.info("Using container runtime from CONTAINER_RUNTIME: docker")
            return "docker"
        log.warning("CONTAINER_RUNTIME set to docker but docker not found in PATH")

    docker_path = resolve_docker_cli_path()
    if docker_path:
        _ensure_supported_docker_daemon(docker_path)
        log.info("Detected container runtime: docker")
        return "docker"

    if shutil.which("podman"):
        _exit_unsupported_runtime(
            "podman",
            "Podman is not supported for local vLLM Semantic Router deployment.",
        )

    log.error("Docker not found in PATH")
    log.error("Please install Docker Desktop or Docker Engine to use this tool")
    log.error("")
    log.error("Installation instructions:")
    log.error("  Docker: https://docs.docker.com/get-docker/")
    sys.exit(1)


def _ensure_supported_docker_cli(docker_path: str) -> None:
    if _docker_path_looks_like_podman(docker_path):
        _exit_unsupported_runtime(
            docker_path,
            "The `docker` command appears to resolve to Podman. "
            "Install Docker and ensure `docker --version` reports Docker.",
        )


def _ensure_supported_docker_daemon(docker_path: str) -> None:
    version_info = _load_docker_version_info(docker_path)
    if not _docker_daemon_looks_like_podman(version_info):
        return

    context_name = version_info.get("Client", {}).get("Context")
    runtime_hint = (
        docker_path if not context_name else f"{docker_path} (context={context_name})"
    )
    _exit_unsupported_runtime(
        runtime_hint,
        "The Docker CLI is connected to a Podman-compatible daemon. "
        "Start Docker Desktop or point Docker to a real Docker daemon.",
    )


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
        _exit_unavailable_docker_daemon(f"{docker_path}: {exc}")

    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        _exit_unavailable_docker_daemon(details or docker_path)

    try:
        version_info = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        _exit_unavailable_docker_daemon(
            f"unable to parse `docker version` output: {exc}"
        )

    if not version_info.get("Server"):
        _exit_unavailable_docker_daemon(
            "Docker server details are unavailable for the active context."
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
    log.error("Use Docker for local `vllm-sr serve` workflows.")
    sys.exit(1)


def _exit_unavailable_docker_daemon(details: str) -> None:
    log.error(
        "Docker daemon is not reachable for local vLLM Semantic Router deployment."
    )
    log.error(f"Docker daemon check failed: {details}")
    log.error(
        "Start Docker Desktop or Docker Engine and ensure the active Docker context "
        "points to a real Docker daemon."
    )
    sys.exit(1)


def docker_image_exists(image_name):
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


def docker_pull_image(image_name):
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
