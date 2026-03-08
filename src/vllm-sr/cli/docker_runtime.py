"""Low-level container runtime helpers for vLLM Semantic Router."""

import os
import shutil
import subprocess
import sys
from functools import lru_cache

from cli.utils import getLogger

log = getLogger(__name__)


def get_container_runtime():
    """Detect and return the active container runtime."""
    return _detect_container_runtime()


@lru_cache(maxsize=1)
def _detect_container_runtime():
    """
    Detect and return the available container runtime (docker or podman).

    Returns:
        str: 'docker' or 'podman'

    Raises:
        SystemExit: If neither docker nor podman is available
    """
    env_runtime = os.getenv("CONTAINER_RUNTIME")
    if env_runtime:
        normalized_runtime = env_runtime.lower()
        if normalized_runtime in ["docker", "podman"]:
            if shutil.which(normalized_runtime):
                log.info(
                    "Using container runtime from CONTAINER_RUNTIME: "
                    f"{normalized_runtime}"
                )
                return normalized_runtime
            log.warning(f"CONTAINER_RUNTIME set to {env_runtime} but not found in PATH")

    if shutil.which("docker"):
        log.info("Detected container runtime: docker")
        return "docker"
    if shutil.which("podman"):
        log.info("Detected container runtime: podman")
        return "podman"

    log.error("Neither docker nor podman found in PATH")
    log.error("Please install Docker or Podman to use this tool")
    log.error("")
    log.error("Installation instructions:")
    log.error("  Docker: https://docs.docker.com/get-docker/")
    log.error("  Podman: https://podman.io/getting-started/installation")
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
