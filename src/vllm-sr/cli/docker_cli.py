"""Docker CLI operations for vLLM Semantic Router."""

import subprocess
import os
import sys
from cli.utils import getLogger
from cli.consts import (
    VLLM_SR_DOCKER_NAME,
    VLLM_SR_DOCKER_IMAGE_DEFAULT,
    VLLM_SR_DOCKER_IMAGE_DEV,
    VLLM_SR_DOCKER_IMAGE_RELEASE,
    IMAGE_PULL_POLICY_ALWAYS,
    IMAGE_PULL_POLICY_IF_NOT_PRESENT,
    IMAGE_PULL_POLICY_NEVER,
    DEFAULT_IMAGE_PULL_POLICY,
)

log = getLogger(__name__)


def get_docker_image(image=None, pull_policy=None):
    """
    Determine which Docker image to use and handle pulling if needed.

    Priority:
    1. Explicit image parameter (--image)
    2. VLLM_SR_IMAGE environment variable
    3. Default image (ghcr.io/vllm-project/semantic-router/vllm-sr:latest)

    Args:
        image: Explicit image name (optional)
        pull_policy: Image pull policy - 'always', 'ifnotpresent', 'never' (optional)

    Returns:
        Docker image name

    Raises:
        SystemExit if no suitable image found or pull fails
    """
    if pull_policy is None:
        pull_policy = DEFAULT_IMAGE_PULL_POLICY

    # Determine which image to use
    selected_image = None

    # 1. Check explicit image parameter
    if image:
        selected_image = image
        log.info(f"Using specified image: {selected_image}")

    # 2. Check environment variable
    elif os.getenv("VLLM_SR_IMAGE"):
        selected_image = os.getenv("VLLM_SR_IMAGE")
        log.info(f"Using image from VLLM_SR_IMAGE: {selected_image}")

    # 3. Use default image
    else:
        selected_image = VLLM_SR_DOCKER_IMAGE_DEFAULT
        log.info(f"Using default image: {selected_image}")

    # Handle pull policy
    image_exists = docker_image_exists(selected_image)

    if pull_policy == IMAGE_PULL_POLICY_ALWAYS:
        # Always pull
        log.info(f"Pull policy: always - pulling image...")
        if not docker_pull_image(selected_image):
            log.error(f"Failed to pull image: {selected_image}")
            sys.exit(1)

    elif pull_policy == IMAGE_PULL_POLICY_IF_NOT_PRESENT:
        # Pull only if not present
        if not image_exists:
            log.info(f"Image not found locally, pulling...")
            if not docker_pull_image(selected_image):
                log.error(f"Failed to pull image: {selected_image}")
                _show_image_not_found_error(selected_image)
                sys.exit(1)
        else:
            log.info(f"✓ Image exists locally: {selected_image}")

    elif pull_policy == IMAGE_PULL_POLICY_NEVER:
        # Never pull, error if not exists
        if not image_exists:
            log.error(f"Image not found locally: {selected_image}")
            log.error(f"Pull policy is 'never', cannot pull image")
            _show_image_not_found_error(selected_image)
            sys.exit(1)
        else:
            log.info(f"✓ Image exists locally: {selected_image}")

    return selected_image


def _show_image_not_found_error(image_name):
    """Show helpful error message when image is not found."""
    log.error("=" * 70)
    log.error("Docker image not found!")
    log.error("=" * 70)
    log.error("")
    log.error(f"Image: {image_name}")
    log.error("")
    log.error("Options:")
    log.error("")
    log.error("  1. Pull the image:")
    log.error(f"     docker pull {image_name}")
    log.error("")
    log.error("  2. Use custom image:")
    log.error("     vllm-sr serve config.yaml --image your-image:tag")
    log.error("")
    log.error("  3. Change pull policy to always:")
    log.error("     vllm-sr serve config.yaml --image-pull-policy always")
    log.error("")
    log.error("=" * 70)


def docker_image_exists(image_name):
    """Check if a Docker image exists locally."""
    try:
        result = subprocess.run(
            ["docker", "images", "-q", image_name],
            capture_output=True,
            text=True,
            check=False,
        )
        return bool(result.stdout.strip())
    except Exception as e:
        log.warning(f"Failed to check Docker image: {e}")
        return False


def docker_pull_image(image_name):
    """
    Pull a Docker image.

    Args:
        image_name: Name of the image to pull

    Returns:
        True if successful, False otherwise
    """
    try:
        log.info(f"Pulling Docker image: {image_name}")
        log.info("This may take a few minutes...")

        result = subprocess.run(
            ["docker", "pull", image_name],
            capture_output=False,  # Show pull progress
            text=True,
            check=True,
        )

        log.info(f"✓ Successfully pulled: {image_name}")
        return True

    except subprocess.CalledProcessError as e:
        log.error(f"Failed to pull image: {e}")
        return False
    except Exception as e:
        log.error(f"Error pulling image: {e}")
        return False


def docker_container_status(container_name):
    """
    Get the status of a Docker container.

    Returns:
        'running', 'exited', 'paused', or 'not found'
    """
    try:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                f"name={container_name}",
                "--format",
                "{{.Status}}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        status = result.stdout.strip()
        if not status:
            return "not found"
        if "Up" in status:
            return "running"
        if "Exited" in status:
            return "exited"
        if "Paused" in status:
            return "paused"
        return "unknown"
    except Exception as e:
        log.error(f"Failed to get container status: {e}")
        return "error"


def docker_stop_container(container_name):
    """Stop a Docker container."""
    try:
        log.info(f"Stopping container: {container_name}")
        subprocess.run(
            ["docker", "stop", container_name], check=True, capture_output=True
        )
        log.info(f"✓ Container stopped: {container_name}")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to stop container: {e}")
        return False


def docker_remove_container(container_name):
    """Remove a Docker container."""
    try:
        log.info(f"Removing container: {container_name}")
        subprocess.run(
            ["docker", "rm", container_name], check=True, capture_output=True
        )
        log.info(f"✓ Container removed: {container_name}")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to remove container: {e}")
        return False


def docker_start_vllm_sr(
    config_file, env_vars, listeners, image=None, pull_policy=None
):
    """
    Start vLLM Semantic Router Docker container.

    Args:
        config_file: Path to config.yaml
        env_vars: Environment variables dict
        listeners: List of listener configurations from config.yaml
        image: Docker image to use (optional)
        pull_policy: Image pull policy (optional)

    Returns:
        (return_code, stdout, stderr)
    """
    # Get and validate image
    image = get_docker_image(image=image, pull_policy=pull_policy)

    # Build docker run command
    cmd = [
        "docker",
        "run",
        "-d",  # Detached mode
        "--name",
        VLLM_SR_DOCKER_NAME,
        "--add-host=host.docker.internal:host-gateway",
    ]

    # Add port mappings for each listener
    for listener in listeners:
        port = listener.get("port")
        if port:
            cmd.extend(["-p", f"{port}:{port}"])  # Map host:container port

    # Add internal service ports
    cmd.extend(
        [
            "-p",
            "50051:50051",  # Router gRPC port (internal)
            "-p",
            "9190:9190",  # Metrics port
            # Note: 8080 (Router API) is not exposed by default
            # Health checks are done inside the container via docker exec
        ]
    )

    # Mount config file
    cmd.extend(
        [
            "-v",
            f"{os.path.abspath(config_file)}:/app/config.yaml:ro",
        ]
    )

    # Mount .vllm-sr directory for user-customizable router defaults
    # This allows users to modify router-defaults.yaml (e.g., external_models)
    config_dir = os.path.dirname(os.path.abspath(config_file))
    vllm_sr_dir = os.path.join(config_dir, ".vllm-sr")
    if os.path.exists(vllm_sr_dir):
        cmd.extend(
            [
                "-v",
                f"{vllm_sr_dir}:/app/.vllm-sr",
            ]
        )
        log.info(f"Mounting .vllm-sr directory: {vllm_sr_dir}")

    # Mount models directory for caching downloaded models
    # This allows models to persist across container restarts
    models_dir = os.path.join(config_dir, "models")
    os.makedirs(models_dir, exist_ok=True)  # Create if doesn't exist
    cmd.extend(
        [
            "-v",
            f"{models_dir}:/app/models",
        ]
    )

    # Add environment variables
    for key, value in env_vars.items():
        cmd.extend(["-e", f"{key}={value}"])

    # Add image name
    cmd.append(image)

    log.info(f"Starting vLLM Semantic Router container...")
    log.debug(f"Docker command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return (0, result.stdout, result.stderr)
    except subprocess.CalledProcessError as e:
        return (e.returncode, e.stdout, e.stderr)


def docker_logs(container_name, follow=False, tail=None):
    """
    Stream logs from a Docker container.

    Args:
        container_name: Name of the container
        follow: Whether to follow logs (tail -f behavior)
        tail: Number of lines to show from the end (e.g., "100", "all")
    """
    cmd = ["docker", "logs"]
    if follow:
        cmd.append("-f")
    if tail:
        cmd.extend(["--tail", str(tail)])
    cmd.append(container_name)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to get logs: {e}")
    except KeyboardInterrupt:
        log.info("Log streaming stopped")


def docker_logs_since(container_name, since_timestamp):
    """
    Get logs from a Docker container since a specific timestamp.

    Args:
        container_name: Name of the container
        since_timestamp: Unix timestamp to get logs since

    Returns:
        (return_code, stdout, stderr)
    """
    cmd = ["docker", "logs", "--since", str(since_timestamp), container_name]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return (0, result.stdout, result.stderr)
    except subprocess.CalledProcessError as e:
        return (e.returncode, e.stdout, e.stderr)


def docker_exec(container_name, command):
    """
    Execute a command in a running container.

    Args:
        container_name: Name of the container
        command: Command to execute (list)

    Returns:
        (return_code, stdout, stderr)
    """
    cmd = ["docker", "exec", container_name] + command

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return (0, result.stdout, result.stderr)
    except subprocess.CalledProcessError as e:
        return (e.returncode, e.stdout, e.stderr)
