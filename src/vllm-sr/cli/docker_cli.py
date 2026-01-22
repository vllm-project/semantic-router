"""Docker CLI operations for vLLM Semantic Router."""

import subprocess
import os
import sys
import shutil
import socket
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
    DEFAULT_NOFILE_LIMIT,
    MIN_NOFILE_LIMIT,
)

log = getLogger(__name__)

# Global variable to cache the detected container runtime
_container_runtime = None


def get_container_runtime():
    """
    Detect and return the available container runtime (docker or podman).

    Returns:
        str: 'docker' or 'podman'

    Raises:
        SystemExit: If neither docker nor podman is available
    """
    global _container_runtime

    # Return cached value if already detected
    if _container_runtime is not None:
        return _container_runtime

    # Check for explicit environment variable
    env_runtime = os.getenv("CONTAINER_RUNTIME")
    if env_runtime:
        if env_runtime.lower() in ["docker", "podman"]:
            if shutil.which(env_runtime.lower()):
                _container_runtime = env_runtime.lower()
                log.info(
                    f"Using container runtime from CONTAINER_RUNTIME: {_container_runtime}"
                )
                return _container_runtime
            else:
                log.warning(
                    f"CONTAINER_RUNTIME set to {env_runtime} but not found in PATH"
                )

    # Auto-detect: prefer docker, fallback to podman
    if shutil.which("docker"):
        _container_runtime = "docker"
        log.info("Detected container runtime: docker")
    elif shutil.which("podman"):
        _container_runtime = "podman"
        log.info("Detected container runtime: podman")
    else:
        log.error("Neither docker nor podman found in PATH")
        log.error("Please install Docker or Podman to use this tool")
        log.error("")
        log.error("Installation instructions:")
        log.error("  Docker: https://docs.docker.com/get-docker/")
        log.error("  Podman: https://podman.io/getting-started/installation")
        sys.exit(1)

    return _container_runtime


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
    runtime = get_container_runtime()
    log.error("=" * 70)
    log.error("Container image not found!")
    log.error("=" * 70)
    log.error("")
    log.error(f"Image: {image_name}")
    log.error("")
    log.error("Options:")
    log.error("")
    log.error("  1. Pull the image:")
    log.error(f"     {runtime} pull {image_name}")
    log.error("")
    log.error("  2. Use custom image:")
    log.error("     vllm-sr serve config.yaml --image your-image:tag")
    log.error("")
    log.error("  3. Change pull policy to always:")
    log.error("     vllm-sr serve config.yaml --image-pull-policy always")
    log.error("")
    log.error("=" * 70)


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
    except Exception as e:
        log.warning(f"Failed to check container image: {e}")
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
    Get the status of a container.

    Returns:
        'running', 'exited', 'paused', or 'not found'
    """
    runtime = get_container_runtime()
    try:
        result = subprocess.run(
            [
                runtime,
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
    """Stop a container."""
    runtime = get_container_runtime()
    try:
        log.info(f"Stopping container: {container_name}")
        subprocess.run(
            [runtime, "stop", container_name], check=True, capture_output=True
        )
        log.info(f"✓ Container stopped: {container_name}")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to stop container: {e}")
        return False


def docker_remove_container(container_name):
    """Remove a container."""
    runtime = get_container_runtime()
    try:
        log.info(f"Removing container: {container_name}")
        subprocess.run([runtime, "rm", container_name], check=True, capture_output=True)
        log.info(f"✓ Container removed: {container_name}")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to remove container: {e}")
        return False


def docker_start_vllm_sr(
    config_file, env_vars, listeners, image=None, pull_policy=None, network_name=None
):
    """
    Start vLLM Semantic Router container.

    Args:
        config_file: Path to config.yaml
        env_vars: Environment variables dict
        listeners: List of listener configurations from config.yaml
        image: Container image to use (optional)
        pull_policy: Image pull policy (optional)
        network_name: Docker network name (optional, for observability)

    Returns:
        (return_code, stdout, stderr)
    """
    runtime = get_container_runtime()

    # Get and validate image
    image = get_docker_image(image=image, pull_policy=pull_policy)

    # File descriptor limit
    nofile_limit = int(os.getenv("VLLM_SR_NOFILE_LIMIT", DEFAULT_NOFILE_LIMIT))

    # Validate limit
    if nofile_limit < MIN_NOFILE_LIMIT:
        log.warning(
            f"File descriptor limit {nofile_limit} is below minimum {MIN_NOFILE_LIMIT}. "
            f"Using minimum value."
        )
        nofile_limit = MIN_NOFILE_LIMIT

    if nofile_limit != DEFAULT_NOFILE_LIMIT:
        log.info(f"Using custom file descriptor limit: {nofile_limit}")

    # Build container run command
    cmd = [
        runtime,
        "run",
        "-d",  # Detached mode
        "--name",
        VLLM_SR_DOCKER_NAME,
        # Set ulimits for file descriptors to handle Envoy's high connection count
        # Default 1024 is too low and causes "Too many open files" errors
        "--ulimit",
        f"nofile={nofile_limit}:{nofile_limit}",
    ]

    # Add network if specified (for observability)
    if network_name:
        cmd.extend(["--network", network_name])

    # Add host gateway (syntax differs between docker and podman)
    if runtime == "docker":
        cmd.append("--add-host=host.docker.internal:host-gateway")
    else:  # podman
        # Podman: Use host network mode or detect host IP
        # For Podman, we have several options:
        # 1. Use --network=host (simplest but exposes all ports)
        # 2. Use host.containers.internal (available by default in Podman)
        # 3. Manually detect and add host IP

        # Option: Try to get the host IP for podman
        try:
            # Get host IP by connecting to a public DNS
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            host_ip = s.getsockname()[0]
            s.close()
            cmd.append(f"--add-host=host.docker.internal:{host_ip}")
            log.info(f"Using host IP for Podman: {host_ip}")
        except Exception as e:
            log.warning(f"Could not detect host IP for Podman: {e}")
            log.info("Podman will use host.containers.internal by default")
            # Podman provides host.containers.internal by default, so we don't need to add anything

    # Add port mappings for each listener
    for listener in listeners:
        port = listener.get("port")
        if port:
            cmd.extend(["-p", f"{port}:{port}"])  # Map host:container port

    # Add internal service ports
    cmd.extend(["-p", "50051:50051"])  # Router gRPC port (internal)
    cmd.extend(["-p", "9190:9190"])  # Metrics port
    cmd.extend(["-p", "8700:8700"])  # Dashboard UI
    cmd.extend(["-p", "8080:8080"])  # Router API port

    # Mount config file (read-write to allow dashboard edits)
    # Use :z for SELinux compatibility on Fedora/RHEL systems
    cmd.extend(
        [
            "-v",
            f"{os.path.abspath(config_file)}:/app/config.yaml:z",
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
                f"{vllm_sr_dir}:/app/.vllm-sr:z",
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
            f"{models_dir}:/app/models:z",
        ]
    )

    # Add environment variables
    for key, value in env_vars.items():
        cmd.extend(["-e", f"{key}={value}"])

    # Add image name
    cmd.append(image)

    log.info(f"Starting vLLM Semantic Router container with {runtime}...")
    log.debug(f"Container command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return (0, result.stdout, result.stderr)
    except subprocess.CalledProcessError as e:
        return (e.returncode, e.stdout, e.stderr)


def docker_logs(container_name, follow=False, tail=None):
    """
    Stream logs from a container.

    Args:
        container_name: Name of the container
        follow: Whether to follow logs (tail -f behavior)
        tail: Number of lines to show from the end (e.g., "100", "all")
    """
    runtime = get_container_runtime()
    cmd = [runtime, "logs"]
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
    Get logs from a container since a specific timestamp.

    Args:
        container_name: Name of the container
        since_timestamp: Unix timestamp to get logs since

    Returns:
        (return_code, stdout, stderr)
    """
    runtime = get_container_runtime()
    cmd = [runtime, "logs", "--since", str(since_timestamp), container_name]

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
    runtime = get_container_runtime()
    cmd = [runtime, "exec", container_name] + command

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return (0, result.stdout, result.stderr)
    except subprocess.CalledProcessError as e:
        return (e.returncode, e.stdout, e.stderr)


def docker_create_network(network_name):
    """
    Create a Docker network if it doesn't exist.

    Args:
        network_name: Name of the network

    Returns:
        (return_code, stdout, stderr)
    """
    runtime = get_container_runtime()

    # Check if network exists
    cmd = [
        runtime,
        "network",
        "ls",
        "--filter",
        f"name={network_name}",
        "--format",
        "{{.Name}}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if network_name in result.stdout:
            log.debug(f"Network {network_name} already exists")
            return (0, "", "")
    except subprocess.CalledProcessError:
        pass

    # Create network
    cmd = [runtime, "network", "create", network_name]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        log.info(f"Created network: {network_name}")
        return (0, result.stdout, result.stderr)
    except subprocess.CalledProcessError as e:
        return (e.returncode, e.stdout, e.stderr)


def docker_remove_network(network_name):
    """
    Remove a Docker network.

    Args:
        network_name: Name of the network

    Returns:
        (return_code, stdout, stderr)
    """
    runtime = get_container_runtime()
    cmd = [runtime, "network", "rm", network_name]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return (0, result.stdout, result.stderr)
    except subprocess.CalledProcessError as e:
        return (e.returncode, e.stdout, e.stderr)


def docker_start_jaeger(network_name="vllm-sr-network"):
    """
    Start Jaeger container for distributed tracing.

    Args:
        network_name: Docker network name

    Returns:
        (return_code, stdout, stderr)
    """
    runtime = get_container_runtime()
    container_name = "vllm-sr-jaeger"

    # Check if container already exists
    status = docker_container_status(container_name)
    if status != "not found":
        log.info(f"Jaeger container already exists (status: {status}), cleaning up...")
        docker_stop_container(container_name)
        docker_remove_container(container_name)

    cmd = [
        runtime,
        "run",
        "-d",
        "--name",
        container_name,
        "--network",
        network_name,
        "-e",
        "COLLECTOR_OTLP_ENABLED=true",
        "-p",
        "4318:4317",  # OTLP gRPC (mapped to 4318 on host to avoid conflicts)
        "-p",
        "16686:16686",  # Web UI
        "jaegertracing/all-in-one:latest",
    ]

    log.info("Starting Jaeger container...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return (0, result.stdout, result.stderr)
    except subprocess.CalledProcessError as e:
        return (e.returncode, e.stdout, e.stderr)


def docker_start_prometheus(network_name="vllm-sr-network", config_dir=None):
    """
    Start Prometheus container for metrics collection.

    Args:
        network_name: Docker network name
        config_dir: Directory containing prometheus.yaml config

    Returns:
        (return_code, stdout, stderr)
    """
    runtime = get_container_runtime()
    container_name = "vllm-sr-prometheus"

    # Check if container already exists
    status = docker_container_status(container_name)
    if status != "not found":
        log.info(
            f"Prometheus container already exists (status: {status}), cleaning up..."
        )
        docker_stop_container(container_name)
        docker_remove_container(container_name)

    # Prepare Prometheus config and data directory
    # Always use .vllm-sr subdirectory
    if config_dir is None:
        config_dir = os.path.join(os.getcwd(), ".vllm-sr")
    else:
        # If config_dir is provided, append .vllm-sr subdirectory
        config_dir = os.path.join(config_dir, ".vllm-sr")
    os.makedirs(config_dir, exist_ok=True)

    # Create Prometheus data directory for persistent storage
    prometheus_data_dir = os.path.join(config_dir, "prometheus-data")
    os.makedirs(prometheus_data_dir, exist_ok=True)

    # Create data subdirectory inside prometheus-data for TSDB storage
    # This ensures Prometheus can create queries.active and other files
    prometheus_tsdb_dir = os.path.join(prometheus_data_dir, "data")
    os.makedirs(prometheus_tsdb_dir, exist_ok=True)

    # Set permissions for Prometheus container (runs as nobody:nobody, UID/GID 65534)
    # This allows Prometheus to write to the data directory
    try:
        os.chmod(prometheus_data_dir, 0o777)
        os.chmod(prometheus_tsdb_dir, 0o777)
    except Exception as e:
        log.warning(f"Failed to set permissions on Prometheus data directory: {e}")
        log.warning(
            "Prometheus may fail to start if it cannot write to the data directory"
        )

    # Store prometheus.yaml inside config subdirectory
    prometheus_config_dir = os.path.join(config_dir, "prometheus-config")
    os.makedirs(prometheus_config_dir, exist_ok=True)
    prometheus_config = os.path.join(prometheus_config_dir, "prometheus.yaml")
    # Always copy template to ensure we use the latest configuration
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    template_file = os.path.join(template_dir, "prometheus.serve.yaml")
    shutil.copy(template_file, prometheus_config)

    cmd = [
        runtime,
        "run",
        "-d",
        "--name",
        container_name,
        "--network",
        network_name,
        "-v",
        f"{os.path.abspath(prometheus_config)}:/etc/prometheus/prometheus.yaml:ro",
        "-v",
        f"{os.path.abspath(prometheus_data_dir)}:/prometheus",
        "-p",
        "9090:9090",
        "prom/prometheus:v2.53.0",
        "--config.file=/etc/prometheus/prometheus.yaml",
        "--storage.tsdb.path=/prometheus/data",
        "--storage.tsdb.retention.time=15d",
    ]

    log.info("Starting Prometheus container...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return (0, result.stdout, result.stderr)
    except subprocess.CalledProcessError as e:
        return (e.returncode, e.stdout, e.stderr)


def docker_start_grafana(network_name="vllm-sr-network", config_dir=None):
    """
    Start Grafana container for visualization.

    Args:
        network_name: Docker network name
        config_dir: Directory containing Grafana configs

    Returns:
        (return_code, stdout, stderr)
    """
    runtime = get_container_runtime()
    container_name = "vllm-sr-grafana"

    # Check if container already exists
    status = docker_container_status(container_name)
    if status != "not found":
        log.info(f"Grafana container already exists (status: {status}), cleaning up...")
        docker_stop_container(container_name)
        docker_remove_container(container_name)

    # Prepare Grafana configs
    # Always use .vllm-sr subdirectory
    if config_dir is None:
        config_dir = os.path.join(os.getcwd(), ".vllm-sr")
    else:
        # If config_dir is provided, append .vllm-sr subdirectory
        config_dir = os.path.join(config_dir, ".vllm-sr")
    grafana_dir = os.path.join(config_dir, "grafana")
    os.makedirs(grafana_dir, exist_ok=True)

    # Copy Grafana configuration files (always overwrite to ensure latest config)
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    for filename in [
        "grafana.serve.ini",
        "grafana-datasource.serve.yaml",
        "grafana-datasource-jaeger.serve.yaml",
        "grafana-dashboard.serve.yaml",
        "llm-router-dashboard.serve.json",
    ]:
        src = os.path.join(template_dir, filename)
        dst = os.path.join(grafana_dir, filename)
        # Always copy to ensure we use the latest configuration
        shutil.copy(src, dst)

    # Note: Grafana data is NOT persisted - container is ephemeral
    # Only Prometheus data is persisted for metrics history

    cmd = [
        runtime,
        "run",
        "-d",
        "--name",
        container_name,
        "--network",
        network_name,
        "-e",
        "GF_SECURITY_ADMIN_USER=admin",
        "-e",
        "GF_SECURITY_ADMIN_PASSWORD=admin",
        "-e",
        "PROMETHEUS_URL=vllm-sr-prometheus:9090",
        "-v",
        f"{os.path.abspath(os.path.join(grafana_dir, 'grafana.serve.ini'))}:/etc/grafana/grafana.ini:ro",
        "-v",
        f"{os.path.abspath(os.path.join(grafana_dir, 'grafana-datasource.serve.yaml'))}:/etc/grafana/provisioning/datasources/datasource.yaml:ro",
        "-v",
        f"{os.path.abspath(os.path.join(grafana_dir, 'grafana-datasource-jaeger.serve.yaml'))}:/etc/grafana/provisioning/datasources/datasource_jaeger.yaml:ro",
        "-v",
        f"{os.path.abspath(os.path.join(grafana_dir, 'grafana-dashboard.serve.yaml'))}:/etc/grafana/provisioning/dashboards/dashboard.yaml:ro",
        "-v",
        f"{os.path.abspath(os.path.join(grafana_dir, 'llm-router-dashboard.serve.json'))}:/etc/grafana/provisioning/dashboards/llm-router-dashboard.json:ro",
        # No volume mount for /var/lib/grafana - Grafana data is ephemeral
        "-p",
        "3000:3000",
        "grafana/grafana:11.5.1",
    ]

    log.info("Starting Grafana container...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return (0, result.stdout, result.stderr)
    except subprocess.CalledProcessError as e:
        return (e.returncode, e.stdout, e.stderr)
