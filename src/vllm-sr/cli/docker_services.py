"""Container service and observability helpers for vLLM Semantic Router."""

import json
import os
import shutil
import subprocess

from cli.docker_runtime import get_container_runtime
from cli.utils import getLogger

log = getLogger(__name__)


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
    except Exception as exc:
        log.error(f"Failed to get container status: {exc}")
        return "error"


def docker_stop_container(container_name):
    """Stop a container."""
    runtime = get_container_runtime()
    try:
        log.info(f"Stopping container: {container_name}")
        subprocess.run(
            [runtime, "stop", container_name], check=True, capture_output=True
        )
        log.info(f"Container stopped: {container_name}")
        return True
    except subprocess.CalledProcessError as exc:
        log.error(f"Failed to stop container: {exc}")
        return False


def docker_remove_container(container_name):
    """Remove a container."""
    runtime = get_container_runtime()
    try:
        log.info(f"Removing container: {container_name}")
        subprocess.run([runtime, "rm", container_name], check=True, capture_output=True)
        log.info(f"Container removed: {container_name}")
        return True
    except subprocess.CalledProcessError as exc:
        log.error(f"Failed to remove container: {exc}")
        return False


def docker_logs(container_name, follow=False, tail=None):
    """Stream logs from a container."""
    runtime = get_container_runtime()
    cmd = [runtime, "logs"]
    if follow:
        cmd.append("-f")
    if tail:
        cmd.extend(["--tail", str(tail)])
    cmd.append(container_name)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        log.error(f"Failed to get logs: {exc}")
    except KeyboardInterrupt:
        log.info("Log streaming stopped")


def docker_logs_since(container_name, since_timestamp):
    """Get logs from a container since a specific timestamp."""
    runtime = get_container_runtime()
    cmd = [runtime, "logs", "--since", str(since_timestamp), container_name]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return (0, result.stdout, result.stderr)
    except subprocess.CalledProcessError as exc:
        return (exc.returncode, exc.stdout, exc.stderr)


def docker_exec(container_name, command):
    """Execute a command in a running container."""
    runtime = get_container_runtime()
    cmd = [runtime, "exec", container_name, *command]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return (0, result.stdout, result.stderr)
    except subprocess.CalledProcessError as exc:
        return (exc.returncode, exc.stdout, exc.stderr)


def docker_create_network(network_name):
    """Create a Docker network if it doesn't exist."""
    runtime = get_container_runtime()
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

    cmd = [runtime, "network", "create", network_name]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        log.info(f"Created network: {network_name}")
        return (0, result.stdout, result.stderr)
    except subprocess.CalledProcessError as exc:
        return (exc.returncode, exc.stdout, exc.stderr)


def docker_remove_network(network_name):
    """Remove a Docker network."""
    runtime = get_container_runtime()
    cmd = [runtime, "network", "rm", network_name]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return (0, result.stdout, result.stderr)
    except subprocess.CalledProcessError as exc:
        return (exc.returncode, exc.stdout, exc.stderr)


def docker_start_jaeger(network_name="vllm-sr-network"):
    """Start Jaeger container for distributed tracing."""
    runtime = get_container_runtime()
    container_name = "vllm-sr-jaeger"
    _replace_existing_container(container_name)

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
        "4318:4317",
        "-p",
        "16686:16686",
        "jaegertracing/all-in-one:latest",
    ]
    return _run_service_start(cmd, "Jaeger")


def docker_start_prometheus(network_name="vllm-sr-network", config_dir=None):
    """Start Prometheus container for metrics collection."""
    runtime = get_container_runtime()
    container_name = "vllm-sr-prometheus"
    _replace_existing_container(container_name)

    config_dir = _ensure_hidden_config_dir(config_dir)
    prometheus_data_dir = os.path.join(config_dir, "prometheus-data")
    os.makedirs(prometheus_data_dir, exist_ok=True)

    prometheus_tsdb_dir = os.path.join(prometheus_data_dir, "data")
    os.makedirs(prometheus_tsdb_dir, exist_ok=True)
    try:
        os.chmod(prometheus_data_dir, 0o777)
        os.chmod(prometheus_tsdb_dir, 0o777)
    except Exception as exc:
        log.warning(f"Failed to set permissions on Prometheus data directory: {exc}")
        log.warning(
            "Prometheus may fail to start if it cannot write to the data directory"
        )

    prometheus_config_dir = os.path.join(config_dir, "prometheus-config")
    os.makedirs(prometheus_config_dir, exist_ok=True)
    prometheus_config = os.path.join(prometheus_config_dir, "prometheus.yaml")
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    shutil.copy(os.path.join(template_dir, "prometheus.serve.yaml"), prometheus_config)

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
    return _run_service_start(cmd, "Prometheus")


def docker_start_grafana(network_name="vllm-sr-network", config_dir=None):
    """Start Grafana container for visualization."""
    runtime = get_container_runtime()
    container_name = "vllm-sr-grafana"
    _replace_existing_container(container_name)

    grafana_dir = os.path.join(_ensure_hidden_config_dir(config_dir), "grafana")
    os.makedirs(grafana_dir, exist_ok=True)

    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    for filename in [
        "grafana.serve.ini",
        "grafana-datasource.serve.yaml",
        "grafana-datasource-jaeger.serve.yaml",
        "grafana-dashboard.serve.yaml",
        "llm-router-dashboard.serve.json",
    ]:
        shutil.copy(
            os.path.join(template_dir, filename), os.path.join(grafana_dir, filename)
        )

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
        "-p",
        "3000:3000",
        "grafana/grafana:11.5.1",
    ]
    return _run_service_start(cmd, "Grafana")


def docker_network_disconnect(network_name, container_name):
    """Disconnect a container from a Docker network."""
    runtime = get_container_runtime()
    cmd = [runtime, "network", "disconnect", network_name, container_name]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return (0, result.stdout, result.stderr)
    except subprocess.CalledProcessError as exc:
        return (exc.returncode, exc.stdout, exc.stderr)


def docker_network_connect(network_name, container_name):
    """Connect a container to a Docker network (idempotent)."""
    runtime = get_container_runtime()
    cmd = [runtime, "network", "connect", network_name, container_name]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return (0, result.stdout, result.stderr)
    except subprocess.CalledProcessError as exc:
        if "already" in (exc.stderr or "").lower():
            return (0, exc.stdout, exc.stderr)
        return (exc.returncode, exc.stdout, exc.stderr)


def docker_start_container(container_name):
    """Start a stopped container."""
    runtime = get_container_runtime()
    try:
        log.info(f"Starting container: {container_name}")
        subprocess.run(
            [runtime, "start", container_name], check=True, capture_output=True
        )
        log.info(f"✓ Container started: {container_name}")
        return True
    except subprocess.CalledProcessError as exc:
        log.error(f"Failed to start container: {exc}")
        return False


def load_openclaw_registry(data_dir):
    """Load OpenClaw container entries from containers.json."""
    registry_path = os.path.join(data_dir, "containers.json")
    if not os.path.exists(registry_path):
        return []
    try:
        with open(registry_path) as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning(f"Failed to load OpenClaw registry: {exc}")
        return []


def _replace_existing_container(container_name):
    status = docker_container_status(container_name)
    if status != "not found":
        log.info(f"{container_name} already exists (status: {status}), cleaning up...")
        docker_stop_container(container_name)
        docker_remove_container(container_name)


def _ensure_hidden_config_dir(config_dir):
    if config_dir is None:
        config_dir = os.path.join(os.getcwd(), ".vllm-sr")
    else:
        config_dir = os.path.join(config_dir, ".vllm-sr")
    os.makedirs(config_dir, exist_ok=True)
    return config_dir


def _run_service_start(cmd, service_name):
    log.info(f"Starting {service_name} container...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return (0, result.stdout, result.stderr)
    except subprocess.CalledProcessError as exc:
        return (exc.returncode, exc.stdout, exc.stderr)
