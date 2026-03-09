"""Core management functions for vLLM Semantic Router."""

import time
import sys
import os
from cli.utils import getLogger, load_config, wait_for_healthy, get_envoy_port
from cli.consts import (
    VLLM_SR_DOCKER_NAME,
    HEALTH_CHECK_TIMEOUT,
    DEFAULT_API_PORT,
    DEFAULT_ENVOY_PORT,
    DEFAULT_LISTENER_PORT,
)
from cli.docker_cli import (
    docker_container_status,
    docker_stop_container,
    docker_remove_container,
    docker_start_vllm_sr,
    docker_logs,
    docker_logs_since,
    docker_exec,
    docker_create_network,
    docker_remove_network,
    docker_network_disconnect,
    docker_network_connect,
    docker_start_container,
    docker_start_jaeger,
    docker_start_prometheus,
    docker_start_grafana,
    load_openclaw_registry,
)
from cli.logo import print_vllm_logo

log = getLogger(__name__)


def _start_observability(network_name, config_dir, env_vars):
    """Start Jaeger, Prometheus and Grafana and inject their env vars."""
    log.info("Starting observability stack (Jaeger + Prometheus + Grafana)...")
    for start_fn, label in [
        (lambda: docker_start_jaeger(network_name), "Jaeger"),
        (lambda: docker_start_prometheus(network_name, config_dir), "Prometheus"),
        (lambda: docker_start_grafana(network_name, config_dir), "Grafana"),
    ]:
        rc, _, err = start_fn()
        if rc != 0:
            log.error(f"Failed to start {label}: {err}")
            sys.exit(1)
        log.info(f"{label} started successfully")
    env_vars.update(
        {
            "TARGET_JAEGER_URL": "http://vllm-sr-jaeger:16686",
            "TARGET_GRAFANA_URL": "http://vllm-sr-grafana:3000",
            "TARGET_PROMETHEUS_URL": "http://vllm-sr-prometheus:9090",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://vllm-sr-jaeger:4317",
        }
    )


def _wait_for_setup_mode_health():
    """Wait for dashboard health in setup mode then print the setup banner."""
    log.info("Setup mode detected: skipping Router and Envoy health checks")
    log.info("Waiting for Dashboard to become healthy...")
    start_time = time.time()
    healthy = False
    while time.time() - start_time < HEALTH_CHECK_TIMEOUT:
        rc, _, _ = docker_exec(
            VLLM_SR_DOCKER_NAME,
            ["curl", "-f", "-s", "http://localhost:8700/healthz"],
        )
        if rc == 0:
            healthy = True
            break
        time.sleep(2)
    if not healthy:
        log.error("Dashboard failed to become healthy in setup mode")
        docker_logs(VLLM_SR_DOCKER_NAME, follow=False, tail=100)
        sys.exit(1)
    if docker_container_status(VLLM_SR_DOCKER_NAME) == "exited":
        log.error("Container exited unexpectedly during setup mode")
        docker_logs(VLLM_SR_DOCKER_NAME, follow=False)
        sys.exit(1)
    log.info("=" * 60)
    log.info("vLLM Semantic Router setup mode is running!")
    log.info("")
    log.info("Next steps:")
    log.info("  • Open http://localhost:8700")
    log.info("  • Configure your first model in the dashboard")
    log.info("  • Activate a runnable config to enable routing")
    log.info("")
    log.info("Commands:")
    log.info("  • vllm-sr dashboard              Open dashboard in browser")
    log.info("  • vllm-sr logs <envoy|router|dashboard> [-f]")
    log.info("  • vllm-sr status [envoy|router|dashboard|all]")
    log.info("  • vllm-sr stop")
    log.info("=" * 60)


def _print_log_streams(stdout, stderr):
    """Print router log lines that contain a caller field."""
    for stream in (stdout, stderr):
        if stream:
            for line in stream.strip().split("\n"):
                if line.strip() and "caller" in line.lower():
                    print(f"  {line}")


def _wait_for_router_health():
    """Poll the router health endpoint while streaming startup logs."""
    log.info("Waiting for Router to become healthy...")
    log.info(f"Health check timeout: {HEALTH_CHECK_TIMEOUT}s")
    log.info("Showing Router logs during startup:")
    log.info("-" * 60)
    start_time = time.time()
    last_log_time = start_time
    healthy = False
    check_count = 0
    while time.time() - start_time < HEALTH_CHECK_TIMEOUT:
        check_count += 1
        _, stdout, stderr = docker_logs_since(VLLM_SR_DOCKER_NAME, int(last_log_time))
        _print_log_streams(stdout, stderr)
        last_log_time = time.time()
        rc, _, _ = docker_exec(
            VLLM_SR_DOCKER_NAME,
            ["curl", "-f", "-s", f"http://localhost:{DEFAULT_API_PORT}/health"],
        )
        if rc == 0:
            log.info("-" * 60)
            elapsed = int(time.time() - start_time)
            log.info(f"Router is healthy (after {elapsed}s, {check_count} checks)")
            healthy = True
            break
        if check_count % 10 == 0:
            elapsed = int(time.time() - start_time)
            remaining = int(HEALTH_CHECK_TIMEOUT - elapsed)
            log.info(
                f"  ... still waiting ({elapsed}s elapsed, {remaining}s remaining)"
            )
        time.sleep(2)
    if not healthy:
        log.info("-" * 60)
        log.error(f"Router failed to become healthy after {HEALTH_CHECK_TIMEOUT}s")
        log.info("Showing full container logs:")
        docker_logs(VLLM_SR_DOCKER_NAME, follow=False, tail=100)
        sys.exit(1)
    if docker_container_status(VLLM_SR_DOCKER_NAME) == "exited":
        log.error("Container exited unexpectedly")
        docker_logs(VLLM_SR_DOCKER_NAME, follow=False)
        sys.exit(1)


def _recover_openclaw_containers(shared_network_name, config_dir, env_vars):
    """Reconnect and restart OpenClaw containers stopped by a previous vllm-sr stop."""
    default_dir = os.path.join(config_dir, ".vllm-sr", "openclaw-data")
    openclaw_data_dir = os.path.abspath(
        env_vars.get("OPENCLAW_DATA_DIR")
        or os.getenv("OPENCLAW_DATA_DIR")
        or default_dir
    )
    openclaw_entries = load_openclaw_registry(openclaw_data_dir)
    if not openclaw_entries:
        return
    log.info(f"Recovering {len(openclaw_entries)} OpenClaw container(s)...")
    for entry in openclaw_entries:
        name = entry.get("name") or entry.get("containerName")
        if not name:
            continue
        cstatus = docker_container_status(name)
        if cstatus == "not found":
            log.warning(f"OpenClaw container {name} no longer exists, skipping")
            continue
        rc, _, _ = docker_network_connect(shared_network_name, name)
        if rc == 0:
            log.info(f"✓ Connected {name} to {shared_network_name}")
        else:
            log.warning(f"Failed to connect {name} to {shared_network_name}")
        if cstatus != "running":
            log.info(f"Starting OpenClaw container: {name}")
            docker_start_container(name)


def _print_success_banner(listeners, enable_observability, dashboard_disabled):
    """Print the running banner with endpoints and a sample curl command."""
    log.info("=" * 60)
    log.info("vLLM Semantic Router is running!")
    log.info("")
    log.info("Endpoints:")
    if not dashboard_disabled:
        log.info(f"  • Dashboard: http://localhost:8700")
    for listener in listeners:
        name = listener.get("name", "unknown")
        port = listener.get("port", "unknown")
        log.info(f"  • {name}: http://localhost:{port}")
    log.info(f"  • Metrics: http://localhost:9190/metrics")
    if enable_observability:
        log.info("")
        log.info("Observability:")
        log.info(f"  • Jaeger UI: http://localhost:16686")
        log.info(f"  • Grafana: http://localhost:3000 (admin/admin)")
        log.info(f"  • Prometheus: http://localhost:9090")
    log.info("")
    log.info("Commands:")
    if not dashboard_disabled:
        log.info("  • vllm-sr dashboard              Open dashboard in browser")
    log.info("  • vllm-sr logs <envoy|router|dashboard> [-f]")
    log.info("  • vllm-sr status [envoy|router|dashboard|all]")
    log.info("  • vllm-sr stop")
    log.info("=" * 60)
    if listeners:
        first_port = listeners[0].get("port", DEFAULT_LISTENER_PORT)
        print()
        print("Test with curl:")
        print()
        print(f"curl -v http://localhost:{first_port}/v1/chat/completions \\")
        print('  -H "Content-Type: application/json" \\')
        print("  -d '{")
        print('    "model": "MoM",')
        print('    "messages": [')
        print('      {"role": "user", "content": "What is the derivative of x^2?"}')
        print("    ]")
        print("  }'")
        print()


def start_vllm_sr(
    config_file, env_vars=None, image=None, pull_policy=None, enable_observability=True
):
    """Start vLLM Semantic Router from config_file."""
    if env_vars is None:
        env_vars = {}

    print_vllm_logo()
    config = load_config(config_file)
    listeners = config.get("listeners", [])
    if not listeners:
        log.error("No listeners configured in config.yaml")
        sys.exit(1)

    log.info(f"Starting vLLM Semantic Router")
    log.info(f"Config file: {config_file}")
    log.info(f"Configured listeners:")
    for listener in listeners:
        name = listener.get("name", "unknown")
        port = listener.get("port", "unknown")
        address = listener.get("address", "0.0.0.0")
        log.info(f"  - {name}: {address}:{port}")

    status = docker_container_status(VLLM_SR_DOCKER_NAME)
    if status != "not found":
        log.info(f"Existing container found (status: {status}), cleaning up...")
        docker_stop_container(VLLM_SR_DOCKER_NAME)
        docker_remove_container(VLLM_SR_DOCKER_NAME)

    shared_network_name = "vllm-sr-network"
    config_dir = os.path.dirname(os.path.abspath(config_file))

    # OpenClaw containers and the dashboard need a stable shared bridge network
    # even when observability is disabled. Always start the main container on
    # this bridge so docker_network_connect (called below) is idempotent and
    # does not fail due to incompatible rootless-Podman network modes (e.g. pasta).
    return_code, _, stderr = docker_create_network(shared_network_name)
    if return_code != 0:
        log.error(f"Failed to create shared OpenClaw network: {stderr}")
        sys.exit(1)

    if enable_observability:
        _start_observability(shared_network_name, config_dir, env_vars)

    dashboard_disabled = env_vars.get("DISABLE_DASHBOARD") == "true"
    setup_mode = str(env_vars.get("VLLM_SR_SETUP_MODE", "")).lower() == "true"

    return_code, _, stderr = docker_start_vllm_sr(
        config_file,
        env_vars,
        listeners,
        image=image,
        pull_policy=pull_policy,
        network_name=shared_network_name,
        openclaw_network_name=shared_network_name,
        minimal=dashboard_disabled,
    )
    if return_code != 0:
        log.error(f"Failed to start container: {stderr}")
        sys.exit(1)
    log.info("vLLM Semantic Router container started successfully")

    rc, _, connect_err = docker_network_connect(
        shared_network_name, VLLM_SR_DOCKER_NAME
    )
    if rc != 0:
        log.error(
            f"Failed to connect {VLLM_SR_DOCKER_NAME} to {shared_network_name}: {connect_err}"
        )
        docker_stop_container(VLLM_SR_DOCKER_NAME)
        docker_remove_container(VLLM_SR_DOCKER_NAME)
        sys.exit(1)
    log.info(f"✓ Connected {VLLM_SR_DOCKER_NAME} to {shared_network_name}")

    if setup_mode:
        if dashboard_disabled:
            log.error("Setup mode started without dashboard enabled")
            sys.exit(1)
        _wait_for_setup_mode_health()
        return

    _wait_for_router_health()
    _recover_openclaw_containers(shared_network_name, config_dir, env_vars)
    _print_success_banner(listeners, enable_observability, dashboard_disabled)


def stop_vllm_sr():
    """Stop vLLM Semantic Router and observability containers."""
    log.info("Stopping vLLM Semantic Router...")

    status = docker_container_status(VLLM_SR_DOCKER_NAME)
    if status == "not found":
        log.info("Container not found. Nothing to stop.")
        return

    # Resolve OpenClaw data directory (same logic as start_vllm_sr)
    config_dir = os.getcwd()
    default_openclaw_data_dir = os.path.join(config_dir, ".vllm-sr", "openclaw-data")
    openclaw_data_dir = os.getenv("OPENCLAW_DATA_DIR") or default_openclaw_data_dir
    openclaw_data_dir = os.path.abspath(openclaw_data_dir)
    network_name = "vllm-sr-network"

    # Stop and disconnect OpenClaw containers before removing the network.
    # Containers are stopped but NOT removed so they can be recovered on next serve.
    openclaw_entries = load_openclaw_registry(openclaw_data_dir)
    for entry in openclaw_entries:
        name = entry.get("name") or entry.get("containerName")
        if not name:
            continue
        cstatus = docker_container_status(name)
        if cstatus == "not found":
            continue
        if cstatus == "running":
            log.info(f"Stopping OpenClaw container: {name}")
            docker_stop_container(name)
        log.info(f"Disconnecting {name} from {network_name}")
        docker_network_disconnect(network_name, name)

    if status == "running":
        docker_stop_container(VLLM_SR_DOCKER_NAME)

    docker_remove_container(VLLM_SR_DOCKER_NAME)
    log.info("vLLM Semantic Router stopped")

    # Stop observability containers if they exist
    observability_containers = [
        "vllm-sr-grafana",
        "vllm-sr-prometheus",
        "vllm-sr-jaeger",
    ]

    for container_name in observability_containers:
        status = docker_container_status(container_name)
        if status != "not found":
            log.info(f"Stopping {container_name}...")
            if status == "running":
                docker_stop_container(container_name)
            docker_remove_container(container_name)
            log.info(f"{container_name} stopped")

    # Remove network (now clean — OpenClaw containers already disconnected)
    return_code, stdout, stderr = docker_remove_network(network_name)
    if return_code == 0:
        log.info(f"Network {network_name} removed")


def _grep_pattern_for_service(service: str) -> str:
    """Return the grep pattern used to filter container logs for a given service."""
    if service == "router":
        return r'"caller"|spawned: \'router\'|success: router|cli\.commands'
    if service == "dashboard":
        return r"dashboard|Dashboard|spawned: \'dashboard\'|success: dashboard|:8700"
    # envoy: match envoy timestamp format and supervisor messages
    return r"\[2[0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].*\]\[.*\]|spawned: \'envoy\'|success: envoy"


def show_logs(service: str, follow: bool = False):
    """
    Show logs from vLLM Semantic Router service.

    Args:
        service: Service to show logs for ('envoy', 'router', or 'dashboard')
        follow: Whether to follow log output
    """
    import subprocess

    if service not in ["envoy", "router", "dashboard"]:
        log.error(f"Invalid service: {service}")
        log.error("Must be 'envoy', 'router', or 'dashboard'")
        sys.exit(1)

    if docker_container_status(VLLM_SR_DOCKER_NAME) == "not found":
        log.error("Container not found. Is vLLM Semantic Router running?")
        log.info("Start it with: vllm-sr serve")
        sys.exit(1)

    grep_pattern = _grep_pattern_for_service(service)

    if follow:
        log.info(f"Following {service} logs (Ctrl+C to stop)...")
        log.info("")
        try:
            subprocess.run(
                f'docker logs -f {VLLM_SR_DOCKER_NAME} 2>&1 | grep -E "{grep_pattern}"',
                shell=True,
            )
        except KeyboardInterrupt:
            log.info("\nStopped following logs")
    else:
        try:
            cmd = f'docker logs --tail 200 {VLLM_SR_DOCKER_NAME} 2>&1 | grep -E "{grep_pattern}" | tail -50'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            else:
                log.info(f"No recent {service} logs found")
        except Exception as e:
            log.error(f"Failed to get {service} logs: {e}")
            sys.exit(1)


def _log_service_status(label: str, cmd: list, ok_stdout: str = ""):
    """Run a health-check curl inside the container and log Running / unknown."""
    try:
        rc, stdout, _ = docker_exec(VLLM_SR_DOCKER_NAME, cmd)
        healthy = rc == 0 and (not ok_stdout or stdout.strip() in ok_stdout.split(","))
        if healthy:
            log.info(f"{label}: Running")
        else:
            log.info(f"⚠ {label}: Status unknown")
    except Exception as e:
        log.error(f"Failed to check {label} status: {e}")


def show_status(service: str = "all"):
    """Show status of vLLM Semantic Router services."""
    status = docker_container_status(VLLM_SR_DOCKER_NAME)
    if status == "not found":
        log.info("Status: Not running")
        log.info("Start with: vllm-sr serve")
        return
    if status == "exited":
        log.info("Status: Container exited (error)")
        log.info("View logs with: vllm-sr logs <envoy|router>")
        return
    if status != "running":
        log.info(f"Status: {status}")
        return

    log.info("=" * 60)
    log.info("Container Status: Running")
    log.info("")

    if service in ["all", "router"]:
        _log_service_status(
            "Router",
            ["curl", "-f", "-s", f"http://localhost:{DEFAULT_API_PORT}/health"],
        )
    if service in ["all", "envoy"]:
        _log_service_status(
            "Envoy",
            [
                "curl",
                "-f",
                "-s",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                f"http://localhost:{DEFAULT_ENVOY_PORT}/ready",
            ],
            ok_stdout="200",
        )
    if service in ["all", "dashboard"]:
        _log_service_status(
            "Dashboard",
            [
                "curl",
                "-f",
                "-s",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                "http://localhost:8700",
            ],
            ok_stdout="200,301,302",
        )

    log.info("")
    log.info("For detailed logs: vllm-sr logs <envoy|router|dashboard>")
    log.info("=" * 60)
