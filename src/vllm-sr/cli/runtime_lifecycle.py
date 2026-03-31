"""Startup and readiness helpers for vLLM Semantic Router runtime."""

from __future__ import annotations

import os
import time
from collections.abc import Callable

from cli.consts import (
    DEFAULT_API_PORT,
    DEFAULT_LISTENER_PORT,
    HEALTH_CHECK_INTERVAL,
    HEALTH_CHECK_TIMEOUT,
)
from cli.docker_cli import (
    docker_container_status,
    docker_create_network,
    docker_exec,
    docker_logs,
    docker_logs_since,
    docker_network_connect,
    docker_remove_container,
    docker_start_container,
    docker_start_fleet_sim,
    docker_start_grafana,
    docker_start_jaeger,
    docker_start_prometheus,
    docker_stop_container,
    load_openclaw_registry,
)
from cli.runtime_stack import RuntimeStackLayout
from cli.utils import get_logger

log = get_logger(__name__)

ServiceStarter = Callable[[], tuple[int, str, str]]


def log_startup_banner(
    config_file, listeners, stack_layout: RuntimeStackLayout
) -> None:
    """Log the selected runtime stack and configured listener endpoints."""
    log.info("Starting vLLM Semantic Router")
    log.info(
        f"Runtime stack: {stack_layout.stack_name} (port offset {stack_layout.port_offset})"
    )
    log.info(f"Config file: {config_file}")
    log.info("Configured listeners:")
    for listener in listeners:
        name = listener.get("name", "unknown")
        port = listener.get("port", "unknown")
        address = listener.get("address", "0.0.0.0")
        log.info(f"  - {name}: {address}:{port}")


def ensure_clean_runtime_container(container_name: str) -> None:
    """Stop and remove any existing runtime container before restarting."""
    status = docker_container_status(container_name)
    if status == "not found":
        return
    log.info(f"Existing container found (status: {status}), cleaning up...")
    if status in {"running", "paused"}:
        docker_stop_container(container_name)
    docker_remove_container(container_name)


def ensure_shared_network(shared_network_name: str) -> None:
    """Create the shared OpenClaw bridge network used by local stacks."""
    return_code, _stdout, stderr = docker_create_network(shared_network_name)
    if return_code != 0:
        log.error(f"Failed to create shared OpenClaw network: {stderr}")
        raise SystemExit(1)


def start_observability_stack(
    enable_observability: bool,
    shared_network_name: str,
    config_dir: str,
    env_vars: dict[str, str],
    stack_layout: RuntimeStackLayout,
) -> str | None:
    """Start Jaeger, Prometheus, and Grafana when observability is enabled."""
    if not enable_observability:
        return None

    log.info("Starting observability stack (Jaeger + Prometheus + Grafana)...")
    _start_named_service(
        "Jaeger",
        lambda: docker_start_jaeger(shared_network_name, stack_layout=stack_layout),
    )
    _start_named_service(
        "Prometheus",
        lambda: docker_start_prometheus(
            shared_network_name, config_dir, stack_layout=stack_layout
        ),
    )
    _start_named_service(
        "Grafana",
        lambda: docker_start_grafana(
            shared_network_name, config_dir, stack_layout=stack_layout
        ),
    )

    env_vars.update(
        {
            "TARGET_JAEGER_URL": stack_layout.jaeger_service_url,
            "TARGET_GRAFANA_URL": stack_layout.grafana_service_url,
            "TARGET_PROMETHEUS_URL": stack_layout.prometheus_service_url,
            "OTEL_EXPORTER_OTLP_ENDPOINT": stack_layout.otlp_service_endpoint,
        }
    )
    return shared_network_name


def start_fleet_sim_sidecar(
    config_dir: str,
    env_vars: dict[str, str],
    stack_layout: RuntimeStackLayout,
    pull_policy: str | None = None,
) -> bool:
    """Start the local simulator sidecar unless an external URL is configured."""
    external_url = env_vars.get("TARGET_FLEET_SIM_URL") or os.getenv(
        "TARGET_FLEET_SIM_URL"
    )
    if external_url:
        env_vars["TARGET_FLEET_SIM_URL"] = external_url
        log.info(f"Using external vllm-sr-sim service: {external_url}")
        return False

    raw_enabled = env_vars.get(
        "VLLM_SR_SIM_ENABLED", os.getenv("VLLM_SR_SIM_ENABLED", "true")
    )
    if str(raw_enabled).lower() == "false":
        log.info("vllm-sr-sim sidecar disabled via VLLM_SR_SIM_ENABLED=false")
        return False

    log.info("Starting vllm-sr-sim sidecar...")
    _start_named_service(
        "vllm-sr-sim",
        lambda: docker_start_fleet_sim(
            network_name=stack_layout.network_name,
            config_dir=config_dir,
            stack_layout=stack_layout,
            pull_policy=pull_policy,
        ),
    )
    env_vars["TARGET_FLEET_SIM_URL"] = stack_layout.fleet_sim_service_url
    return True


def connect_runtime_container(
    shared_network_name: str, stack_layout: RuntimeStackLayout
) -> None:
    """Attach the runtime container to the shared OpenClaw bridge network."""
    connected = []
    for container_name in stack_layout.runtime_container_names:
        if docker_container_status(container_name) == "not found":
            continue

        return_code, _stdout, stderr = docker_network_connect(
            shared_network_name, container_name
        )
        if return_code != 0:
            log.error(
                f"Failed to connect {container_name} to {shared_network_name}: {stderr}"
            )
            for started_container in reversed(connected):
                docker_stop_container(started_container)
                docker_remove_container(started_container)
            docker_stop_container(container_name)
            docker_remove_container(container_name)
            raise SystemExit(1)
        connected.append(container_name)
        log.info(f"Connected {container_name} to {shared_network_name}")


def maybe_finish_setup_mode(
    setup_mode: bool,
    dashboard_disabled: bool,
    stack_layout: RuntimeStackLayout,
) -> bool:
    """Wait for dashboard-only setup mode and print next-step guidance."""
    if not setup_mode:
        return False
    if dashboard_disabled:
        log.error("Setup mode started without dashboard enabled")
        raise SystemExit(1)

    log.info("Setup mode detected: skipping Router and Envoy health checks")
    log.info("Waiting for Dashboard to become healthy...")
    dashboard_container = _runtime_service_container_name(stack_layout, "dashboard")
    _wait_for_setup_dashboard(dashboard_container)
    ensure_runtime_container_not_exited(dashboard_container, phase="during setup mode")

    log.info("=" * 60)
    log.info("vLLM Semantic Router setup mode is running!")
    log.info("")
    log.info("Next steps:")
    log.info(f"  - Open {stack_layout.dashboard_url}")
    log.info("  - Configure your first model in the dashboard")
    log.info("  - Activate a runnable config to enable routing")
    _log_runtime_commands(dashboard_disabled=False, fleet_sim_enabled=False)
    log.info("=" * 60)
    return True


def wait_for_router_health(stack_layout: RuntimeStackLayout) -> None:
    """Block until the router health endpoint responds or the timeout elapses."""
    log.info("Waiting for Router to become healthy...")
    log.info(f"Health check timeout: {HEALTH_CHECK_TIMEOUT}s")
    log.info("Showing Router logs during startup:")
    log.info("-" * 60)

    router_container = _runtime_service_container_name(stack_layout, "router")
    start_time = time.time()
    last_log_time = start_time
    check_count = 0

    while time.time() - start_time < HEALTH_CHECK_TIMEOUT:
        check_count += 1
        _emit_router_startup_logs(router_container, int(last_log_time))
        last_log_time = time.time()

        return_code, _stdout, _stderr = docker_exec(
            router_container,
            ["curl", "-f", "-s", f"http://localhost:{DEFAULT_API_PORT}/health"],
        )
        if return_code == 0:
            elapsed = int(time.time() - start_time)
            log.info("-" * 60)
            log.info(f"Router is healthy (after {elapsed}s, {check_count} checks)")
            return

        if check_count % 10 == 0:
            elapsed = int(time.time() - start_time)
            remaining = int(HEALTH_CHECK_TIMEOUT - elapsed)
            log.info(
                f"  ... still waiting ({elapsed}s elapsed, {remaining}s remaining)"
            )

        time.sleep(HEALTH_CHECK_INTERVAL)

    log.info("-" * 60)
    log.error(f"Router failed to become healthy after {HEALTH_CHECK_TIMEOUT}s")
    log.info("Showing full container logs:")
    docker_logs(router_container, follow=False, tail=100)
    raise SystemExit(1)


def ensure_runtime_container_not_exited(
    container_name: str, phase: str | None = None
) -> None:
    """Abort if the runtime container exited unexpectedly."""
    status = docker_container_status(container_name)
    if status != "exited":
        return

    suffix = f" {phase}" if phase else ""
    log.error(f"Container exited unexpectedly{suffix}")
    log.info("Showing container logs:")
    docker_logs(container_name, follow=False)
    raise SystemExit(1)


def _runtime_service_container_name(
    stack_layout: RuntimeStackLayout, service: str
) -> str:
    preferred = stack_layout.service_container_name(service)
    if docker_container_status(preferred) != "not found":
        return preferred
    return stack_layout.container_name


def recover_openclaw_containers(
    config_dir: str, env_vars: dict[str, str], shared_network_name: str
) -> None:
    """Reconnect and restart previously stopped OpenClaw containers."""
    openclaw_data_dir = resolve_openclaw_data_dir(config_dir, env_vars)
    openclaw_entries = load_openclaw_registry(openclaw_data_dir)
    if not openclaw_entries:
        return

    log.info(f"Recovering {len(openclaw_entries)} OpenClaw container(s)...")
    for entry in openclaw_entries:
        name = entry.get("name") or entry.get("containerName")
        if not name:
            continue
        status = docker_container_status(name)
        if status == "not found":
            log.warning(f"OpenClaw container {name} no longer exists, skipping")
            continue

        return_code, _stdout, _stderr = docker_network_connect(
            shared_network_name, name
        )
        if return_code == 0:
            log.info(f"Connected {name} to {shared_network_name}")
        else:
            log.warning(f"Failed to connect {name} to {shared_network_name}")

        if status != "running":
            log.info(f"Starting OpenClaw container: {name}")
            docker_start_container(name)


def resolve_openclaw_data_dir(
    config_dir: str, env_vars: dict[str, str] | None = None
) -> str:
    """Resolve the persisted OpenClaw data directory for the current workspace."""
    env_vars = env_vars or {}
    default_path = os.path.join(config_dir, ".vllm-sr", "openclaw-data")
    openclaw_data_dir = (
        env_vars.get("OPENCLAW_DATA_DIR")
        or os.getenv("OPENCLAW_DATA_DIR")
        or default_path
    )
    return os.path.abspath(openclaw_data_dir)


def log_runtime_summary(
    listeners,
    stack_layout: RuntimeStackLayout,
    dashboard_disabled: bool,
    enable_observability: bool,
    fleet_sim_enabled: bool,
    started_backends: set[str] | None = None,
) -> None:
    """Print the local endpoints and common follow-up commands."""
    log.info("=" * 60)
    log.info("vLLM Semantic Router is running!")
    log.info("")
    log.info("Endpoints:")
    if not dashboard_disabled:
        log.info(f"  - Dashboard: {stack_layout.dashboard_url}")
    for listener in listeners:
        name = listener.get("name", "unknown")
        port = listener.get("port", "unknown")
        if isinstance(port, int):
            port += stack_layout.port_offset
        log.info(f"  - {name}: http://localhost:{port}")
    log.info(f"  - Metrics: {stack_layout.metrics_url}")
    if fleet_sim_enabled:
        log.info(f"  - Fleet Sim: {stack_layout.fleet_sim_url}")

    if started_backends:
        log.info("")
        log.info("Storage:")
        if "redis" in started_backends:
            log.info(f"  - Redis: {stack_layout.redis_url}")
        if "postgres" in started_backends:
            log.info(f"  - Postgres: {stack_layout.postgres_url}")

    if enable_observability:
        log.info("")
        log.info("Observability:")
        log.info(f"  - Jaeger UI: {stack_layout.jaeger_ui_url}")
        log.info(f"  - Grafana: {stack_layout.grafana_url} (admin/admin)")
        log.info(f"  - Prometheus: {stack_layout.prometheus_url}")

    _log_runtime_commands(dashboard_disabled, fleet_sim_enabled)
    _print_curl_example(listeners, stack_layout)


def _start_named_service(service_name: str, starter: ServiceStarter) -> None:
    return_code, _stdout, stderr = starter()
    if return_code != 0:
        log.error(f"Failed to start {service_name}: {stderr}")
        raise SystemExit(1)
    log.info(f"{service_name} started successfully")


def _wait_for_setup_dashboard(container_name: str) -> None:
    start_time = time.time()
    while time.time() - start_time < HEALTH_CHECK_TIMEOUT:
        return_code, _stdout, _stderr = docker_exec(
            container_name,
            ["curl", "-f", "-s", "http://localhost:8700/healthz"],
        )
        if return_code == 0:
            return
        time.sleep(HEALTH_CHECK_INTERVAL)

    log.error("Dashboard failed to become healthy in setup mode")
    docker_logs(container_name, follow=False, tail=100)
    raise SystemExit(1)


def _emit_router_startup_logs(container_name: str, since_timestamp: int) -> None:
    return_code, stdout, stderr = docker_logs_since(container_name, since_timestamp)
    if return_code != 0:
        return
    _print_matching_lines(stdout)
    _print_matching_lines(stderr)


def _print_matching_lines(text: str) -> None:
    if not text:
        return
    for line in text.strip().split("\n"):
        if line.strip() and "caller" in line.lower():
            print(f"  {line}")


def _log_runtime_commands(dashboard_disabled: bool, fleet_sim_enabled: bool) -> None:
    log.info("")
    log.info("Commands:")
    if not dashboard_disabled:
        log.info("  - vllm-sr dashboard              Open dashboard in browser")
    if fleet_sim_enabled:
        log.info("  - vllm-sr logs <envoy|router|dashboard|simulator> [-f]")
        log.info("  - vllm-sr status [envoy|router|dashboard|simulator|all]")
    else:
        log.info("  - vllm-sr logs <envoy|router|dashboard> [-f]")
        log.info("  - vllm-sr status [envoy|router|dashboard|all]")
    log.info("  - vllm-sr stop")


def _print_curl_example(listeners, stack_layout: RuntimeStackLayout) -> None:
    if not listeners:
        return
    first_port = listeners[0].get("port", DEFAULT_LISTENER_PORT)
    if isinstance(first_port, int):
        first_port += stack_layout.port_offset

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
