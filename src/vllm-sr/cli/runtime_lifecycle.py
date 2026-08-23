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
from cli.container_cli import (
    container_create_network,
    container_exec,
    container_logs,
    container_logs_since,
    container_network_connect,
    container_remove_container,
    container_start_container,
    container_start_fleet_sim,
    container_start_grafana,
    container_start_jaeger,
    container_start_prometheus,
    container_status,
    container_stop_container,
    load_openclaw_registry,
)
from cli.runtime_stack import RuntimeStackLayout
from cli.terminal import echo, fields, heading, progress, success
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
    status = container_status(container_name)
    if status == "not found":
        return
    log.info(f"Existing container found (status: {status}), cleaning up...")
    if status in {"running", "paused"}:
        container_stop_container(container_name)
    container_remove_container(container_name)


def ensure_shared_network(shared_network_name: str) -> None:
    """Create the shared OpenClaw bridge network used by local stacks."""
    return_code, _stdout, stderr = container_create_network(shared_network_name)
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
        lambda: container_start_jaeger(shared_network_name, stack_layout=stack_layout),
    )
    _start_named_service(
        "Prometheus",
        lambda: container_start_prometheus(
            shared_network_name, config_dir, stack_layout=stack_layout
        ),
    )
    _start_named_service(
        "Grafana",
        lambda: container_start_grafana(
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
    sim_image: str | None = None,
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
        lambda: container_start_fleet_sim(
            image=sim_image,
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
        if container_status(container_name) == "not found":
            continue

        return_code, _stdout, stderr = container_network_connect(
            shared_network_name, container_name
        )
        if return_code != 0:
            log.error(
                f"Failed to connect {container_name} to {shared_network_name}: {stderr}"
            )
            for started_container in reversed(connected):
                container_stop_container(started_container)
                container_remove_container(started_container)
            container_stop_container(container_name)
            container_remove_container(container_name)
            raise SystemExit(1)
        connected.append(container_name)
        log.info(f"Connected {container_name} to {shared_network_name}")


def wait_for_router_health(
    stack_layout: RuntimeStackLayout,
    management_port: int = DEFAULT_API_PORT,
    readiness_token_env: str | None = None,
    management_tls_certificate: str | None = None,
) -> None:
    """Block until the router readiness endpoint responds or the timeout elapses."""
    log.info("Waiting for Router to become ready...")
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

        status = container_status(router_container)
        if status != "running":
            log.error(
                f"Router container is not running during readiness wait: {status}"
            )
            log.info("Showing Router container logs:")
            container_logs(router_container, follow=False, tail=120)
            raise SystemExit(1)

        return_code, _stdout, _stderr = container_exec(
            router_container,
            _router_readiness_command(
                management_port, readiness_token_env, management_tls_certificate
            ),
        )
        if return_code == 0:
            elapsed = int(time.time() - start_time)
            log.info("-" * 60)
            log.info(f"Router is ready (after {elapsed}s, {check_count} checks)")
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
    container_logs(router_container, follow=False, tail=100)
    raise SystemExit(1)


def _router_readiness_command(
    management_port: int,
    readiness_token_env: str | None,
    management_tls_certificate: str | None = None,
) -> list[str]:
    scheme = "https" if management_tls_certificate is not None else "http"
    endpoint = f"{scheme}://localhost:{management_port}/ready"
    tls_arguments = (
        ["--cacert", management_tls_certificate] if management_tls_certificate else []
    )
    if readiness_token_env is None:
        return ["curl", "-f", "-s", *tls_arguments, endpoint]
    script = (
        'set -eu; token="$(printenv "$1")"; test -n "$token"; shift; '
        "printf 'Authorization: Bearer %s\\n' \"$token\" | "
        'curl -f -s -H @- "$@"'
    )
    return [
        "sh",
        "-c",
        script,
        "vllm-sr-readiness",
        readiness_token_env,
        *tls_arguments,
        endpoint,
    ]


def wait_and_verify_runtime(
    stack_layout: RuntimeStackLayout,
    dashboard_disabled: bool,
    management_port: int = DEFAULT_API_PORT,
    readiness_token_env: str | None = None,
    management_tls_certificate: str | None = None,
) -> None:
    """Wait for readiness and verify every required runtime container."""
    wait_for_router_health(
        stack_layout,
        management_port=management_port,
        readiness_token_env=readiness_token_env,
        management_tls_certificate=management_tls_certificate,
    )
    for service in ("router", "envoy"):
        ensure_runtime_container_not_exited(
            stack_layout.service_container_name(service)
        )
    if not dashboard_disabled:
        ensure_runtime_container_not_exited(stack_layout.dashboard_container_name)


def ensure_runtime_container_not_exited(
    container_name: str, phase: str | None = None
) -> None:
    """Abort if the runtime container exited unexpectedly."""
    status = container_status(container_name)
    if status != "exited":
        return

    suffix = f" {phase}" if phase else ""
    log.error(f"Container exited unexpectedly{suffix}")
    log.info("Showing container logs:")
    container_logs(container_name, follow=False)
    raise SystemExit(1)


def _runtime_service_container_name(
    stack_layout: RuntimeStackLayout, service: str
) -> str:
    return stack_layout.service_container_name(service)


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
        status = container_status(name)
        if status == "not found":
            log.warning(f"OpenClaw container {name} no longer exists, skipping")
            continue

        return_code, _stdout, _stderr = container_network_connect(
            shared_network_name, name
        )
        if return_code == 0:
            log.info(f"Connected {name} to {shared_network_name}")
        else:
            log.warning(f"Failed to connect {name} to {shared_network_name}")

        if status != "running":
            log.info(f"Starting OpenClaw container: {name}")
            container_start_container(name)


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
    success("vLLM Semantic Router is running")
    echo()
    heading("Endpoints")
    endpoints = []
    if not dashboard_disabled:
        endpoints.append(("Dashboard", stack_layout.dashboard_url))
    for listener in listeners:
        name = listener.get("name", "unknown")
        port = listener.get("port", "unknown")
        if isinstance(port, int):
            port += stack_layout.port_offset
        endpoints.append((name, f"http://localhost:{port}"))
    endpoints.append(("Metrics", stack_layout.metrics_url))
    if fleet_sim_enabled:
        endpoints.append(("Fleet Sim", stack_layout.fleet_sim_url))
    fields(endpoints)

    if started_backends:
        storage = []
        if "redis" in started_backends:
            storage.append(("Redis", stack_layout.redis_url))
        if "postgres" in started_backends:
            storage.append(("Postgres", stack_layout.postgres_url))
        if storage:
            echo()
            heading("Storage")
            fields(storage)

    if enable_observability:
        echo()
        heading("Observability")
        fields(
            (
                ("Jaeger UI", stack_layout.jaeger_ui_url),
                ("Grafana", f"{stack_layout.grafana_url} (admin/admin)"),
                ("Prometheus", stack_layout.prometheus_url),
            )
        )

    _log_runtime_commands(dashboard_disabled, fleet_sim_enabled)
    _print_curl_example(listeners, stack_layout)


def _start_named_service(service_name: str, starter: ServiceStarter) -> None:
    return_code, _stdout, stderr = starter()
    if return_code != 0:
        log.error(f"Failed to start {service_name}: {stderr}")
        raise SystemExit(1)
    log.info(f"{service_name} started successfully")


def _emit_router_startup_logs(container_name: str, since_timestamp: int) -> None:
    return_code, stdout, stderr = container_logs_since(container_name, since_timestamp)
    if return_code != 0:
        return
    _print_matching_lines(stdout)
    _print_matching_lines(stderr)


def _print_matching_lines(text: str) -> None:
    if not text:
        return
    for line in text.strip().split("\n"):
        if line.strip() and "caller" in line.lower():
            progress(f"  {line}")


def _log_runtime_commands(dashboard_disabled: bool, fleet_sim_enabled: bool) -> None:
    commands = []
    if not dashboard_disabled:
        commands.append(("Dashboard", "vllm-sr dashboard"))
    if fleet_sim_enabled:
        commands.extend(
            (
                ("Logs", "vllm-sr logs <envoy|router|dashboard|simulator> [-f]"),
                (
                    "Status",
                    "vllm-sr status [envoy|router|dashboard|simulator|all]",
                ),
            )
        )
    else:
        commands.extend(
            (
                ("Logs", "vllm-sr logs <envoy|router|dashboard> [-f]"),
                ("Status", "vllm-sr status [envoy|router|dashboard|all]"),
            )
        )
    commands.append(("Stop", "vllm-sr stop"))
    echo()
    heading("Commands")
    fields(commands)


def _print_curl_example(listeners, stack_layout: RuntimeStackLayout) -> None:
    if not listeners:
        return
    first_port = listeners[0].get("port", DEFAULT_LISTENER_PORT)
    if isinstance(first_port, int):
        first_port += stack_layout.port_offset

    echo()
    heading("Try it")
    echo(f"  curl -v http://localhost:{first_port}/v1/chat/completions \\")
    echo('    -H "Content-Type: application/json" \\')
    echo("    -d '{")
    echo('      "model": "vllm-sr/auto",')
    echo('      "messages": [')
    echo('        {"role": "user", "content": "What is the derivative of x^2?"}')
    echo("      ]")
    echo("    }'")
