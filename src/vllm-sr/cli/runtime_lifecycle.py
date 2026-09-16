"""Startup and readiness helpers for vLLM Semantic Router runtime."""

from __future__ import annotations

import json
import math
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
    container_start_grafana,
    container_start_jaeger,
    container_start_prometheus,
    container_status,
    container_status_strict,
    container_stop_container,
    load_openclaw_registry,
)
from cli.container_runtime import get_container_runtime
from cli.runtime_lifecycle_lock import acquire_runtime_lifecycle_lock
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
    if status in {"running", "paused"} and not container_stop_container(container_name):
        raise RuntimeError(f"Failed to stop runtime container: {container_name}")
    container_remove_container(container_name)


def stop_runtime_before_config_replacement(stack_layout: RuntimeStackLayout) -> None:
    """Stop old config consumers before a restart publishes their replacement.

    Keep the stopped containers for normal deployment cleanup. A failed stop or
    uncertain state must leave the old active document and provenance intact.
    """
    with acquire_runtime_lifecycle_lock(
        runtime=get_container_runtime(), stack_name=stack_layout.stack_name
    ):
        names = stack_layout.runtime_container_names
        states = {name: container_status_strict(name) for name in names}
        for name, state in states.items():
            if state in {"running", "paused", "restarting"}:
                if not container_stop_container(name):
                    raise RuntimeError(f"Failed to stop runtime container: {name}")
            elif state not in {"not found", "exited", "created", "dead"}:
                raise RuntimeError(f"Runtime container is not stopped: {name}")
        for name in names:
            if container_status_strict(name) not in {
                "not found",
                "exited",
                "created",
                "dead",
            }:
                raise RuntimeError(f"Runtime container is not stopped: {name}")


def ensure_shared_network(shared_network_name: str) -> None:
    """Create the shared OpenClaw bridge network used by local stacks."""
    _ensure_network(shared_network_name, "shared OpenClaw")


def ensure_data_network(data_network_name: str) -> None:
    """Create the bridge network reserved for this stack's storage services.

    It exists so that joining the application network is not enough to reach
    Redis, Postgres, or Milvus. Only those three and Router are attached to it.
    """
    _ensure_network(data_network_name, "storage data")


def _ensure_network(network_name: str, description: str) -> None:
    return_code, _stdout, stderr = container_create_network(network_name)
    if return_code != 0:
        log.error(f"Failed to create {description} network: {stderr}")
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


def maybe_finish_setup_mode(
    setup_mode: bool,
    dashboard_disabled: bool,
    stack_layout: RuntimeStackLayout,
    startup_timeout: int = HEALTH_CHECK_TIMEOUT,
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
    _wait_for_setup_dashboard(dashboard_container, startup_timeout)
    ensure_runtime_container_not_exited(
        dashboard_container, phase="during setup mode", timeout=5
    )

    success("vLLM Semantic Router setup mode is running")
    echo()
    heading("Next steps")
    fields(
        (
            ("Dashboard", stack_layout.dashboard_url),
            ("Configure", "Add your first model in the dashboard"),
            ("Activate", "Activate a runnable config to enable routing"),
        )
    )
    _log_runtime_commands(dashboard_disabled=False)
    return True


def validate_startup_timeout(seconds: int) -> None:
    """Reject invalid host-side budgets before changing runtime state."""
    try:
        finite = math.isfinite(seconds)
    except (TypeError, OverflowError):
        finite = False
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, int)
        or seconds <= 0
        or not finite
    ):
        raise ValueError(
            "startup timeout must be a finite positive integer number of seconds"
        )


class _StartupDeadline:
    def __init__(self, seconds: int):
        validate_startup_timeout(seconds)
        self.seconds = seconds
        self.started = time.monotonic()

    def remaining(self) -> float:
        return max(0.0, self.seconds - (time.monotonic() - self.started))

    def io_timeout(self) -> float:
        remaining = self.remaining()
        if remaining <= 0:
            raise TimeoutError("startup readiness deadline expired")
        return min(5.0, remaining)


def wait_for_router_health(
    stack_layout: RuntimeStackLayout,
    management_port: int = DEFAULT_API_PORT,
    readiness_token_env: str | None = None,
    startup_timeout: int = HEALTH_CHECK_TIMEOUT,
) -> None:
    """Block until the router readiness endpoint responds or the timeout elapses."""
    _wait_for_readiness(
        _runtime_service_container_name(stack_layout, "router"),
        "Router",
        startup_timeout,
        lambda timeout: _router_readiness_command(
            management_port, readiness_token_env, timeout
        ),
        show_router_logs=True,
    )


def _wait_for_readiness(
    container_name: str,
    service: str,
    startup_timeout: int,
    command: Callable[[float], list[str]],
    *,
    show_router_logs: bool = False,
) -> None:
    deadline = _StartupDeadline(startup_timeout)
    log.info(f"Waiting for {service} to become ready...")
    log.info(f"Startup readiness timeout: {startup_timeout}s")
    last_log_time = time.time()
    check_count = 0
    if show_router_logs:
        log.info("Showing Router logs during startup:")
        log.info("-" * 60)

    while deadline.remaining() > 0:
        try:
            check_count += 1
            if show_router_logs:
                _emit_router_startup_logs(
                    container_name, int(last_log_time), timeout=deadline.io_timeout()
                )
                last_log_time = time.time()
            try:
                status = container_status_strict(
                    container_name, timeout=deadline.io_timeout()
                )
            except RuntimeError:
                # A busy or unavailable daemon is not proof that the container
                # stopped. Retry inspection within the same startup budget.
                status = None
                log.debug(f"{service} container inspection unavailable; retrying")
            if status is not None and status != "running":
                log.error(
                    f"{service} container is not running during readiness wait: {status}"
                )
                container_logs(container_name, follow=False, tail=120, timeout=5)
                raise SystemExit(1)
            return_code = 1
            if status == "running":
                probe_timeout = deadline.io_timeout()
                return_code, _stdout, _stderr = container_exec(
                    container_name, command(probe_timeout), timeout=probe_timeout
                )
            remaining = deadline.remaining()
            if return_code == 0 and remaining > 0:
                elapsed = int(time.monotonic() - deadline.started)
                log.info(f"{service} is ready (after {elapsed}s, {check_count} checks)")
                return
            if check_count % 10 == 0:
                elapsed = int(time.monotonic() - deadline.started)
                log.info(
                    f"  ... still waiting ({elapsed}s elapsed, {int(remaining)}s remaining)"
                )
            if remaining > 0:
                time.sleep(min(HEALTH_CHECK_INTERVAL, remaining))
        except TimeoutError:
            break

    log.error(f"{service} did not become ready within {startup_timeout}s")
    log.info(
        "Startup wait ended without stopping containers; inspect vllm-sr status and vllm-sr logs."
    )
    # Diagnostics have their own small bound after the readiness budget expires.
    container_logs(container_name, follow=False, tail=100, timeout=5)
    raise SystemExit(1)


def _router_readiness_command(
    management_port: int, readiness_token_env: str | None, timeout: float = 5.0
) -> list[str]:
    endpoint = f"http://localhost:{management_port}/ready"
    max_time = f"{max(0.001, timeout):.3f}"
    if readiness_token_env is None:
        return ["curl", "-f", "-s", "--max-time", max_time, endpoint]
    script = (
        'set -eu; token="$(printenv "$1")"; test -n "$token"; '
        "printf 'Authorization: Bearer %s\\n' \"$token\" | "
        'curl -f -s --max-time "$3" -H @- "$2"'
    )
    return [
        "sh",
        "-c",
        script,
        "vllm-sr-readiness",
        readiness_token_env,
        endpoint,
        max_time,
    ]


def wait_and_verify_runtime(
    stack_layout: RuntimeStackLayout,
    dashboard_disabled: bool,
    management_port: int = DEFAULT_API_PORT,
    readiness_token_env: str | None = None,
    startup_timeout: int = HEALTH_CHECK_TIMEOUT,
) -> None:
    """Wait for readiness and verify every required runtime container."""
    wait_for_router_health(
        stack_layout,
        management_port=management_port,
        readiness_token_env=readiness_token_env,
        startup_timeout=startup_timeout,
    )
    for service in ("router", "envoy"):
        ensure_runtime_container_not_exited(
            stack_layout.service_container_name(service), timeout=5
        )
    if not dashboard_disabled:
        ensure_runtime_container_not_exited(
            stack_layout.dashboard_container_name, timeout=5
        )


def ensure_runtime_container_not_exited(
    container_name: str, phase: str | None = None, *, timeout: float | None = None
) -> None:
    """Abort if the runtime container exited unexpectedly."""
    try:
        status = container_status_strict(
            container_name, **({"timeout": timeout} if timeout is not None else {})
        )
    except RuntimeError:
        status = "inspection unavailable"
    if status == "running":
        return

    suffix = f" {phase}" if phase else ""
    log.error(f"Container is not confirmed running{suffix}: {status}")
    log.info("Showing container logs:")
    container_logs(container_name, follow=False, timeout=timeout)
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
    started_backends: set[str] | None = None,
    config: dict | None = None,
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

    _log_runtime_commands(dashboard_disabled)
    _print_curl_example(listeners, stack_layout, config)


def _start_named_service(service_name: str, starter: ServiceStarter) -> None:
    return_code, _stdout, stderr = starter()
    if return_code != 0:
        log.error(f"Failed to start {service_name}: {stderr}")
        raise SystemExit(1)
    log.info(f"{service_name} started successfully")


def _wait_for_setup_dashboard(
    container_name: str, startup_timeout: int = HEALTH_CHECK_TIMEOUT
) -> None:
    _wait_for_readiness(
        container_name,
        "Dashboard",
        startup_timeout,
        lambda timeout: [
            "curl",
            "-f",
            "-s",
            "--max-time",
            f"{max(0.001, timeout):.3f}",
            "http://localhost:8700/healthz",
        ],
    )


def _emit_router_startup_logs(
    container_name: str, since_timestamp: int, *, timeout: float | None = None
) -> None:
    return_code, stdout, stderr = container_logs_since(
        container_name, since_timestamp, timeout=timeout
    )
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


def _log_runtime_commands(dashboard_disabled: bool) -> None:
    commands = []
    if not dashboard_disabled:
        commands.append(("Dashboard", "vllm-sr dashboard"))
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


def _example_model(config: dict | None) -> str:
    for entrypoint in (config or {}).get("entrypoints") or []:
        names = entrypoint.get("model_names") or []
        if names:
            return names[0]
    return "vllm-sr/auto"


def _print_curl_example(
    listeners, stack_layout: RuntimeStackLayout, config: dict | None = None
) -> None:
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
    model_json = json.dumps(_example_model(config)).replace("'", "'\"'\"'")
    echo(f'      "model": {model_json},')
    echo('      "messages": [')
    echo('        {"role": "user", "content": "What is the derivative of x^2?"}')
    echo("      ]")
    echo("    }'")
