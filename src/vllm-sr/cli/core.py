"""Core management functions for vLLM Semantic Router."""

import os
import subprocess

from cli.consts import DEFAULT_API_PORT, DEFAULT_ENVOY_PORT
from cli.docker_cli import (
    docker_container_status,
    docker_exec,
    docker_network_disconnect,
    docker_remove_container,
    docker_remove_network,
    docker_start_vllm_sr,
    docker_stop_container,
    load_openclaw_registry,
)
from cli.logo import print_vllm_logo
from cli.runtime_lifecycle import (
    connect_runtime_container,
    ensure_clean_runtime_container,
    ensure_runtime_container_not_exited,
    ensure_shared_network,
    log_runtime_summary,
    log_startup_banner,
    maybe_finish_setup_mode,
    recover_openclaw_containers,
    resolve_openclaw_data_dir,
    start_observability_stack,
    wait_for_router_health,
)
from cli.runtime_stack import resolve_runtime_stack
from cli.utils import getLogger, load_config

log = getLogger(__name__)

SERVICE_LOG_PATTERNS = {
    "router": r'"caller"|spawned: \'router\'|success: router|cli\.commands',
    "dashboard": r"dashboard|Dashboard|spawned: 'dashboard'|success: dashboard|:8700",
    "envoy": (
        r"\[2[0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].*\]\[.*\]"
        r"|spawned: 'envoy'|success: envoy"
    ),
}


def start_vllm_sr(
    config_file, env_vars=None, image=None, pull_policy=None, enable_observability=True
):
    """Start vLLM Semantic Router."""
    if env_vars is None:
        env_vars = {}
    stack_layout = resolve_runtime_stack()

    print_vllm_logo()
    listeners = load_config(config_file).get("listeners", [])
    if not listeners:
        log.error("No listeners configured in config.yaml")
        raise SystemExit(1)

    log_startup_banner(config_file, listeners, stack_layout)
    ensure_clean_runtime_container(stack_layout.container_name)

    shared_network_name = stack_layout.network_name
    config_dir = os.path.dirname(os.path.abspath(config_file))
    ensure_shared_network(shared_network_name)
    network_name = start_observability_stack(
        enable_observability,
        shared_network_name,
        config_dir,
        env_vars,
        stack_layout,
    )

    dashboard_disabled = env_vars.get("DISABLE_DASHBOARD") == "true"
    setup_mode = str(env_vars.get("VLLM_SR_SETUP_MODE", "")).lower() == "true"
    return_code, _stdout, stderr = docker_start_vllm_sr(
        config_file,
        env_vars,
        listeners,
        image=image,
        pull_policy=pull_policy,
        network_name=network_name,
        openclaw_network_name=shared_network_name,
        minimal=dashboard_disabled,
        stack_layout=stack_layout,
    )
    if return_code != 0:
        log.error(f"Failed to start container: {stderr}")
        raise SystemExit(1)

    log.info("vLLM Semantic Router container started successfully")
    connect_runtime_container(shared_network_name, stack_layout)
    if maybe_finish_setup_mode(setup_mode, dashboard_disabled, stack_layout):
        return

    wait_for_router_health(stack_layout)
    ensure_runtime_container_not_exited(stack_layout.container_name)
    recover_openclaw_containers(config_dir, env_vars, shared_network_name)
    log_runtime_summary(
        listeners, stack_layout, dashboard_disabled, enable_observability
    )


def stop_vllm_sr():
    """Stop vLLM Semantic Router and observability containers."""
    log.info("Stopping vLLM Semantic Router...")
    stack_layout = resolve_runtime_stack()

    status = docker_container_status(stack_layout.container_name)
    if status == "not found":
        log.info("Container not found. Nothing to stop.")
        return

    openclaw_data_dir = resolve_openclaw_data_dir(os.getcwd())
    network_name = stack_layout.network_name

    openclaw_entries = load_openclaw_registry(openclaw_data_dir)
    for entry in openclaw_entries:
        name = entry.get("name") or entry.get("containerName")
        if not name:
            continue
        openclaw_status = docker_container_status(name)
        if openclaw_status == "not found":
            continue
        if openclaw_status == "running":
            log.info(f"Stopping OpenClaw container: {name}")
            docker_stop_container(name)
        log.info(f"Disconnecting {name} from {network_name}")
        docker_network_disconnect(network_name, name)

    if status == "running":
        docker_stop_container(stack_layout.container_name)

    docker_remove_container(stack_layout.container_name)
    log.info("vLLM Semantic Router stopped")

    observability_containers = [
        stack_layout.grafana_container_name,
        stack_layout.prometheus_container_name,
        stack_layout.jaeger_container_name,
    ]
    for container_name in observability_containers:
        service_status = docker_container_status(container_name)
        if service_status == "not found":
            continue
        log.info(f"Stopping {container_name}...")
        if service_status == "running":
            docker_stop_container(container_name)
        docker_remove_container(container_name)
        log.info(f"{container_name} stopped")

    return_code, _stdout, _stderr = docker_remove_network(network_name)
    if return_code == 0:
        log.info(f"Network {network_name} removed")


def show_logs(service: str, follow: bool = False):
    """Show logs from a runtime service."""
    _validate_runtime_service(service)
    stack_layout = resolve_runtime_stack()
    _ensure_runtime_container_available(stack_layout.container_name)

    if follow:
        log.info(f"Following {service} logs (Ctrl+C to stop)...")
        log.info("")

    grep_pattern = SERVICE_LOG_PATTERNS[service]
    command = _log_command(stack_layout.container_name, grep_pattern, follow)
    try:
        if follow:
            subprocess.run(command, shell=True, check=False)
            return

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout:
            print(result.stdout)
        else:
            log.info(f"No recent {service} logs found")
    except KeyboardInterrupt:
        log.info("\nStopped following logs")
    except Exception as exc:
        log.error(f"Failed to get {service} logs: {exc}")
        raise SystemExit(1) from exc


def show_status(service: str = "all"):
    """Show runtime service status."""
    stack_layout = resolve_runtime_stack()
    status = docker_container_status(stack_layout.container_name)
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

    for requested_service in _requested_services(service):
        _report_service_status(requested_service, stack_layout)

    log.info("")
    log.info("For detailed logs: vllm-sr logs <envoy|router|dashboard>")
    log.info("=" * 60)


def _validate_runtime_service(service: str) -> None:
    if service in SERVICE_LOG_PATTERNS:
        return
    log.error(f"Invalid service: {service}")
    log.error("Must be 'envoy', 'router', or 'dashboard'")
    raise SystemExit(1)


def _requested_services(service: str) -> list[str]:
    if service == "all":
        return ["router", "envoy", "dashboard"]
    _validate_runtime_service(service)
    return [service]


def _ensure_runtime_container_available(container_name: str) -> None:
    if docker_container_status(container_name) != "not found":
        return
    log.error("Container not found. Is vLLM Semantic Router running?")
    log.info("Start it with: vllm-sr serve")
    raise SystemExit(1)


def _log_command(container_name: str, grep_pattern: str, follow: bool) -> str:
    if follow:
        return f'docker logs -f {container_name} 2>&1 | grep -E "{grep_pattern}"'
    return (
        f'docker logs --tail 200 {container_name} 2>&1 | grep -E "{grep_pattern}" '
        "| tail -50"
    )


def _report_service_status(service: str, stack_layout) -> None:
    checkers = {
        "router": ("Router", _check_router_status, None),
        "envoy": ("Envoy", _check_envoy_status, None),
        "dashboard": (
            "Dashboard",
            _check_dashboard_status,
            stack_layout.dashboard_url,
        ),
    }
    label, checker, detail = checkers[service]
    try:
        is_running = checker(stack_layout.container_name)
        _log_service_status(label, is_running, detail if is_running else None)
    except Exception as exc:
        log.error(f"Failed to check {service} status: {exc}")


def _check_router_status(container_name: str) -> bool:
    return_code, _stdout, _stderr = docker_exec(
        container_name,
        ["curl", "-f", "-s", f"http://localhost:{DEFAULT_API_PORT}/health"],
    )
    return return_code == 0


def _check_envoy_status(container_name: str) -> bool:
    return_code, stdout, _stderr = docker_exec(
        container_name,
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
    )
    return return_code == 0 and stdout.strip() == "200"


def _check_dashboard_status(container_name: str) -> bool:
    return_code, stdout, _stderr = docker_exec(
        container_name,
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
    )
    return return_code == 0 and stdout.strip() in {"200", "301", "302"}


def _log_service_status(
    label: str, is_running: bool, detail: str | None = None
) -> None:
    if not is_running:
        log.info(f"WARNING {label}: Status unknown")
        return
    if detail:
        log.info(f"{label}: Running ({detail})")
        return
    log.info(f"{label}: Running")
