"""Core management functions for vLLM Semantic Router."""

import os
import subprocess

from cli.consts import DEFAULT_API_PORT, DEFAULT_ENVOY_PORT
from cli.docker_cli import (
    docker_container_status,
    docker_exec,
    docker_logs,
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
    start_fleet_sim_sidecar,
    start_observability_stack,
    wait_for_router_health,
)
from cli.runtime_stack import RuntimeStackLayout, resolve_runtime_stack
from cli.utils import get_logger, load_config

log = get_logger(__name__)

SERVICE_LOG_PATTERNS = {
    "router": r'"caller"|spawned: \'router\'|success: router|cli\.commands',
    "dashboard": r"dashboard|Dashboard|spawned: 'dashboard'|success: dashboard|:8700",
    "envoy": (
        r"\[2[0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].*\]\[.*\]"
        r"|spawned: 'envoy'|success: envoy"
    ),
}


def start_vllm_sr(
    config_file,
    env_vars=None,
    image=None,
    pull_policy=None,
    enable_observability=True,
    source_config_file=None,
    runtime_config_file=None,
):
    """Start vLLM Semantic Router."""
    if env_vars is None:
        env_vars = {}
    stack_layout = resolve_runtime_stack()

    print_vllm_logo()
    source_config_file = source_config_file or config_file
    runtime_config_file = runtime_config_file or config_file
    listeners = load_config(source_config_file).get("listeners", [])
    if not listeners:
        log.error("No listeners configured in config.yaml")
        raise SystemExit(1)

    log_startup_banner(source_config_file, listeners, stack_layout)
    ensure_clean_runtime_container(stack_layout.container_name)

    shared_network_name = stack_layout.network_name
    config_dir = os.path.dirname(os.path.abspath(source_config_file))
    ensure_shared_network(shared_network_name)
    network_name = start_observability_stack(
        enable_observability,
        shared_network_name,
        config_dir,
        env_vars,
        stack_layout,
    )
    fleet_sim_enabled = start_fleet_sim_sidecar(
        config_dir,
        env_vars,
        stack_layout,
        pull_policy=pull_policy,
    )
    if fleet_sim_enabled:
        env_vars.setdefault("TARGET_FLEET_SIM_URL", stack_layout.fleet_sim_service_url)

    dashboard_disabled = env_vars.get("DISABLE_DASHBOARD") == "true"
    setup_mode = str(env_vars.get("VLLM_SR_SETUP_MODE", "")).lower() == "true"
    return_code, _stdout, stderr = docker_start_vllm_sr(
        config_file=source_config_file,
        env_vars=env_vars,
        listeners=listeners,
        image=image,
        pull_policy=pull_policy,
        network_name=network_name,
        openclaw_network_name=shared_network_name,
        minimal=dashboard_disabled,
        stack_layout=stack_layout,
        state_root_dir=config_dir,
        runtime_config_file=runtime_config_file,
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
        listeners,
        stack_layout,
        dashboard_disabled,
        enable_observability,
        fleet_sim_enabled,
    )


def stop_vllm_sr():
    """Stop vLLM Semantic Router and observability containers."""
    log.info("Stopping vLLM Semantic Router...")
    stack_layout = resolve_runtime_stack()
    container_statuses = _managed_container_statuses(stack_layout)
    if _all_containers_absent(container_statuses):
        log.info("Container not found. Nothing to stop.")
        return

    openclaw_data_dir = resolve_openclaw_data_dir(os.getcwd())
    network_name = stack_layout.network_name
    _disconnect_openclaw_registry_containers(
        network_name,
        load_openclaw_registry(openclaw_data_dir),
    )
    _stop_managed_container(
        stack_layout.container_name,
        container_statuses[stack_layout.container_name],
        stopped_message="vLLM Semantic Router stopped",
    )
    _stop_managed_container(
        stack_layout.fleet_sim_container_name,
        container_statuses[stack_layout.fleet_sim_container_name],
        stop_message=f"Stopping {stack_layout.fleet_sim_container_name}...",
        stopped_message=f"{stack_layout.fleet_sim_container_name} stopped",
    )
    for container_name in _observability_container_names(stack_layout):
        _stop_managed_container(
            container_name,
            container_statuses[container_name],
            stop_message=f"Stopping {container_name}...",
            stopped_message=f"{container_name} stopped",
        )
    _remove_runtime_network(network_name)


def _managed_container_statuses(stack_layout: RuntimeStackLayout) -> dict[str, str]:
    container_names = [
        stack_layout.container_name,
        stack_layout.fleet_sim_container_name,
        *_observability_container_names(stack_layout),
    ]
    return {
        container_name: docker_container_status(container_name)
        for container_name in container_names
    }


def _all_containers_absent(container_statuses: dict[str, str]) -> bool:
    return all(status == "not found" for status in container_statuses.values())


def _disconnect_openclaw_registry_containers(
    network_name: str, openclaw_entries: list[dict[str, str]]
) -> None:
    for entry in openclaw_entries:
        name = entry.get("name") or entry.get("containerName")
        if not name:
            continue
        _disconnect_openclaw_container(network_name, name)


def _disconnect_openclaw_container(network_name: str, container_name: str) -> None:
    container_status = docker_container_status(container_name)
    if container_status == "not found":
        return
    if container_status == "running":
        log.info(f"Stopping OpenClaw container: {container_name}")
        docker_stop_container(container_name)
    log.info(f"Disconnecting {container_name} from {network_name}")
    docker_network_disconnect(network_name, container_name)


def _stop_managed_container(
    container_name: str,
    container_status: str,
    *,
    stop_message: str | None = None,
    stopped_message: str | None = None,
) -> None:
    if container_status == "not found":
        return
    if stop_message and container_status == "running":
        log.info(stop_message)
    if container_status == "running":
        docker_stop_container(container_name)
    docker_remove_container(container_name)
    if stopped_message:
        log.info(stopped_message)


def _observability_container_names(stack_layout: RuntimeStackLayout) -> tuple[str, ...]:
    return (
        stack_layout.grafana_container_name,
        stack_layout.prometheus_container_name,
        stack_layout.jaeger_container_name,
    )


def _remove_runtime_network(network_name: str) -> None:
    return_code, _stdout, _stderr = docker_remove_network(network_name)
    if return_code == 0:
        log.info(f"Network {network_name} removed")


def show_logs(service: str, follow: bool = False):
    """Show logs from a runtime service."""
    _validate_runtime_service(service)
    stack_layout = resolve_runtime_stack()
    if service == "simulator":
        _ensure_runtime_container_available(stack_layout.fleet_sim_container_name)
        docker_logs(stack_layout.fleet_sim_container_name, follow=follow, tail=200)
        return

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
    sim_status = docker_container_status(stack_layout.fleet_sim_container_name)
    if status == "not found":
        if sim_status == "running":
            log.info(
                "Status: Router stack not running, but simulator sidecar is still running"
            )
            log.info(f"Simulator: Running ({stack_layout.fleet_sim_url})")
            log.info("Stop it with: vllm-sr stop")
            return
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
    log.info("For detailed logs: vllm-sr logs <envoy|router|dashboard|simulator>")
    log.info("=" * 60)


def _validate_runtime_service(service: str) -> None:
    if service == "simulator" or service in SERVICE_LOG_PATTERNS:
        return
    log.error(f"Invalid service: {service}")
    log.error("Must be 'envoy', 'router', 'dashboard', or 'simulator'")
    raise SystemExit(1)


def _requested_services(service: str) -> list[str]:
    if service == "all":
        return ["router", "envoy", "dashboard", "simulator"]
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
        "simulator": (
            "Fleet Sim",
            _check_fleet_sim_status,
            stack_layout.fleet_sim_url,
        ),
    }
    label, checker, detail = checkers[service]
    try:
        container_name = (
            stack_layout.fleet_sim_container_name
            if service == "simulator"
            else stack_layout.container_name
        )
        is_running = checker(container_name)
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


def _check_fleet_sim_status(container_name: str) -> bool:
    return_code, stdout, _stderr = docker_exec(
        container_name,
        [
            "python",
            "-c",
            (
                "import sys, urllib.request; "
                "resp = urllib.request.urlopen('http://localhost:8000/healthz', timeout=3); "
                "sys.stdout.write(str(resp.getcode()))"
            ),
        ],
    )
    return return_code == 0 and stdout.strip() == "200"


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
