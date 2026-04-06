"""Runtime status and log support for the local CLI flow."""

from __future__ import annotations

import subprocess

from cli.consts import DEFAULT_API_PORT, DEFAULT_ENVOY_PORT
from cli.runtime_stack import RuntimeStackLayout

SERVICE_LOG_PATTERNS = {
    "router": r'"caller"|spawned: \'router\'|success: router|cli\.commands',
    "dashboard": r"dashboard|Dashboard|spawned: 'dashboard'|success: dashboard|:8700",
    "envoy": (
        r"\[2[0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].*\]\[.*\]"
        r"|spawned: 'envoy'|success: envoy"
    ),
}


def show_logs(
    service: str,
    *,
    follow: bool,
    validate_runtime_service,
    resolve_runtime_stack,
    ensure_runtime_container_available,
    docker_logs,
    runtime_service_container_name,
    log_command,
    log,
) -> None:
    """Show logs from a runtime service."""
    validate_runtime_service(service)
    stack_layout = resolve_runtime_stack()
    if service == "simulator":
        ensure_runtime_container_available(stack_layout.fleet_sim_container_name)
        docker_logs(stack_layout.fleet_sim_container_name, follow=follow, tail=200)
        return

    container_name = runtime_service_container_name(service, stack_layout)
    ensure_runtime_container_available(container_name)

    if follow:
        log.info(f"Following {service} logs (Ctrl+C to stop)...")
        log.info("")

    command = log_command(container_name, SERVICE_LOG_PATTERNS[service], follow)
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


def show_status(
    service: str,
    *,
    resolve_runtime_stack,
    resolve_runtime_status_snapshot,
    requested_services,
    report_service_status,
    log,
) -> None:
    """Show runtime service status."""
    stack_layout = resolve_runtime_stack()
    status, sim_status = resolve_runtime_status_snapshot(stack_layout)
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

    for requested_service in requested_services(service):
        report_service_status(requested_service, stack_layout)

    log.info("")
    log.info("For detailed logs: vllm-sr logs <envoy|router|dashboard|simulator>")
    log.info("=" * 60)


def resolve_runtime_status_snapshot(
    stack_layout: RuntimeStackLayout,
    *,
    runtime_stack_status,
    docker_container_status,
    log,
) -> tuple[str, str]:
    try:
        return (
            runtime_stack_status(stack_layout),
            docker_container_status(stack_layout.fleet_sim_container_name),
        )
    except SystemExit:
        log.info("Status: Not running")
        log.info(
            "Docker daemon is not reachable, so local container status cannot be inspected"
        )
        log.info("Start with: vllm-sr serve")
        return "not found", "not found"


def validate_runtime_service(service: str, *, log) -> None:
    if service == "simulator" or service in SERVICE_LOG_PATTERNS:
        return
    log.error(f"Invalid service: {service}")
    log.error("Must be 'envoy', 'router', 'dashboard', or 'simulator'")
    raise SystemExit(1)


def requested_services(service: str, *, validate_runtime_service) -> list[str]:
    if service == "all":
        return ["router", "envoy", "dashboard", "simulator"]
    validate_runtime_service(service)
    return [service]


def ensure_runtime_container_available(
    container_name: str,
    *,
    docker_container_status,
    log,
) -> None:
    if docker_container_status(container_name) != "not found":
        return
    log.error("Container not found. Is vLLM Semantic Router running?")
    log.info("Start it with: vllm-sr serve")
    raise SystemExit(1)


def log_command(container_name: str, grep_pattern: str, follow: bool) -> str:
    if follow:
        return f'docker logs -f {container_name} 2>&1 | grep -E "{grep_pattern}"'
    return (
        f'docker logs --tail 200 {container_name} 2>&1 | grep -E "{grep_pattern}" '
        "| tail -50"
    )


def runtime_service_container_name(
    service: str, stack_layout: RuntimeStackLayout
) -> str:
    return stack_layout.service_container_name(service)


def runtime_stack_status(
    stack_layout: RuntimeStackLayout,
    *,
    docker_container_status,
) -> str:
    fallback_status = "not found"
    for container_name in stack_layout.runtime_container_names:
        status = docker_container_status(container_name)
        if status == "running":
            return status
        if status != "not found" and fallback_status == "not found":
            fallback_status = status
    return fallback_status


def report_service_status(
    service: str,
    stack_layout: RuntimeStackLayout,
    *,
    runtime_service_container_name,
    check_router_status,
    check_envoy_status,
    check_dashboard_status,
    check_fleet_sim_status,
    log_service_status,
    log,
) -> None:
    checkers = {
        "router": ("Router", check_router_status, None),
        "envoy": (
            "Envoy",
            lambda container_name: check_envoy_status(container_name, stack_layout),
            None,
        ),
        "dashboard": (
            "Dashboard",
            check_dashboard_status,
            stack_layout.dashboard_url,
        ),
        "simulator": (
            "Fleet Sim",
            check_fleet_sim_status,
            stack_layout.fleet_sim_url,
        ),
    }
    label, checker, detail = checkers[service]
    try:
        container_name = (
            stack_layout.fleet_sim_container_name
            if service == "simulator"
            else runtime_service_container_name(service, stack_layout)
        )
        is_running = checker(container_name)
        log_service_status(label, is_running, detail if is_running else None)
    except Exception as exc:
        log.error(f"Failed to check {service} status: {exc}")


def check_router_status(container_name: str, *, docker_exec) -> bool:
    return_code, _stdout, _stderr = docker_exec(
        container_name,
        ["curl", "-f", "-s", f"http://localhost:{DEFAULT_API_PORT}/health"],
    )
    return return_code == 0


def check_envoy_status(
    container_name: str,
    stack_layout: RuntimeStackLayout,
    *,
    docker_exec,
    docker_container_status,
) -> bool:
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
    if return_code == 0 and stdout.strip() == "200":
        return True

    return fallback_check_envoy_status(
        container_name,
        stack_layout,
        docker_exec=docker_exec,
        docker_container_status=docker_container_status,
    )


def fallback_check_envoy_status(
    container_name: str,
    stack_layout: RuntimeStackLayout,
    *,
    docker_exec,
    docker_container_status,
) -> bool:
    if container_name != stack_layout.envoy_container_name:
        return False
    if docker_container_status(container_name) != "running":
        return False

    return_code, _stdout, _stderr = docker_exec(
        container_name,
        [
            "/usr/local/bin/envoy",
            "--mode",
            "validate",
            "-c",
            "/etc/envoy/envoy.yaml",
        ],
    )
    return return_code == 0


def check_dashboard_status(container_name: str, *, docker_exec) -> bool:
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


def check_fleet_sim_status(container_name: str, *, docker_exec) -> bool:
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


def log_service_status(
    label: str, is_running: bool, detail: str | None = None, *, log
) -> None:
    if not is_running:
        log.info(f"WARNING {label}: Status unknown")
        return
    if detail:
        log.info(f"{label}: Running ({detail})")
        return
    log.info(f"{label}: Running")
