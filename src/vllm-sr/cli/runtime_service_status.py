"""Per-service readiness probes behind ``vllm-sr status``.

Each managed service answers "is it up" differently -- Router and Dashboard over
HTTP inside the container, Envoy through its admin ``/ready`` with a config
validation fallback, the simulator through a short Python one-liner -- and none
of that is runtime orchestration. Keeping the probes here leaves ``core`` owning
the start and stop flow, which is what its ratchet is for.

Every probe reports "not ready" rather than raising: status output must survive
a container that is up but not answering yet.
"""

from pathlib import Path

import yaml

from cli.commands.runtime_paths import _container_compiled_bootstrap_path
from cli.consts import DEFAULT_ENVOY_PORT
from cli.container_cli import container_exec, container_status
from cli.runtime_management_config import (
    _configured_management_port,
    _configured_management_tls_certificate_file,
)
from cli.runtime_stack import RuntimeStackLayout
from cli.terminal import fields
from cli.utils import get_logger

log = get_logger(__name__)


def runtime_service_container_name(
    service: str, stack_layout: RuntimeStackLayout
) -> str:
    """Return the container that serves *service* in this stack."""

    return stack_layout.service_container_name(service)


def report_service_status(service: str, stack_layout) -> None:
    checkers = {
        "router": (
            "Router",
            lambda container_name: _check_router_status(container_name, stack_layout),
            None,
        ),
        "envoy": (
            "Envoy",
            lambda container_name: _check_envoy_status(container_name, stack_layout),
            None,
        ),
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
            else runtime_service_container_name(service, stack_layout)
        )
        is_running = checker(container_name)
        _log_service_status(label, is_running, detail if is_running else None)
    except Exception as exc:
        log.error(f"Failed to check {service} status: {exc}")


def _check_router_status(container_name: str, stack_layout: RuntimeStackLayout) -> bool:
    config_path = _container_compiled_bootstrap_path(
        Path("config.yaml"), stack_name=stack_layout.stack_name
    )
    return_code, stdout, _stderr = container_exec(container_name, ["cat", config_path])
    if return_code != 0:
        return False
    config = yaml.safe_load(stdout) or {}
    management_port = _configured_management_port(config)
    certificate_file = _configured_management_tls_certificate_file(config)
    scheme = "https" if certificate_file is not None else "http"
    tls_arguments = ["--cacert", certificate_file] if certificate_file else []
    return_code, _stdout, _stderr = container_exec(
        container_name,
        [
            "curl",
            "-f",
            "-s",
            *tls_arguments,
            f"{scheme}://localhost:{management_port}/health",
        ],
    )
    return return_code == 0


def _check_envoy_status(container_name: str, stack_layout: RuntimeStackLayout) -> bool:
    return_code, stdout, _stderr = container_exec(
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

    return _fallback_check_envoy_status(container_name, stack_layout)


def _fallback_check_envoy_status(
    container_name: str, stack_layout: RuntimeStackLayout
) -> bool:
    if container_name != stack_layout.envoy_container_name:
        return False
    if container_status(container_name) != "running":
        return False

    return_code, _stdout, _stderr = container_exec(
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


def _check_dashboard_status(container_name: str) -> bool:
    return_code, stdout, _stderr = container_exec(
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
    return_code, stdout, _stderr = container_exec(
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
        fields(((label, "Status unknown"),))
        return
    if detail:
        fields(((label, f"Running ({detail})"),))
        return
    fields(((label, "Running"),))
