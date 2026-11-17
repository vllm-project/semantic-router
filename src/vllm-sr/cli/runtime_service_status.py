"""Per-service readiness probes behind ``vllm-sr status``.

Each managed service answers "is it up" differently -- Router and Dashboard over
HTTP inside the container, and Envoy through its admin ``/ready`` with a config
validation fallback. None of that is runtime orchestration. Keeping the probes here leaves ``core`` owning
the start and stop flow, which is what its ratchet is for.

Every probe reports "not ready" rather than raising: status output must survive
a container that is up but not answering yet.
"""

import yaml

from cli.consts import DEFAULT_ENVOY_PORT
from cli.container_cli import container_exec, container_status
from cli.decision_model import configured_decision_model
from cli.runtime_management_config import (
    ROUTER_MANAGEMENT_PORT_PATH,
    _configured_management_port,
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
        "router": ("Router", _check_router_status, None),
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
    }
    label, checker, detail = checkers[service]
    try:
        container_name = runtime_service_container_name(service, stack_layout)
        is_running = checker(container_name)
        _log_service_status(label, is_running, detail if is_running else None)
        if service == "router" and is_running:
            decision_model = active_decision_model(stack_layout)
            if decision_model:
                fields((("Decision model", decision_model),))
    except Exception as exc:
        log.error(f"Failed to check {service} status: {exc}")


def _active_router_config(container_name: str) -> dict | None:
    """Read the running Router's config without exposing its contents in errors."""

    try:
        return_code, stdout, _stderr = container_exec(
            container_name,
            ["sh", "-c", 'cat "$VLLM_SR_RUNTIME_CONFIG_PATH"'],
        )
        if return_code != 0:
            return None
        document = yaml.safe_load(stdout) or {}
        return document if isinstance(document, dict) else None
    except Exception:  # status output must survive an unreadable config
        return None


def active_decision_model(stack_layout: RuntimeStackLayout) -> str | None:
    """The decision model of the Router's active config, or None when unreadable."""

    document = _active_router_config(stack_layout.service_container_name("router"))
    if document is None:
        return None
    try:
        return configured_decision_model(document)
    except (AttributeError, TypeError, ValueError):
        return None


def _check_router_status(container_name: str) -> bool:
    try:
        port = _router_management_port(container_name)
    except ValueError:
        return False
    return_code, _stdout, _stderr = container_exec(
        container_name,
        ["curl", "-f", "-s", f"http://localhost:{port}/health"],
    )
    return return_code == 0


def _router_management_port(container_name: str) -> int:
    try:
        return_code, stdout, _stderr = container_exec(
            container_name, ["cat", ROUTER_MANAGEMENT_PORT_PATH]
        )
    except Exception:  # retain compatibility with containers without a record
        return_code = 1
    if return_code == 0:
        value = stdout.strip()
        if not value.isascii() or not value.isdecimal() or int(value) == 0:
            raise ValueError("invalid Router management port record")
        return _configured_management_port(
            {"global": {"services": {"management_api": {"port": int(value)}}}}
        )
    # Containers created before the startup wrapper use this best-effort path.
    # Managed recreation enables the record before a future config can replace it.
    return _configured_management_port(_active_router_config(container_name) or {})


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
