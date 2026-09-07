"""Management-listener contract for the local split container stack."""

from cli.consts import DEFAULT_METRICS_PORT, DEFAULT_ROUTER_PORT
from cli.parser import parse_user_config
from cli.runtime_stack import RuntimeStackLayout

_MAX_PORT = 65_535
_ROUTER_SERVICE_PORTS = frozenset({DEFAULT_ROUTER_PORT, DEFAULT_METRICS_PORT})


def _managed_management_listener(
    config_path: str, stack_layout: RuntimeStackLayout
) -> dict[str, int | str]:
    """Resolve the management listener contract for the split Docker stack."""

    management = _management_api_config(config_path)
    bind_address, port = _management_endpoint(management)
    _validate_management_access(management)
    _validate_management_endpoint(bind_address, port)
    host_port = port + stack_layout.port_offset
    if host_port > _MAX_PORT:
        raise ValueError("management API host port is outside the valid range")
    return {
        "bind_address": bind_address,
        "port": port,
        "host_port": host_port,
    }


def _management_api_config(config_path: str) -> dict | None:
    config = parse_user_config(config_path)
    global_config = config.global_ or {}
    services = global_config.get("services")
    if services is None:
        return None
    elif not isinstance(services, dict):
        raise ValueError("global.services must be a mapping")
    management = services.get("management_api")
    if management is not None and not isinstance(management, dict):
        raise ValueError("global.services.management_api must be a mapping")
    return management


def _management_endpoint(management: dict | None) -> tuple[str, int]:
    # Normal CLI realization materializes this local default. Retain the same
    # fallback for focused callers that invoke container_start directly.
    if management is None:
        return "0.0.0.0", 8080
    bind_address = str(management.get("bind_address") or "127.0.0.1").strip()
    raw_port = management.get("port", 8080)
    if isinstance(raw_port, bool) or not isinstance(raw_port, int):
        raise ValueError("management API port must be an integer")
    return bind_address, raw_port


def _validate_management_access(management: dict | None) -> None:
    management_config = management or {}
    remote_exposure = management_config.get("remote_exposure", False)
    if not isinstance(remote_exposure, bool):
        raise ValueError("management API remote_exposure must be a boolean")
    auth = management_config.get("auth", {})
    if not isinstance(auth, dict):
        raise ValueError("global.services.management_api.auth must be a mapping")
    auth_mode = str(auth.get("mode") or "disabled").strip()
    if auth_mode not in {"disabled", "bearer"}:
        raise ValueError("management API auth mode must be disabled or bearer")
    tokens = auth.get("tokens", [])
    if not isinstance(tokens, list):
        raise ValueError("management API auth tokens must be a list")
    if remote_exposure and (auth_mode != "bearer" or not tokens):
        raise ValueError("management API remote exposure requires bearer auth tokens")


def _validate_management_endpoint(bind_address: str, port: int) -> None:
    if bind_address != "0.0.0.0":
        raise ValueError(
            "split Docker requires management_api.bind_address 0.0.0.0 so "
            "Dashboard and Envoy can reach the Router"
        )
    if port < 1 or port > _MAX_PORT:
        raise ValueError("management API port must be between 1 and 65535")
    if port in _ROUTER_SERVICE_PORTS:
        raise ValueError("management API port conflicts with a Router service port")
