"""Local split-runtime defaults for the Looper integration."""

from __future__ import annotations

import ipaddress
from urllib.parse import SplitResult, urlsplit

from cli.consts import DEFAULT_LISTENER_PORT
from cli.runtime_stack import DEFAULT_ENVOY_CONTAINER_NAME, RuntimeStackLayout

LOOPER_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


def apply_local_looper_endpoint(
    config: dict[str, object], stack_layout: RuntimeStackLayout
) -> bool:
    """Point a missing or loopback Looper endpoint at stack-local Envoy."""
    mappings = _local_looper_mappings(config)
    if mappings is None:
        return False
    global_config, integrations, looper = mappings

    endpoint = looper.get("endpoint")
    if endpoint is not None and not isinstance(endpoint, str):
        return False
    if isinstance(endpoint, str) and endpoint.strip():
        try:
            parsed = _parse_endpoint(endpoint)
        except ValueError:
            return False
        if not _is_stack_local_endpoint(parsed, stack_layout):
            return False
    else:
        parsed = None

    if "global" not in config:
        config["global"] = global_config
    if "integrations" not in global_config:
        global_config["integrations"] = integrations
    if "looper" not in integrations:
        integrations["looper"] = looper

    listener_port = _effective_listener_port(config)
    service_url = stack_layout.envoy_listener_service_url(listener_port)
    rewritten = service_url + _endpoint_path_and_query(parsed)
    if endpoint == rewritten:
        return False
    looper["endpoint"] = rewritten
    return True


def _local_looper_mappings(
    config: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]] | None:
    global_config = config.get("global")
    if global_config is None:
        global_config = {}
    if not isinstance(global_config, dict):
        return None

    integrations = global_config.get("integrations")
    if integrations is None:
        integrations = {}
    if not isinstance(integrations, dict):
        return None

    looper = integrations.get("looper")
    if looper is None:
        looper = {}
    if not isinstance(looper, dict):
        return None
    return global_config, integrations, looper


def _effective_listener_port(config: dict[str, object]) -> int:
    listeners = config.get("listeners")
    if isinstance(listeners, list):
        for listener in listeners:
            if not isinstance(listener, dict):
                continue
            port = listener.get("port")
            if isinstance(port, int) and not isinstance(port, bool) and port > 0:
                return port
    return DEFAULT_LISTENER_PORT


def _parse_endpoint(endpoint: str) -> SplitResult:
    stripped = endpoint.strip()
    if "://" in stripped:
        return urlsplit(stripped)
    return urlsplit(f"//{stripped}")


def _is_loopback_endpoint(parsed: SplitResult) -> bool:
    try:
        host = parsed.hostname
    except ValueError:
        return False
    if not host:
        return False
    normalized_host = host.rstrip(".").lower()
    if normalized_host == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized_host).is_loopback
    except ValueError:
        return False


def _is_stack_local_endpoint(
    parsed: SplitResult, stack_layout: RuntimeStackLayout
) -> bool:
    if _is_loopback_endpoint(parsed):
        return True
    try:
        host = parsed.hostname
    except ValueError:
        return False
    if not host:
        return False
    return host.rstrip(".").lower() in {
        DEFAULT_ENVOY_CONTAINER_NAME,
        stack_layout.envoy_container_name.lower(),
    }


def _endpoint_path_and_query(parsed: SplitResult | None) -> str:
    path = parsed.path if parsed is not None else ""
    if not path or path == "/":
        path = LOOPER_CHAT_COMPLETIONS_PATH
    elif not path.startswith("/"):
        path = f"/{path}"
    if parsed is not None and parsed.query:
        return f"{path}?{parsed.query}"
    return path
