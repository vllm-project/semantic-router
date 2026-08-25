"""Management API config helpers used by local runtime orchestration."""

from collections.abc import Mapping

from cli.consts import DEFAULT_API_PORT
from cli.runtime_env_names import runtime_env_name_is_allowed

_MAX_PORT = 65_535
_DEFAULT_READY_ROLES = frozenset({"viewer", "operator", "admin"})


def _configured_management_port(user_config: dict) -> int:
    management = _management_api_config(user_config)
    port = management.get("port", DEFAULT_API_PORT)
    if isinstance(port, bool) or not isinstance(port, int):
        raise ValueError("management API port must be between 1 and 65535")
    if port == 0:
        port = DEFAULT_API_PORT
    if not 1 <= port <= _MAX_PORT:
        raise ValueError("management API port must be between 1 and 65535")
    return port


def _configured_management_tls_certificate_file(user_config: dict) -> str | None:
    management = _management_api_config(user_config)
    tls = management.get("tls") or {}
    if not isinstance(tls, dict):
        raise ValueError("global.services.management_api.tls must be a mapping")
    certificate_file = tls.get("certificate_file")
    certificate_env = tls.get("certificate_env")
    private_file = tls.get("private_key_file")
    private_env = tls.get("private_key_env")
    if not any((certificate_file, certificate_env, private_file, private_env)):
        return None
    if isinstance(certificate_file, str) and certificate_file.strip():
        return certificate_file.strip()
    return ""


def _configured_management_readiness_token_env(
    user_config: dict, env_vars: Mapping[str, str]
) -> str | None:
    """Select an available bearer env without moving its value into process argv."""

    management = _management_api_config(user_config)
    auth = _bearer_management_auth(management)
    if auth is None:
        return None
    for _index, env_name, role in _resolved_management_tokens(auth, env_vars):
        if _role_can_read_readiness(auth, role):
            return env_name
    raise ValueError(
        "management API bearer auth requires an available token with ready.read "
        "permission for startup readiness checks"
    )


def _bearer_management_auth(management: dict) -> dict | None:
    auth = management.get("auth")
    if auth is None:
        auth = {}
    elif not isinstance(auth, dict):
        raise ValueError("global.services.management_api.auth must be a mapping")
    mode = auth.get("mode")
    if mode is None or mode == "":
        mode = "disabled"
    if not isinstance(mode, str):
        raise ValueError("management API auth mode must be disabled, bearer, or router")
    if mode in {"disabled", "router"}:
        return None
    if mode != "bearer":
        raise ValueError("management API auth mode must be disabled, bearer, or router")
    return auth


def _resolved_management_tokens(
    auth: dict, env_vars: Mapping[str, str]
) -> list[tuple[int, str, str]]:
    tokens = auth.get("tokens")
    if tokens is None:
        tokens = []
    elif not isinstance(tokens, list):
        raise ValueError("management API auth tokens must be a list")
    resolved_tokens: dict[str, tuple[int, str, str]] = {}
    for index, token in enumerate(tokens):
        if not isinstance(token, dict):
            raise ValueError("management API auth token entries must be mappings")
        env_name = _management_token_env_name(token.get("env"))
        role = _management_token_role(token.get("role"))
        value = env_vars.get(env_name, "")
        if not isinstance(value, str) or not value.strip():
            continue
        if "\r" in value or "\n" in value:
            raise ValueError("management API auth token must be a single line")
        resolved_tokens[value.strip()] = (index, env_name, role)
    return sorted(resolved_tokens.values())


def _management_token_env_name(value: object) -> str:
    if not isinstance(value, str) or not runtime_env_name_is_allowed(value):
        raise ValueError("management API auth token env name is invalid")
    return value


def _management_token_role(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("management API auth token role is invalid")
    return value.strip()


def _role_can_read_readiness(auth: dict, role: str) -> bool:
    roles = auth.get("roles")
    if roles is None:
        roles = {}
    elif not isinstance(roles, dict):
        raise ValueError("management API auth roles must be a mapping")
    if not roles:
        return role in _DEFAULT_READY_ROLES
    permissions = roles.get(role, [])
    if not isinstance(permissions, list):
        raise ValueError("management API auth role permissions must be a list")
    return any(
        isinstance(permission, str) and permission in {"ready.read", "*"}
        for permission in permissions
    )


def _management_api_config(user_config: dict) -> dict:
    global_config = user_config.get("global")
    if global_config is None:
        global_config = {}
    elif not isinstance(global_config, dict):
        raise ValueError("global must be a mapping")
    services = global_config.get("services")
    if services is None:
        services = {}
    elif not isinstance(services, dict):
        raise ValueError("global.services must be a mapping")
    management = services.get("management_api")
    if management is None:
        management = {}
    elif not isinstance(management, dict):
        raise ValueError("global.services.management_api must be a mapping")
    return management
