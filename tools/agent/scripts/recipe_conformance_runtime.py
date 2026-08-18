"""Runtime bindings needed by live Recipe conformance."""

from __future__ import annotations

import re
from typing import Any

RUNTIME_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DEFAULT_READY_ROLES = frozenset({"viewer", "operator", "admin"})


def management_auth_bindings(config: dict[str, Any]) -> list[tuple[str, bool]]:
    """Return configured bearer env names and identify a readiness credential."""

    management = _mapping(
        _mapping(_mapping(config.get("global")).get("services")).get("management_api")
    )
    auth = _mapping(management.get("auth"))
    mode = str(auth.get("mode") or "disabled").strip()
    if mode == "disabled":
        return []
    if mode != "bearer":
        raise ValueError("management API auth mode must be disabled or bearer")

    roles = _mapping(auth.get("roles"))
    bindings: list[tuple[str, bool]] = []
    for raw_token in _sequence(auth.get("tokens")):
        token = _mapping(raw_token)
        env_name = str(token.get("env") or "").strip()
        role = str(token.get("role") or "").strip()
        if not RUNTIME_ENV_NAME.fullmatch(env_name):
            raise ValueError("management API auth token env name is invalid")
        if not role:
            raise ValueError("management API auth token role is invalid")
        permissions = roles.get(role, [])
        if roles and not isinstance(permissions, list):
            raise ValueError("management API auth role permissions must be a list")
        can_read_ready = (
            role in DEFAULT_READY_ROLES
            if not roles
            else any(
                isinstance(permission, str) and permission in {"ready.read", "*"}
                for permission in permissions
            )
        )
        bindings.append((env_name, can_read_ready))

    if not any(can_read_ready for _env_name, can_read_ready in bindings):
        raise ValueError(
            "management API bearer auth requires a token with ready.read permission"
        )
    return bindings


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _sequence(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
