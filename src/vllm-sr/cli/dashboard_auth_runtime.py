"""Immutable local-runtime plan for dashboard authentication settings."""

from __future__ import annotations

import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

DASHBOARD_JWT_SECRET_ENV = "DASHBOARD_JWT_SECRET"
DASHBOARD_JWT_EXPIRY_HOURS_ENV = "DASHBOARD_JWT_EXPIRY_HOURS"
DASHBOARD_ADMIN_EMAIL_ENV = "DASHBOARD_ADMIN_EMAIL"
DASHBOARD_ADMIN_PASSWORD_ENV = "DASHBOARD_ADMIN_PASSWORD"
DASHBOARD_ADMIN_NAME_ENV = "DASHBOARD_ADMIN_NAME"
DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV = "DASHBOARD_PASSWORD_BLOCKLIST_PATH"
DASHBOARD_ALLOW_OPEN_BOOTSTRAP_ENV = "DASHBOARD_ALLOW_OPEN_BOOTSTRAP"

DASHBOARD_AUTH_ENV_NAMES = frozenset(
    {
        DASHBOARD_JWT_SECRET_ENV,
        DASHBOARD_JWT_EXPIRY_HOURS_ENV,
        DASHBOARD_ADMIN_EMAIL_ENV,
        DASHBOARD_ADMIN_PASSWORD_ENV,
        DASHBOARD_ADMIN_NAME_ENV,
        DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV,
        DASHBOARD_ALLOW_OPEN_BOOTSTRAP_ENV,
    }
)
DASHBOARD_SECRET_ENV_NAMES = frozenset(
    {DASHBOARD_JWT_SECRET_ENV, DASHBOARD_ADMIN_PASSWORD_ENV}
)
DASHBOARD_PASSWORD_BLOCKLIST_CONTAINER_PATH = (
    "/etc/vllm-sr/dashboard-auth/password-blocklist.txt"
)


@dataclass(frozen=True)
class DashboardAuthRuntimePlan:
    """Validated dashboard-only env and mounts for one local startup."""

    enabled: bool
    _container_env_items: tuple[tuple[str, str], ...] = field(repr=False)
    mount_specs: tuple[str, ...] = ()

    @property
    def container_env(self) -> dict[str, str]:
        """Return a defensive copy of the dashboard container environment."""
        return dict(self._container_env_items)

    @property
    def inherited_secret_names(self) -> frozenset[str]:
        """Secret names that Docker must inherit instead of receiving in argv."""
        present_names = {name for name, _ in self._container_env_items}
        return DASHBOARD_SECRET_ENV_NAMES & present_names

    @property
    def secret_process_env(self) -> dict[str, str]:
        """Return secret values for only the dashboard Docker subprocess."""
        return {
            name: value
            for name, value in self._container_env_items
            if name in DASHBOARD_SECRET_ENV_NAMES
        }


DISABLED_DASHBOARD_AUTH_PLAN = DashboardAuthRuntimePlan(False, ())


def build_dashboard_auth_runtime_plan(
    env_vars: Mapping[str, str] | None,
    *,
    dashboard_enabled: bool,
) -> DashboardAuthRuntimePlan:
    """Validate dashboard auth inputs before local-runtime state can be mutated."""
    if not dashboard_enabled:
        return DISABLED_DASHBOARD_AUTH_PLAN

    source = env_vars or {}
    container_env = {
        name: str(source[name]) for name in DASHBOARD_AUTH_ENV_NAMES if name in source
    }
    mount_specs: tuple[str, ...] = ()

    blocklist_value = container_env.pop(DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV, "")
    if blocklist_value.strip():
        blocklist_path = _resolve_regular_blocklist_path(blocklist_value)
        container_env[DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV] = (
            DASHBOARD_PASSWORD_BLOCKLIST_CONTAINER_PATH
        )
        mount_specs = (
            f"{blocklist_path}:{DASHBOARD_PASSWORD_BLOCKLIST_CONTAINER_PATH}:ro,z",
        )

    return DashboardAuthRuntimePlan(
        True,
        tuple(sorted(container_env.items())),
        mount_specs,
    )


def without_dashboard_auth_env(
    env_vars: Mapping[str, str] | None,
) -> dict[str, str]:
    """Copy an environment mapping without dashboard-only auth settings."""
    return {
        name: value
        for name, value in (env_vars or {}).items()
        if name not in DASHBOARD_AUTH_ENV_NAMES
    }


def _resolve_regular_blocklist_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
        mode = resolved.stat().st_mode
    except OSError as exc:
        raise ValueError(
            f"{DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV} must reference an existing "
            "regular file"
        ) from exc

    if not stat.S_ISREG(mode):
        raise ValueError(
            f"{DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV} must reference an existing "
            "regular file"
        )
    return resolved
