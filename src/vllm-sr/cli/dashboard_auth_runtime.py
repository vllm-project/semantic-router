"""Immutable local-runtime plan for dashboard authentication settings."""

from __future__ import annotations

import hashlib
import hmac
import re
import stat
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

DASHBOARD_JWT_SECRET_ENV = "DASHBOARD_JWT_SECRET"
DASHBOARD_JWT_EXPIRY_HOURS_ENV = "DASHBOARD_JWT_EXPIRY_HOURS"
DASHBOARD_ADMIN_EMAIL_ENV = "DASHBOARD_ADMIN_EMAIL"
DASHBOARD_ADMIN_PASSWORD_ENV = "DASHBOARD_ADMIN_PASSWORD"
DASHBOARD_ADMIN_NAME_ENV = "DASHBOARD_ADMIN_NAME"
DASHBOARD_SECURITY_PROFILE_ENV = "DASHBOARD_SECURITY_PROFILE"
DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV = "DASHBOARD_PASSWORD_BLOCKLIST_PATH"
DASHBOARD_PASSWORD_BLOCKLIST_SHA256_ENV = "DASHBOARD_PASSWORD_BLOCKLIST_SHA256"
DASHBOARD_ALLOW_OPEN_BOOTSTRAP_ENV = "DASHBOARD_ALLOW_OPEN_BOOTSTRAP"

DASHBOARD_SECURITY_PROFILE_DEVELOPMENT = "development"
DASHBOARD_SECURITY_PROFILE_PRODUCTION = "production"

DASHBOARD_AUTH_ENV_NAMES = frozenset(
    {
        DASHBOARD_JWT_SECRET_ENV,
        DASHBOARD_JWT_EXPIRY_HOURS_ENV,
        DASHBOARD_ADMIN_EMAIL_ENV,
        DASHBOARD_ADMIN_PASSWORD_ENV,
        DASHBOARD_ADMIN_NAME_ENV,
        DASHBOARD_SECURITY_PROFILE_ENV,
        DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV,
        DASHBOARD_PASSWORD_BLOCKLIST_SHA256_ENV,
        DASHBOARD_ALLOW_OPEN_BOOTSTRAP_ENV,
    }
)
DASHBOARD_SECRET_ENV_NAMES = frozenset(
    {DASHBOARD_JWT_SECRET_ENV, DASHBOARD_ADMIN_PASSWORD_ENV}
)
DASHBOARD_PASSWORD_BLOCKLIST_CONTAINER_PATH = (
    "/etc/vllm-sr/dashboard-auth/password-blocklist.txt"
)
MAXIMUM_BLOCKLIST_FILE_BYTES = 8 * 1024 * 1024
MAXIMUM_BLOCKLIST_LINE_BYTES = 4096
MAXIMUM_BLOCKLIST_ENTRIES = 250_000
MINIMUM_PRODUCTION_BLOCKLIST_ENTRIES = 10_000


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
    profile = container_env.get(
        DASHBOARD_SECURITY_PROFILE_ENV, DASHBOARD_SECURITY_PROFILE_DEVELOPMENT
    ).strip()
    if profile not in {
        DASHBOARD_SECURITY_PROFILE_DEVELOPMENT,
        DASHBOARD_SECURITY_PROFILE_PRODUCTION,
    }:
        raise ValueError(
            f"{DASHBOARD_SECURITY_PROFILE_ENV} must be "
            f"{DASHBOARD_SECURITY_PROFILE_DEVELOPMENT!r} or "
            f"{DASHBOARD_SECURITY_PROFILE_PRODUCTION!r}"
        )
    container_env[DASHBOARD_SECURITY_PROFILE_ENV] = profile
    mount_specs: tuple[str, ...] = ()

    blocklist_value = container_env.pop(
        DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV, ""
    ).strip()
    expected_digest = (
        container_env.pop(DASHBOARD_PASSWORD_BLOCKLIST_SHA256_ENV, "").strip().lower()
    )
    if expected_digest and re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None:
        raise ValueError(
            f"{DASHBOARD_PASSWORD_BLOCKLIST_SHA256_ENV} must be exactly 64 "
            "hexadecimal characters"
        )
    if blocklist_value.strip():
        blocklist_path = _resolve_regular_blocklist_path(blocklist_value)
        actual_digest, unique_entry_count = _inspect_blocklist(blocklist_path)
        if expected_digest and not hmac.compare_digest(actual_digest, expected_digest):
            raise ValueError(
                f"{DASHBOARD_PASSWORD_BLOCKLIST_SHA256_ENV} does not match "
                "the selected password blocklist"
            )
        if profile == DASHBOARD_SECURITY_PROFILE_PRODUCTION:
            if not expected_digest:
                raise ValueError(
                    f"{DASHBOARD_SECURITY_PROFILE_ENV}=production requires "
                    f"{DASHBOARD_PASSWORD_BLOCKLIST_SHA256_ENV}"
                )
            if unique_entry_count < MINIMUM_PRODUCTION_BLOCKLIST_ENTRIES:
                raise ValueError(
                    "production password blocklist must contain at least "
                    f"{MINIMUM_PRODUCTION_BLOCKLIST_ENTRIES} unique NFC entries"
                )
        container_env[DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV] = (
            DASHBOARD_PASSWORD_BLOCKLIST_CONTAINER_PATH
        )
        if expected_digest:
            container_env[DASHBOARD_PASSWORD_BLOCKLIST_SHA256_ENV] = expected_digest
        mount_specs = (
            f"{blocklist_path}:{DASHBOARD_PASSWORD_BLOCKLIST_CONTAINER_PATH}:ro,z",
        )
    elif expected_digest:
        raise ValueError(
            f"{DASHBOARD_PASSWORD_BLOCKLIST_SHA256_ENV} requires "
            f"{DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV}"
        )
    elif profile == DASHBOARD_SECURITY_PROFILE_PRODUCTION:
        raise ValueError(
            f"{DASHBOARD_SECURITY_PROFILE_ENV}=production requires "
            f"{DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV}"
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


def _inspect_blocklist(path: Path) -> tuple[str, int]:
    try:
        size = path.stat().st_size
        if size > MAXIMUM_BLOCKLIST_FILE_BYTES:
            raise ValueError(
                f"{DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV} exceeds "
                f"{MAXIMUM_BLOCKLIST_FILE_BYTES} bytes"
            )
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"{DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV} could not be read"
        ) from exc
    if len(raw) > MAXIMUM_BLOCKLIST_FILE_BYTES:
        raise ValueError(
            f"{DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV} exceeds "
            f"{MAXIMUM_BLOCKLIST_FILE_BYTES} bytes"
        )

    unique_entries: set[str] = set()
    usable_entries = 0
    for raw_line in raw.split(b"\n"):
        if len(raw_line) > MAXIMUM_BLOCKLIST_LINE_BYTES:
            raise ValueError(
                f"{DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV} contains a line "
                f"over {MAXIMUM_BLOCKLIST_LINE_BYTES} bytes"
            )
        if not raw_line or raw_line.startswith(b"#"):
            continue
        try:
            entry = raw_line.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV} contains invalid Unicode"
            ) from exc
        usable_entries += 1
        if usable_entries > MAXIMUM_BLOCKLIST_ENTRIES:
            raise ValueError(
                f"{DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV} exceeds "
                f"{MAXIMUM_BLOCKLIST_ENTRIES} entries"
            )
        unique_entries.add(unicodedata.normalize("NFC", entry))

    if not unique_entries:
        raise ValueError(
            f"{DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV} contains no usable entries"
        )
    return hashlib.sha256(raw).hexdigest(), len(unique_entries)
