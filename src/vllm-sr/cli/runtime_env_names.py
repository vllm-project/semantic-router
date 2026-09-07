"""Shared admission policy for host environment names used at runtime."""

from __future__ import annotations

import re
from collections.abc import Iterable

_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")

# These names either expose process identity or control CLI/container lifecycle.
# A user-authored config must not repurpose them as credentials or Recipe inputs.
RESERVED_RUNTIME_ENV_NAMES = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "SHELL",
        "PWD",
        "LOGNAME",
        "VLLM_SR_SETUP_MODE",
        "DASHBOARD_SETUP_MODE",
        "VLLM_SR_ALGORITHM_OVERRIDE",
        "DISABLE_DASHBOARD",
        "DASHBOARD_READONLY",
        "DASHBOARD_PLATFORM",
        "VLLM_SR_PLATFORM",
        "VLLM_SR_RECIPE_ENV_ALLOWLIST",
        "VLLM_SR_RUNTIME_CONFIG_PATH",
        "VLLM_SR_SOURCE_CONFIG_PATH",
        "VLLM_SR_STATE_ROOT_DIR",
        "VLLM_SR_CONFIG_BASE_DIR",
        "VLLM_SR_RECIPE_STORE_DIR",
        "VLLM_SR_MANAGED_STORAGE_BACKENDS",
        "VLLM_SR_ACTIVE_RECIPE_DIR",
        "VLLM_SR_STACK_NAME",
        "VLLM_SR_STACK_POSTGRES_PASSWORD",
        "VLLM_SR_STACK_REDIS_PASSWORD",
    }
)


def runtime_env_name_is_allowed(value: object) -> bool:
    """Return whether *value* is one exact, non-reserved host env name."""

    return (
        isinstance(value, str)
        and value == value.strip()
        and _ENV_NAME.fullmatch(value) is not None
        and value not in RESERVED_RUNTIME_ENV_NAMES
    )


def normalize_runtime_env_names(values: Iterable[object]) -> tuple[str, ...]:
    """Validate, deduplicate, and stabilize explicit runtime env names."""

    normalized: set[str] = set()
    for value in values:
        if not runtime_env_name_is_allowed(value):
            raise ValueError("invalid uppercase, non-reserved runtime environment name")
        normalized.add(value)
    return tuple(sorted(normalized))
