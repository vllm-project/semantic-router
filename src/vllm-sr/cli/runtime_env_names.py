"""Shared admission policy for host environment names used at runtime."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

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
        "DASHBOARD_JWT_SECRET",
        "VLLM_SR_PLATFORM",
        "VLLM_SR_RECIPE_ENV_ALLOWLIST",
        "VLLM_SR_RUNTIME_CONFIG_PATH",
        "VLLM_SR_SOURCE_CONFIG_PATH",
        "VLLM_SR_STATE_ROOT_DIR",
        "VLLM_SR_CONFIG_BASE_DIR",
        "VLLM_SR_LISTENER_TLS_DIRS",
        "VLLM_SR_LISTENER_TLS_PATHS",
        "VLLM_SR_KEY_FILE_GROUP",
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


_DOLLAR = "\x00DOLLAR\x00"


def expand_router_references(value: str, lookup: Callable[[str], str | None]) -> str:
    """value with its environment references expanded as the Router expands
    them when it loads a config: ${NAME} and $NAME, ${NAME:-default} when NAME
    is unset or empty, ${NAME-default} when it is unset, and $$ for a literal
    $. NAME is uppercase; any other $ text, such as ${lowercase}, stays as
    written. lookup returns a variable's value, or None when it is unset."""

    if "$" not in value:
        return value
    escaped = value.replace("$$", _DOLLAR)
    out: list[str] = []
    index = 0
    while index < len(escaped):
        if escaped[index] != "$":
            out.append(escaped[index])
            index += 1
            continue
        if escaped.startswith("${", index):
            close = escaped.find("}", index + 2)
            if close < 0:
                out.append("$")
                index += 1
                continue
            out.append(_resolve_braced(escaped[index + 2 : close], lookup))
            index = close + 1
            continue
        end = index + 1
        while end < len(escaped) and (
            escaped[end].isascii() and (escaped[end].isalnum() or escaped[end] == "_")
        ):
            end += 1
        name = escaped[index + 1 : end]
        if _ENV_NAME.fullmatch(name):
            out.append(lookup(name) or "")
        else:
            out.append(escaped[index:end])
        index = end
    return "".join(out).replace(_DOLLAR, "$")


def _resolve_braced(inner: str, lookup: Callable[[str], str | None]) -> str:
    name, separator, default = inner.partition("-")
    empty_uses_default = False
    if separator and name.endswith(":"):
        name, empty_uses_default = name[:-1], True
    if not _ENV_NAME.fullmatch(name):
        return "${" + inner + "}"
    value = lookup(name)
    if not separator:
        return value or ""
    if value is not None and (value != "" or not empty_uses_default):
        return value
    return default
