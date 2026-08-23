"""Shared primitives for the offline v0.3 to v0.4 config upgrade."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

_ENV_REFERENCE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SENSITIVE_KEYS = {
    "access_key",
    "api_key",
    "authorization",
    "client_secret",
    "password",
    "private_key",
    "secret",
    "token",
}
_SENSITIVE_SUFFIXES = tuple(f"_{name}" for name in _SENSITIVE_KEYS)


@dataclass(frozen=True)
class MigrationIssue:
    """One blocking, source-addressed migration problem."""

    code: str
    path: str
    message: str
    resolution: str

    def render(self) -> str:
        return (
            f"[{self.code}] {self.path}: {self.message} "
            f"Resolution: {self.resolution}"
        )


class ConfigMigrationError(RuntimeError):
    """Raised when an offline migration cannot prove a safe v0.4 result."""

    def __init__(self, issues: list[MigrationIssue] | tuple[MigrationIssue, ...]):
        self.issues = tuple(issues)
        if not self.issues:
            raise ValueError("ConfigMigrationError requires at least one issue")
        super().__init__(self._render())

    def _render(self) -> str:
        details = "\n".join(f"  - {issue.render()}" for issue in self.issues)
        return (
            "Configuration migration is blocked; the source was not changed and no "
            f"output was written:\n{details}"
        )


class MigrationContext:
    """Collect independent migration problems before failing as one report."""

    def __init__(self) -> None:
        self.issues: list[MigrationIssue] = []

    def add(
        self,
        code: str,
        path: str,
        message: str,
        resolution: str,
    ) -> None:
        self.issues.append(
            MigrationIssue(
                code=code,
                path=path,
                message=message,
                resolution=resolution,
            )
        )

    def raise_if_blocked(self) -> None:
        if self.issues:
            raise ConfigMigrationError(self.issues)


def as_mapping(value: Any, path: str, context: MigrationContext) -> dict[str, Any]:
    """Return a mapping or record one source-shape issue."""

    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    context.add(
        "invalid_shape",
        path,
        "expected a YAML mapping",
        "replace the value with a mapping and rerun the migration",
    )
    return {}


def as_list(value: Any, path: str, context: MigrationContext) -> list[Any]:
    """Return a list or record one source-shape issue."""

    if value is None:
        return []
    if isinstance(value, list):
        return value
    context.add(
        "invalid_shape",
        path,
        "expected a YAML list",
        "replace the value with a list and rerun the migration",
    )
    return []


def reject_unknown_fields(
    value: dict[str, Any],
    allowed: set[str] | frozenset[str],
    path: str,
    context: MigrationContext,
) -> None:
    """Reject fields that have no explicit translation rule."""

    for field_name in sorted(set(value) - set(allowed)):
        context.add(
            "unsupported_field",
            f"{path}.{field_name}",
            "v0.4 has no verified translation for this field",
            "remove it or express the behavior with the documented v0.4 contract",
        )


def environment_reference(value: Any) -> str | None:
    """Resolve an environment name or ${NAME} reference without reading it."""

    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if _ENV_NAME.fullmatch(stripped):
        return stripped
    match = _ENV_REFERENCE.fullmatch(stripped)
    return match.group(1) if match else None


def canonical_decimal(
    value: Any,
    path: str,
    context: MigrationContext,
    *,
    positive: bool,
) -> str | None:
    """Render a source number as the exact decimal string required by v0.4."""

    if isinstance(value, bool):
        parsed = None
    else:
        try:
            parsed = Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            parsed = None
    if (
        parsed is None
        or not parsed.is_finite()
        or parsed < 0
        or (positive and parsed == 0)
    ):
        expectation = (
            "expected a finite positive decimal"
            if positive
            else "expected a finite non-negative decimal"
        )
        context.add(
            "invalid_decimal",
            path,
            expectation,
            "replace the value with an ordinary decimal number",
        )
        return None
    rendered = format(parsed, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def scan_plaintext_secrets(value: Any, context: MigrationContext) -> None:
    """Reject secret-looking scalar values before any output is assembled."""

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for raw_key, child in node.items():
                key = str(raw_key)
                child_path = f"{path}.{key}" if path else key
                normalized = key.strip().lower().replace("-", "_")
                if _is_secret_value_key(normalized) and _contains_plaintext(child):
                    context.add(
                        "plaintext_secret",
                        child_path,
                        "plaintext secret material cannot be copied into v0.4 YAML",
                        "replace it with ${ENV_NAME}, an *_env field, or a file-backed secret reference",
                    )
                walk(child, child_path)
            return
        if isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, f"{path}[{index}]")

    walk(value, "")


def _is_secret_value_key(key: str) -> bool:
    if key.endswith(("_env", "_file", "_ref", "_name")):
        return False
    return key in _SENSITIVE_KEYS or key.endswith(_SENSITIVE_SUFFIXES)


def _contains_plaintext(value: Any) -> bool:
    if value in (None, "", [], {}):
        return False
    if isinstance(value, str):
        return _ENV_REFERENCE.fullmatch(value.strip()) is None
    return True
