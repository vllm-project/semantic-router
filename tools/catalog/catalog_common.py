"""Shared strict primitives for the model catalog compiler."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

VERSIONED_ID = re.compile(
    r"^[a-z0-9][a-z0-9._-]*(?:/[a-z0-9][a-z0-9._-]*)+@\d+\.\d+\.\d+$"
)
PROTOCOL_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*(?:/[a-z0-9][a-z0-9._-]*)+@\d+$")
SLUG = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
MODEL_ID = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:+-]*(?:/[A-Za-z0-9][A-Za-z0-9._:+-]*)+$"
)
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
PAIR_LENGTH = 2
MIN_PIECEWISE_POINTS = 2


class CatalogBuildError(ValueError):
    """The authored catalog cannot produce a trustworthy registry."""


def mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CatalogBuildError(f"{path} must be a mapping")
    return value


def sequence(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise CatalogBuildError(f"{path} must be a list")
    return value


def nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogBuildError(f"{path} must be a non-empty string")
    return value.strip()


def reject_unknown(
    mapping_value: dict[str, Any], allowed: Iterable[str], path: str
) -> None:
    unknown = sorted(set(mapping_value) - set(allowed))
    if unknown:
        raise CatalogBuildError(f"{path} has unknown fields: {', '.join(unknown)}")


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )
