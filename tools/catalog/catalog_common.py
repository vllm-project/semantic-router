"""Shared strict primitives for the model catalog compiler."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlsplit

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
ASCII_SPACE = 0x20
ASCII_DELETE = 0x7F
MAX_TCP_PORT = 65535


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


def validate_https_url(
    value: Any,
    path: str,
    *,
    allow_fragment: bool = True,
) -> str:
    """Validate a public HTTPS URL before it enters generated consumers."""

    kind = "document" if allow_fragment else "transport"
    if not isinstance(value, str) or not value:
        raise CatalogBuildError(f"{path} must be a non-empty HTTPS {kind} URL")
    if any(
        ord(character) <= ASCII_SPACE or ord(character) == ASCII_DELETE
        for character in value
    ):
        raise CatalogBuildError(
            f"{path} HTTPS {kind} URL contains whitespace or control characters"
        )
    try:
        parsed = urlsplit(value)
        port = parsed.port
        hostname = parsed.hostname
    except ValueError as error:
        raise CatalogBuildError(f"{path} is not a valid HTTPS {kind} URL") from error
    host_port = parsed.netloc.rsplit("@", 1)[-1]
    if (
        parsed.scheme != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or (not allow_fragment and bool(parsed.fragment))
        or (port is None and host_port.endswith(":"))
        or (port is not None and not 1 <= port <= MAX_TCP_PORT)
    ):
        raise CatalogBuildError(
            f"{path} must be an HTTPS {kind} URL with a valid host and port, "
            "without userinfo, whitespace, or controls"
        )
    return value
