"""Publishability checks for Router Model provenance manifests.

Manifests are committed and published alongside model cards, so they must not
carry credentials, raw dataset rows, or machine-specific infrastructure. The
checks run before schema validation so a rejected value is never echoed back in
a schema error message.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

MAX_STRING_LENGTH = 2048

# Field names that must never appear anywhere in a manifest.
FORBIDDEN_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "auth",
        "authorization",
        "aws_access_key_id",
        "aws_secret_access_key",
        "cookie",
        "credential",
        "credentials",
        "examples",
        "hf_token",
        "password",
        "private_key",
        "rows",  # only forbidden at leaf-string level; see _check_key
        "samples",
        "secret",
        "session",
        "token",
    }
)

# ``rows`` is a legitimate integer count in the dataset schema, so it is only
# rejected when it carries string content.
COUNT_LIKE_KEYS = frozenset({"rows"})

SECRET_VALUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{16,}\b")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]|github_pat)_[A-Za-z0-9_]{16,}\b")),
    ("AWS access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("bearer header", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}\b")),
    ("inline URL credentials", re.compile(r"://[^/\s:@]+:[^/\s@]+@")),
)

ENVIRONMENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("absolute POSIX path", re.compile(r"(?<![\w.])/(?:home|Users|root|mnt|tmp)/")),
    ("Windows path", re.compile(r"\b[A-Za-z]:\\")),
    (
        "private IPv4 address",
        re.compile(r"\b(?:10|127|192\.168)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    ),
    ("localhost endpoint", re.compile(r"(?i)\blocalhost:\d{2,5}\b")),
)


class RedactionError(ValueError):
    """Raised when a manifest carries content that must not be published."""


def assert_publishable(manifest: dict[str, Any], path: Path) -> None:
    """Fail if the manifest carries secrets, raw samples, or host-specific paths."""
    for location, key, value in _walk(manifest):
        _check_key(location, key, value, path)
        if isinstance(value, str):
            _check_value(location, value, path)


def _check_key(location: str, key: str | None, value: Any, path: Path) -> None:
    if key is None:
        return
    normalized = key.strip().lower()
    if normalized not in FORBIDDEN_KEYS:
        return
    if normalized in COUNT_LIKE_KEYS and not isinstance(value, str):
        return
    raise RedactionError(
        f"{path} {location} uses reserved field {key!r}; manifests must not carry "
        "credentials or raw dataset rows"
    )


def _check_value(location: str, value: str, path: Path) -> None:
    if len(value) > MAX_STRING_LENGTH:
        raise RedactionError(
            f"{path} {location} is {len(value)} characters; values above "
            f"{MAX_STRING_LENGTH} are treated as embedded sample data"
        )
    for label, pattern in SECRET_VALUE_PATTERNS:
        if pattern.search(value):
            raise RedactionError(f"{path} {location} looks like a {label}")
    for label, pattern in ENVIRONMENT_PATTERNS:
        if pattern.search(value):
            raise RedactionError(
                f"{path} {location} contains a {label}; manifests must stay "
                "portable across machines"
            )


def _walk(node: Any, location: str = "<root>") -> Iterator[tuple[str, str | None, Any]]:
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{location}.{key}" if location != "<root>" else str(key)
            yield child, str(key), value
            yield from _walk(value, child)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            child = f"{location}[{index}]"
            yield child, None, value
            yield from _walk(value, child)
