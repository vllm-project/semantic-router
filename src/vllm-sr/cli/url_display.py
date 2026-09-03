"""Display-safe URL handling for human-facing CLI output."""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_SENSITIVE_PARAMETER_KEYS = frozenset(
    {
        "accesstoken",
        "apikey",
        "auth",
        "authorization",
        "credential",
        "credentials",
        "password",
        "secret",
        "sig",
        "signature",
        "token",
    }
)
_SENSITIVE_PARAMETER_KEY_SUFFIXES = (
    "credential",
    "key",
    "password",
    "secret",
    "signature",
    "token",
)


def redact_url(value: str) -> str:
    """Return a display-safe URL without embedded credentials."""
    if value == "-":
        return value

    try:
        parsed = urlsplit(value)
    except ValueError:
        return "[invalid URL redacted]"

    # A malformed network URL can leave userinfo in ``path`` rather than
    # ``netloc``. Never echo that ambiguous value back to the terminal.
    if not parsed.netloc and "@" in parsed.path:
        return "[invalid URL redacted]"

    netloc = parsed.netloc
    if "@" in netloc:
        netloc = "***@" + netloc.rsplit("@", 1)[1]

    redacted_query = _redact_url_parameters(parsed.query)
    redacted_fragment = _redact_url_parameters(
        parsed.fragment,
        preserve_when_clean=True,
    )
    return urlunsplit(
        (parsed.scheme, netloc, parsed.path, redacted_query, redacted_fragment)
    )


def _redact_url_parameters(value: str, *, preserve_when_clean: bool = False) -> str:
    items = parse_qsl(value, keep_blank_values=True)
    if not items:
        return value
    contains_sensitive_value = any(_is_sensitive_parameter_key(key) for key, _ in items)
    if preserve_when_clean and not contains_sensitive_value:
        return value
    redacted = [
        (key, "***" if _is_sensitive_parameter_key(key) else item_value)
        for key, item_value in items
    ]
    return urlencode(redacted, doseq=True, safe="*")


def _is_sensitive_parameter_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
    return normalized in _SENSITIVE_PARAMETER_KEYS or normalized.endswith(
        _SENSITIVE_PARAMETER_KEY_SUFFIXES
    )
