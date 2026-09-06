"""Primitive validators shared by strict evaluation contracts."""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
import uuid
from urllib.parse import urlsplit

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ENV_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_MAX_RUN_NAME_BYTES = 200
_MAX_RUN_DESCRIPTION_BYTES = 4000
_MAX_SUITE_REVISION_LENGTH = 160
_MAX_INLINE_IMAGE_BYTES = 2 * 1024 * 1024
_INLINE_IMAGE_PREFIXES = (
    "data:image/png;base64,",
    "data:image/jpeg;base64,",
    "data:image/webp;base64,",
    "data:image/gif;base64,",
)


def validate_portable_id(value: str) -> str:
    if not is_portable_id(value):
        raise ValueError(
            "must be a portable identifier (letters, digits, '.', '_' or '-')"
        )
    return value


def is_portable_id(value: str) -> bool:
    return _ID_RE.fullmatch(value) is not None


def derived_portable_id(namespace: str, *parts: str) -> str:
    """Build an unambiguous content-addressed identity inside the wire bound."""

    validate_portable_id(namespace)
    if not parts:
        raise ValueError("a derived portable identifier requires identity parts")
    for part in parts:
        validate_portable_id(part)
    digest = hashlib.sha256("\x00".join((namespace, *parts)).encode()).hexdigest()
    compact = f"{namespace}-{digest}"
    return compact if is_portable_id(compact) else f"derived-{digest}"


def is_subject_target_id(value: str, subject_id: str) -> bool:
    """Accept a standalone subject or its explicit deployment-scoped target."""
    deployment_suffix = f"--{subject_id}"
    return value == subject_id or (
        value.endswith(deployment_suffix) and len(value) > len(deployment_suffix)
    )


def validate_canonical_uuid(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise ValueError("must be a canonical UUID") from exc
    if str(parsed) != value:
        raise ValueError("must be a canonical UUID")
    return value


def validate_run_name(value: str) -> str:
    if (
        not value
        or value.strip() != value
        or len(value.encode("utf-8")) > _MAX_RUN_NAME_BYTES
    ):
        raise ValueError("run name must be 1-200 trimmed UTF-8 bytes")
    return value


def validate_run_description(value: str) -> str:
    if (
        value.strip() != value
        or len(value.encode("utf-8")) > _MAX_RUN_DESCRIPTION_BYTES
    ):
        raise ValueError("run description must be trimmed and at most 4000 UTF-8 bytes")
    return value


def validate_http_origin(value: str, *, label: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{label} must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(f"{label} cannot contain credentials, query, or fragment")
    if parsed.path:
        raise ValueError(
            f"{label} must be a canonical origin without an API path or trailing slash"
        )
    if parsed.geturl() != value:
        raise ValueError(f"{label} must use its canonical serialized form")
    return value


def validate_secret_env(value: str) -> str:
    if not _ENV_RE.fullmatch(value):
        raise ValueError("secret env must be an uppercase environment variable name")
    return value


def validate_inline_image_url(value: str) -> str:
    encoded = next(
        (
            value.removeprefix(prefix)
            for prefix in _INLINE_IMAGE_PREFIXES
            if value.startswith(prefix)
        ),
        None,
    )
    if not encoded or len(encoded) > 4 * ((_MAX_INLINE_IMAGE_BYTES + 2) // 3):
        raise ValueError("live evaluation images must be bounded inline data URIs")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("live evaluation image data is invalid") from exc
    if len(decoded) > _MAX_INLINE_IMAGE_BYTES:
        raise ValueError("live evaluation image exceeds its byte budget")
    return value


def is_valid_suite_revision(value: str) -> bool:
    return bool(value.strip()) and len(value) <= _MAX_SUITE_REVISION_LENGTH
