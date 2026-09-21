"""Canonical evidence hashes and cross-client numeric protocol identities."""

from __future__ import annotations

import hashlib
import json

_CANONICAL_ENCODER = json.JSONEncoder(
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
)


def canonical(value):
    return _CANONICAL_ENCODER.encode(value)


def digest(value):
    checksum = hashlib.sha256()
    for chunk in _CANONICAL_ENCODER.iterencode(value):
        checksum.update(chunk.encode())
    return checksum.hexdigest()


def _protocol_numbers(item):
    if isinstance(item, float) and item.is_integer():
        return int(item)
    if isinstance(item, dict):
        return {key: _protocol_numbers(child) for key, child in item.items()}
    if isinstance(item, list):
        return [_protocol_numbers(child) for child in item]
    return item


def protocol_canonical(value):
    """Compare control values across JSON clients without rewriting evidence.

    JSON has one numeric type: browsers serialize 1.0 as 1 and -0.0 as 0.
    Preserve booleans, strings, array order and fractional values, and retain
    canonical()'s rejection of non-finite numbers. Raw case/dataset digests
    deliberately keep their separate, exact-content contract.
    """
    return canonical(_protocol_numbers(value))


def plan_digest(manifest):
    """Hash a reviewed plan's numeric semantics, excluding its own receipt."""
    content = {key: value for key, value in manifest.items() if key != "plan_sha256"}
    return digest(_protocol_numbers(content))
