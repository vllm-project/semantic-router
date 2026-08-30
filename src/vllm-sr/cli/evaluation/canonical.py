"""Canonical serialization and content digests."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel


def json_value(value: Any, *, exclude_none: bool = True) -> Any:
    """Convert models into JSON-native values before canonical encoding."""

    if isinstance(value, BaseModel):
        return json_value(
            value.model_dump(mode="json", exclude_none=exclude_none),
            exclude_none=exclude_none,
        )
    if isinstance(value, dict):
        return {
            str(key): json_value(item, exclude_none=exclude_none)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_value(item, exclude_none=exclude_none) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return json_value(value.value, exclude_none=exclude_none)
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        json_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            json_value(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def sha256_digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def digest_value(value: Any) -> str:
    return sha256_digest(canonical_json_bytes(value))
