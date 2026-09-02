"""Canonical serialization and content digests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import date, datetime
from enum import Enum
from typing import Any, BinaryIO, NoReturn, TextIO

from pydantic import BaseModel


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object repeats key {key!r}")
        result[key] = value
    return result


def _reject_non_finite_json_constant(value: str) -> NoReturn:
    raise ValueError(f"JSON number must be finite, got {value}")


def strict_json_loads(data: str | bytes | bytearray) -> Any:
    """Decode contract JSON without duplicate keys or non-finite numbers."""

    return json.loads(
        data,
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_non_finite_json_constant,
    )


def strict_json_load(handle: TextIO | BinaryIO) -> Any:
    return strict_json_loads(handle.read())


def json_value(value: Any, *, exclude_none: bool = True) -> Any:
    """Convert models into JSON-native values before canonical encoding."""

    if isinstance(value, BaseModel):
        return json_value(
            value.model_dump(mode="json", exclude_none=exclude_none),
            exclude_none=exclude_none,
        )
    if isinstance(value, Mapping):
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
