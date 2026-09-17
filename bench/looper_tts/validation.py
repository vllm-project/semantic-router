"""Small strict JSON validators shared by experiment and evidence contracts."""

import hashlib
import json
import math
from pathlib import Path

from . import SCHEMA_VERSION

SHA256_HEX_LENGTH = 64


class ContractError(ValueError):
    """An input violates the versioned benchmark contract."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


def fields(value, names, where):
    require(isinstance(value, dict), f"{where}: expected an object")
    require(set(value) == set(names.split()), f"{where}: expected fields {names}")


def string(value, where):
    require(isinstance(value, str) and bool(value.strip()), f"{where}: empty string")


def number(value, where, minimum=0, integer=False, nullable=False):
    if value is None and nullable:
        return
    types = (int,) if integer else (int, float)
    require(
        type(value) in types
        and (type(value) is int or math.isfinite(value))
        and value >= minimum,
        f"{where}: expected finite {'integer' if integer else 'number'} >= {minimum}",
    )


def sequence(value, where, nonempty=True):
    require(isinstance(value, list), f"{where}: expected a list")
    require(not nonempty or bool(value), f"{where}: must not be empty")


def strings(value, where, nonempty=True):
    sequence(value, where, nonempty)
    for entry in value:
        string(entry, where)
    require(len(value) == len(set(value)), f"{where}: duplicate values")


def indexed(value, where):
    sequence(value, where)
    result = {}
    for entry in value:
        require(isinstance(entry, dict) and "id" in entry, f"{where}: missing id")
        string(entry["id"], where)
        require(entry["id"] not in result, f"{where}: duplicate id {entry['id']}")
        result[entry["id"]] = entry
    return result


def version(value):
    require(value == SCHEMA_VERSION, "unsupported schema_version")


def digest(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256(value, where):
    require(
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(c in "0123456789abcdef" for c in value),
        f"{where}: expected lowercase SHA-256",
    )


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object
    )


def write_json(path, value, exclusive=False):
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as handle:
        handle.write(encoded)
