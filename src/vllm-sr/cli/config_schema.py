"""Validation against the generated Go canonical configuration schema."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from cli.config_contract import CANONICAL_VERSION

_SCHEMA_PATH = (
    Path(__file__).resolve().parent / "templates" / "canonical-config-schema.json"
)


@lru_cache(maxsize=1)
def load_canonical_schema() -> dict[str, Any]:
    with _SCHEMA_PATH.open(encoding="utf-8") as schema_file:
        schema = json.load(schema_file)
    supported = schema.get("supportedVersions")
    if supported != [CANONICAL_VERSION]:
        raise RuntimeError(
            "canonical config schema version does not match the CLI contract: "
            f"{supported!r}"
        )
    return schema


def reject_unknown_canonical_fields(data: Any) -> None:
    """Reject unknown fields using stable dotted paths and list indices."""
    errors: list[str] = []
    schema = load_canonical_schema()
    _collect_unknown_fields(
        data, schema["root"], schema.get("definitions", {}), "", errors
    )
    if errors:
        raise ValueError("unsupported config fields: " + ", ".join(errors))


def _collect_unknown_fields(
    value: Any,
    node: dict[str, Any],
    definitions: dict[str, Any],
    path: str,
    errors: list[str],
) -> None:
    if "ref" in node:
        node = definitions[node["ref"]]
    if node.get("opaque"):
        return
    node_type = node.get("type")
    if node_type == "object":
        _collect_object_fields(value, node, definitions, path, errors)
        return
    if node_type == "array":
        _collect_array_fields(value, node, definitions, path, errors)
        return
    if node_type == "map":
        _collect_map_fields(value, node, definitions, path, errors)


def _collect_object_fields(
    value: Any,
    node: dict[str, Any],
    definitions: dict[str, Any],
    path: str,
    errors: list[str],
) -> None:
    if not isinstance(value, dict):
        return
    fields = node.get("fields", {})
    for key in sorted(value):
        child_path = _join_path(path, str(key))
        child = fields.get(key)
        if not path and key == "setup":
            # setup is a typed CLI/dashboard bootstrap marker and is removed
            # before the Go router loads the config.
            continue
        if child is None:
            suggestion = _closest_field(str(key), fields)
            if suggestion:
                child_path += f' (did you mean "{suggestion}"?)'
            errors.append(child_path)
            continue
        _collect_unknown_fields(value[key], child, definitions, child_path, errors)


def _collect_array_fields(
    value: Any,
    node: dict[str, Any],
    definitions: dict[str, Any],
    path: str,
    errors: list[str],
) -> None:
    if not isinstance(value, list):
        return
    item_schema = node.get("items", {})
    for index, item in enumerate(value):
        _collect_unknown_fields(
            item, item_schema, definitions, f"{path}[{index}]", errors
        )


def _collect_map_fields(
    value: Any,
    node: dict[str, Any],
    definitions: dict[str, Any],
    path: str,
    errors: list[str],
) -> None:
    if not isinstance(value, dict):
        return
    value_schema = node.get("values", {})
    for key in sorted(value):
        child_path = _join_path(path, str(key))
        _collect_unknown_fields(
            value[key], value_schema, definitions, child_path, errors
        )


def _join_path(parent: str, child: str) -> str:
    if not parent:
        return child
    return f"{parent}.{child}"


def _closest_field(unknown: str, fields: dict[str, Any]) -> str | None:
    closest: str | None = None
    closest_distance = 4
    for field in sorted(fields):
        distance = _levenshtein(unknown, field)
        if distance < closest_distance:
            closest = field
            closest_distance = distance
    return closest


def _levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]
