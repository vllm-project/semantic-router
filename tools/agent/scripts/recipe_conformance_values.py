"""Strict value helpers shared by recipe conformance modules."""

from typing import Any


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def sequence(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def reject_fields(value: dict[str, Any], path: str, fields: set[str]) -> None:
    rejected = sorted(value.keys() & fields)
    if rejected:
        raise ValueError(
            f"{path} uses unsupported generated fields: {', '.join(rejected)}"
        )
