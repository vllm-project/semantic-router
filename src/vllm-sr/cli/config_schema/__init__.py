"""Access the generated canonical Router configuration contract."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

SCHEMA_FILENAME = "router-config-v0.3.schema.json"


def schema_text() -> str:
    """Return the exact schema bundled with this CLI build."""

    return files(__package__).joinpath(SCHEMA_FILENAME).read_text(encoding="utf-8")


def schema_document() -> dict[str, Any]:
    """Return the bundled schema as a new mapping."""

    return json.loads(schema_text())


def routing_surface_catalog() -> dict[str, Any]:
    """Return generated signal, algorithm, plugin, and projection metadata."""

    extension = schema_document().get("x-vllm-sr")
    if not isinstance(extension, dict):
        raise RuntimeError("bundled Router config schema is missing x-vllm-sr")
    return extension


def surface_types(surface: str) -> tuple[str, ...]:
    """Return discriminator values from one generated routing catalog."""

    entries = routing_surface_catalog().get(surface, [])
    if not isinstance(entries, list):
        raise RuntimeError(f"invalid generated routing surface: {surface}")
    return tuple(
        value
        for entry in entries
        if isinstance(entry, dict) and isinstance((value := entry.get("type")), str)
    )
