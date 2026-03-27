"""Helpers for selecting the local runtime topology mode."""

from __future__ import annotations

import os

from cli.consts import (
    DEFAULT_RUNTIME_TOPOLOGY,
    RUNTIME_TOPOLOGY_ENV,
    RUNTIME_TOPOLOGY_LEGACY,
    RUNTIME_TOPOLOGY_SPLIT,
)

VALID_RUNTIME_TOPOLOGIES = (
    RUNTIME_TOPOLOGY_LEGACY,
    RUNTIME_TOPOLOGY_SPLIT,
)


def resolve_runtime_topology(topology: str | None = None) -> str:
    raw_value = topology if topology is not None else os.getenv(RUNTIME_TOPOLOGY_ENV)
    if raw_value is None:
        return DEFAULT_RUNTIME_TOPOLOGY

    normalized = str(raw_value).strip().lower()
    if not normalized:
        return DEFAULT_RUNTIME_TOPOLOGY
    if normalized not in VALID_RUNTIME_TOPOLOGIES:
        raise ValueError(
            f"Invalid runtime topology '{raw_value}'. "
            f"Must be one of: {', '.join(VALID_RUNTIME_TOPOLOGIES)}"
        )
    return normalized


def split_runtime_enabled(topology: str | None = None) -> bool:
    return resolve_runtime_topology(topology) == RUNTIME_TOPOLOGY_SPLIT
