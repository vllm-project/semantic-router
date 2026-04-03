"""Helpers for selecting the local runtime topology mode."""

from __future__ import annotations

import os

from cli.consts import (
    DEFAULT_RUNTIME_TOPOLOGY,
    RUNTIME_TOPOLOGY_ENV,
    RUNTIME_TOPOLOGY_SPLIT,
)

VALID_RUNTIME_TOPOLOGIES = (RUNTIME_TOPOLOGY_SPLIT,)


def resolve_runtime_topology(topology: str | None = None) -> str:
    raw_value = topology if topology is not None else os.getenv(RUNTIME_TOPOLOGY_ENV)
    if raw_value is None:
        return DEFAULT_RUNTIME_TOPOLOGY

    normalized = str(raw_value).strip().lower()
    if not normalized:
        return DEFAULT_RUNTIME_TOPOLOGY
    if normalized not in VALID_RUNTIME_TOPOLOGIES:
        raise ValueError(
            f"Unsupported runtime topology '{raw_value}'. "
            f"{RUNTIME_TOPOLOGY_ENV} may be unset or set to "
            f"'{RUNTIME_TOPOLOGY_SPLIT}'."
        )
    return RUNTIME_TOPOLOGY_SPLIT


def split_runtime_enabled(topology: str | None = None) -> bool:
    resolve_runtime_topology(topology)
    return True
