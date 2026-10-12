"""`vllm-sr config migrate`: retire the Looper loopback endpoint."""

from __future__ import annotations

from typing import Any

from cli.config_migration_notes import MigrationNotes

LOOPER_ENDPOINT_PATH = "global.integrations.looper.endpoint"


def drop_looper_endpoint(canonical: dict[str, Any], notes: MigrationNotes) -> None:
    """Remove the Looper endpoint, which the router no longer reads.

    The router makes a Looper's model calls in process, through each model's
    providers.models[].backend_refs, so the loopback address has no use.
    """
    integrations = _mapping(canonical.get("global"), "integrations")
    looper = _mapping(integrations, "looper")
    if looper is None or "endpoint" not in looper:
        return
    del looper["endpoint"]
    notes.changed(
        LOOPER_ENDPOINT_PATH,
        "removed; the router makes Looper calls in process, through each "
        "model's providers.models[].backend_refs",
    )
    if not looper:
        del integrations["looper"]
    if not integrations:
        del canonical["global"]["integrations"]


def _mapping(parent: Any, key: str) -> dict[str, Any] | None:
    if not isinstance(parent, dict):
        return None
    value = parent.get(key)
    return value if isinstance(value, dict) else None
