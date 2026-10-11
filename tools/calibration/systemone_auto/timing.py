"""Validate native runtime identity before using its reported elapsed compute."""

from __future__ import annotations

import re

from .artifacts import finite

COST_METRICS = ("server_compute_ms", "client_elapsed_ms")
RUNTIME_FIELDS = (
    "revision",
    "model_sha256",
    "engine",
    "profile",
    "numerics",
    "accelerator",
)


def native_timing(raw: dict, target: dict) -> tuple[float, dict]:
    meta = raw.get("meta")
    if not isinstance(meta, dict) or any(
        not meta.get(field) for field in RUNTIME_FIELDS
    ):
        raise ValueError(
            "native compute costs require complete runtime identity metadata"
        )
    if meta["revision"] != target["revision"] or raw.get("model") != target["model_id"]:
        raise ValueError(
            "native response model/revision differs from the pinned target"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", meta["model_sha256"]):
        raise ValueError("native runtime must report its model content SHA256")
    cost = finite(meta.get("compute_ms"))
    if cost <= 0:
        raise ValueError("native compute elapsed time must be positive")
    return cost, {
        "model_id": raw["model"],
        **{field: meta[field] for field in RUNTIME_FIELDS},
    }
