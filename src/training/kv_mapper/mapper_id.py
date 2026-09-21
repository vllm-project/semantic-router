"""Mapper id strings for router config (pins served artifact, not routing alias)."""

from __future__ import annotations

import re


def _revision_token(revision: str, max_len: int = 12) -> str:
    """HF revision or commit id, safe for use in a config string."""
    token = re.sub(r"[^a-zA-Z0-9._-]", "-", revision.strip())
    if not token:
        raise ValueError("revision must be non-empty")
    if len(token) > max_len:
        return token[-max_len:]
    return token


def make_mapper_id(
    *,
    pair_slug: str,
    variant: str,
    precision: str,
    source_revision: str,
    target_revision: str,
    source_tp: int,
    target_tp: int,
    n_kv_heads: int,
    bundle_version: int = 1,
) -> str:
    """Build a config-facing mapper id.

    Pins the Hugging Face weight revisions the mapper was fitted on (not a routing
    alias), plus dtype and KV head count. ``bundle_version`` bumps when the same
    revisions are re-fitted with a new recipe; it is not a model revision.
    """
    prec = precision.lower().replace("float", "fp")
    src = _revision_token(source_revision)
    tgt = _revision_token(target_revision)
    tp = (
        f"tp{source_tp}"
        if source_tp == target_tp
        else f"tp{source_tp}to{target_tp}"
    )
    return (
        f"{pair_slug}-{variant}-{prec}-{tp}-h{n_kv_heads}"
        f"-s{src}-t{tgt}-b{bundle_version}"
    )
