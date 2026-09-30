"""Mapper id strings for router config (pins served artifact, not routing alias)."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

_PRECISION_ALIASES = {
    "fp16": "fp16",
    "float16": "fp16",
    "half": "fp16",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp32": "fp32",
    "float32": "fp32",
}


def normalize_precision(precision: str) -> str:
    try:
        return _PRECISION_ALIASES[precision.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported mapper precision: {precision!r}") from exc


_COMMIT_SHA = re.compile(r"[0-9a-f]{40}\Z")


def require_weight_commit(revision: str) -> str:
    """Return a canonical immutable Hugging Face weight commit."""
    commit = revision.strip().lower()
    if not _COMMIT_SHA.fullmatch(commit):
        raise ValueError("model weight revision must be a full 40-character commit SHA")
    return commit


def resolve_weight_commit(
    model_id: str, revision: str, model_info: Callable[..., Any]
) -> str:
    """Resolve a branch or tag once, before loading or recording model weights."""
    info = model_info(model_id, revision=revision)
    return require_weight_commit(info.sha)


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
    prec = normalize_precision(precision)
    src = require_weight_commit(source_revision)
    tgt = require_weight_commit(target_revision)
    tp = f"tp{source_tp}" if source_tp == target_tp else f"tp{source_tp}to{target_tp}"
    return (
        f"{pair_slug}-{variant}-{prec}-{tp}-h{n_kv_heads}"
        f"-s{src}-t{tgt}-b{bundle_version}"
    )
