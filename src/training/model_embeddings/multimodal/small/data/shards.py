"""Shard discovery, loading, and validation primitives."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch

try:
    from safetensors.torch import load_file as safetensors_load
except ImportError:  # pragma: no cover - exercised only without the optional format
    safetensors_load = None


MIN_SHARD_SIZE_BYTES = 1024 * 1024
SHARD_SUFFIXES = (".pt", ".safetensors")


def load_shard(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    """Load a SafeTensors or legacy PyTorch tensor shard."""
    shard_path = str(path)
    if shard_path.endswith(".safetensors"):
        if safetensors_load is None:
            raise RuntimeError("Install safetensors to load .safetensors shards")
        return safetensors_load(shard_path, device=device)
    return torch.load(shard_path, map_location=device, weights_only=False)


def is_complete_shard(path: str | Path, settle_seconds: float = 2.0) -> bool:
    """Return whether a shard is large enough and no longer being written."""
    shard_path = Path(path)
    try:
        stat = shard_path.stat()
    except OSError:
        return False
    return (
        stat.st_size >= MIN_SHARD_SIZE_BYTES
        and time.time() - stat.st_mtime >= settle_seconds
    )


def discover_complete_shards(cache_dir: str | Path) -> list[Path]:
    """List complete shards in deterministic filename order."""
    root = Path(os.path.expandvars(str(cache_dir))).expanduser()
    if not root.is_dir():
        return []
    return [
        path
        for path in sorted(root.iterdir())
        if path.name.startswith("shard_")
        and path.suffix in SHARD_SUFFIXES
        and is_complete_shard(path)
    ]


def validate_shard(path: str | Path, feature_key: str) -> tuple[bool, str]:
    """Validate one shard's required keys, lengths, and finite feature values."""
    shard_path = Path(path)
    if not is_complete_shard(shard_path):
        return False, "file is incomplete or smaller than 1 MiB"
    try:
        data = load_shard(shard_path)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    required = {feature_key, "input_ids", "attention_mask"}
    missing = required.difference(data)
    if missing:
        return False, f"missing keys: {sorted(missing)}"

    row_count = len(data["input_ids"])
    if row_count == 0:
        return False, "empty shard"
    if any(len(data[key]) != row_count for key in required):
        return False, "tensor row counts do not match"
    features = data[feature_key]
    if torch.isnan(features).any() or torch.isinf(features).any():
        return False, f"{feature_key} contains NaN or Inf"
    return True, f"{row_count} samples"
