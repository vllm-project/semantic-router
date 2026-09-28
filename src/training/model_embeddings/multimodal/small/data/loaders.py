"""Cached tensor collation, DataLoader construction, and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .cached import CachedAudioDataset, CachedTensorDataset
from .shards import discover_complete_shards, validate_shard


def _stack(batch: list[dict[str, Any]], feature_key: str) -> dict[str, Any]:
    return {
        feature_key: torch.stack([item[feature_key] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
    }


def cached_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return _stack(batch, "pixel_values")


def audio_cached_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return _stack(batch, "input_features")


def create_cached_dataloader(
    cache_dir: str,
    batch_size: int,
    num_workers: int = 8,
    max_samples: int | None = None,
    shuffle: bool = True,
    pin_memory: bool = True,
    dynamic_discovery: bool = True,
    discovery_interval: int = 60,
) -> DataLoader:
    dataset = CachedTensorDataset(
        cache_dir=cache_dir,
        max_samples=max_samples,
        shuffle_shards=shuffle,
        dynamic_discovery=dynamic_discovery,
        discovery_interval=discovery_interval,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=cached_collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def create_audio_cached_dataloader(
    cache_dir: str,
    batch_size: int,
    num_workers: int = 8,
    max_samples: int | None = None,
    shuffle: bool = True,
    pin_memory: bool = True,
    dynamic_discovery: bool = True,
    discovery_interval: int = 60,
) -> DataLoader:
    dataset = CachedAudioDataset(
        cache_dir=cache_dir,
        max_samples=max_samples,
        shuffle_shards=shuffle,
        dynamic_discovery=dynamic_discovery,
        discovery_interval=discovery_interval,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=audio_cached_collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def validate_all_shards(
    cache_dir: str,
    feature_key: str = "pixel_values",
    verbose: bool = True,
) -> tuple[list[str], list[str]]:
    """Validate every completed shard before a distributed run."""
    root = Path(cache_dir)
    if not root.is_dir():
        return [], [f"Directory does not exist: {root}"]
    paths = discover_complete_shards(root)
    if not paths:
        return [], ["No complete shard files found"]

    valid: list[str] = []
    invalid: list[str] = []
    for path in paths:
        ok, detail = validate_shard(path, feature_key)
        if ok:
            valid.append(path.name)
        else:
            invalid.append(f"{path.name}: {detail}")
        if verbose:
            print(f"{path.name}: {'OK' if ok else 'FAIL'} ({detail})")
    return valid, invalid


def detect_cached_dataset(path: str | None) -> str | None:
    """Return an explicitly configured usable cache path, if present."""
    if not path:
        return None
    return path if discover_complete_shards(path) else None
