"""Dataset and DataLoader construction for the compact training workflow."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from .data import (
    CachedAudioDataset,
    CachedTensorDataset,
    SequentialShardDataset,
    audio_cached_collate_fn,
    cached_collate_fn,
    validate_all_shards,
)
from .distributed import DistributedContext, barrier
from .raw_data import COCOCaptionsDataset, LLaVACC3MDataset, raw_collate_fn


@dataclass
class DataBundle:
    train: Any
    validation: Any
    collate: Any
    sequential: bool
    cached: bool


def _cached_dataset(config: dict[str, Any], validation: bool = False) -> Any:
    key = "validation_cache" if validation else "train_cache"
    dataset_type = (
        CachedAudioDataset
        if config["feature_key"] == "input_features"
        else CachedTensorDataset
    )
    return dataset_type(
        config[key],
        shuffle_shards=not validation,
        dynamic_discovery=bool(config.get("dynamic_discovery", False)),
    )


def _raw_dataset(config: dict[str, Any]) -> Any:
    kind = config["kind"]
    if kind == "llava_cc3m":
        return LLaVACC3MDataset(
            config["data_dir"],
            image_size=int(config.get("image_size", 256)),
            max_samples=config.get("max_samples"),
        )
    if kind == "coco":
        return COCOCaptionsDataset(
            config["annotations_file"],
            config["images_dir"],
            image_size=int(config.get("image_size", 256)),
            max_samples=config.get("max_samples"),
        )
    raise ValueError(f"Unsupported raw dataset kind: {kind}")


def create_data_bundle(
    data_config: dict[str, Any],
    context: DistributedContext,
) -> DataBundle:
    """Build cached or raw train/validation datasets."""
    mode = data_config["mode"]
    if mode == "raw":
        return DataBundle(
            train=_raw_dataset(data_config["train"]),
            validation=_raw_dataset(data_config["validation"]),
            collate=raw_collate_fn,
            sequential=False,
            cached=False,
        )
    if mode != "cached":
        raise ValueError(f"Unsupported data mode: {mode}")

    feature_key = data_config.get("feature_key", "pixel_values")
    if data_config.get("validate_shards", True) and context.is_main:
        for key in ("train_cache", "validation_cache"):
            valid, invalid = validate_all_shards(
                data_config[key],
                feature_key=feature_key,
                verbose=True,
            )
            if invalid:
                raise RuntimeError(f"Invalid {key} shards: {invalid[:10]}")
            if not valid:
                raise RuntimeError(f"No valid shards found in {data_config[key]}")
    barrier(context)

    collate = (
        audio_cached_collate_fn
        if feature_key == "input_features"
        else cached_collate_fn
    )
    sequential = bool(data_config.get("sequential_shards", True))
    if sequential:
        train = SequentialShardDataset(
            data_config["train_cache"],
            dynamic_discovery=bool(data_config.get("dynamic_discovery", False)),
            rank=context.rank,
            world_size=context.world_size,
            feature_key=feature_key,
        )
    else:
        train = _cached_dataset(data_config)
    return DataBundle(
        train=train,
        validation=_cached_dataset(data_config, validation=True),
        collate=collate,
        sequential=sequential,
        cached=True,
    )


def create_loader(
    dataset: Any,
    collate: Any,
    data_config: dict[str, Any],
    context: DistributedContext,
    *,
    batch_size: int,
    training: bool,
    sequential: bool = False,
) -> DataLoader:
    sampler = None
    shuffle = training and not sequential
    if context.distributed and not sequential:
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=training,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(data_config.get("num_workers", 0)),
        collate_fn=collate,
        pin_memory=bool(data_config.get("pin_memory", True)),
        drop_last=training,
        persistent_workers=int(data_config.get("num_workers", 0)) > 0,
    )


def synchronized_batch_count(local_count: int, context: DistributedContext) -> int:
    """Find the largest per-rank shard batch count to prevent DDP divergence."""
    if not context.distributed:
        return local_count
    count = torch.tensor([local_count], device=context.device, dtype=torch.int64)
    dist.all_reduce(count, op=dist.ReduceOp.MAX)
    return int(count.item())


def estimate_optimizer_steps(
    bundle: DataBundle,
    batch_size: int,
    epochs: int,
    grad_accum: int,
    world_size: int,
) -> int:
    samples = (
        bundle.train.total_samples_estimate if bundle.sequential else len(bundle.train)
    )
    batches = math.ceil(samples / max(1, batch_size * world_size))
    return max(1, epochs * math.ceil(batches / grad_accum))
