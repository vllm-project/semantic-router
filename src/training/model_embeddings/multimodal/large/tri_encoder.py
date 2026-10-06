"""Data, model, checkpoint, and evaluation components for the tri-encoder path."""

import json
import math
import os
import random
from typing import Any

import torch
from accelerate import Accelerator
from torch.utils.data import BatchSampler, DataLoader
from tqdm.auto import tqdm

from .data import CachedShardDataset, SequentialShardDataset
from .model import MultiModalSentenceEmbedder, multiple_negatives_ranking_loss

MAX_MODALITY_PATTERN_SIZE = 3
MIN_MODALITY_PATTERN_SIZE = 2


def summarize_cached_dataset(
    dataset: CachedShardDataset, sample_size: int = 256
) -> dict[str, Any]:
    observed_modalities = set()
    negatives_present = 0
    negatives_missing = 0
    rows_to_sample = min(len(dataset), sample_size)

    for idx in range(rows_to_sample):
        record = dataset[idx]
        observed_modalities.add(record.query.modality)
        observed_modalities.add(record.positive.modality)
        if record.negative is None:
            negatives_missing += 1
        else:
            negatives_present += 1
            observed_modalities.add(record.negative.modality)

    return {
        "num_rows": len(dataset),
        "modalities": sorted(observed_modalities),
        "num_negatives_present": negatives_present,
        "num_negatives_missing": negatives_missing,
        "has_uniform_negatives": negatives_present == 0 or negatives_missing == 0,
        "sampled_rows": rows_to_sample,
    }


def load_datacenter_tri_encoder_datasets(cfg: dict[str, Any]):
    data_cfg = cfg["data"]
    training_cfg = cfg.get("training", {})
    summary_dataset = CachedShardDataset(data_cfg["cache_dir"])
    train_info = summarize_cached_dataset(summary_dataset)

    if bool(training_cfg.get("sequential_shard_loading", False)):
        rank = int(
            os.environ.get("ACCELERATE_PROCESS_INDEX") or os.environ.get("RANK") or 0
        )
        world_size = int(
            os.environ.get("WORLD_SIZE")
            or os.environ.get("ACCELERATE_NUM_PROCESSES")
            or 1
        )
        train_dataset = SequentialShardDataset(
            data_cfg["cache_dir"],
            shuffle=bool(training_cfg.get("shuffle", True)),
            rank=rank,
            world_size=world_size,
            shard_cache_limit=int(training_cfg.get("shard_cache_limit", 4)),
            prefetch_shards=int(training_cfg.get("shard_prefetch", 2)),
        )
    else:
        train_dataset = CachedShardDataset(
            data_cfg["cache_dir"],
            shard_cache_limit=int(training_cfg.get("shard_cache_limit", 4)),
            prefetch_shards=int(training_cfg.get("shard_prefetch", 2)),
        )

    validation_cfg = cfg.get("validation", {})
    eval_dataset = None
    eval_info = None
    if validation_cfg.get("cache_dir"):
        eval_dataset = CachedShardDataset(
            validation_cfg["cache_dir"],
            shard_cache_limit=int(validation_cfg.get("shard_cache_limit", 2)),
            prefetch_shards=int(validation_cfg.get("shard_prefetch", 1)),
        )
        eval_info = summarize_cached_dataset(eval_dataset)

    return train_dataset, train_info, eval_dataset, eval_info


def build_datacenter_tri_encoder_model(
    cfg: dict[str, Any],
) -> MultiModalSentenceEmbedder:
    model_cfg = cfg["model"]
    return MultiModalSentenceEmbedder(
        text_encoder_name=model_cfg["text_encoder_name"],
        image_encoder_name=model_cfg["image_encoder_name"],
        audio_encoder_name=model_cfg["audio_encoder_name"],
        embedding_dim=int(
            model_cfg.get("embedding_dim", model_cfg.get("output_dim", 768))
        ),
        max_text_length=int(model_cfg.get("max_text_length", 32768)),
    )


def _encode_items_for_training(model: torch.nn.Module, items):
    encode_owner = model.module if hasattr(model, "module") else model
    return encode_owner.encode_items(items)


def _encode_query_positive_batch(
    model: torch.nn.Module, query_items, positive_items
) -> tuple[torch.Tensor, torch.Tensor]:
    query_count = len(query_items)
    combined_embeddings = _encode_items_for_training(
        model, list(query_items) + list(positive_items)
    )
    return combined_embeddings[:query_count], combined_embeddings[query_count:]


def _build_cached_loader_kwargs(
    num_workers: int, prefetch_factor: int
) -> dict[str, Any]:
    loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return loader_kwargs


class InterleavedModalityBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: CachedShardDataset,
        batch_size: int,
        drop_last: bool,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.pattern = self._detect_query_modality_pattern()
        self.period = len(self.pattern)
        self.modality_offsets = {
            modality: offset for offset, modality in enumerate(self.pattern)
        }
        self.modality_counts = {
            modality: self._count_indices_for_offset(offset)
            for modality, offset in self.modality_offsets.items()
        }

    def _count_indices_for_offset(self, offset: int) -> int:
        if offset >= len(self.dataset):
            return 0
        return ((len(self.dataset) - 1 - offset) // self.period) + 1

    def _detect_query_modality_pattern(self) -> list[str]:
        sample_rows = min(len(self.dataset), 18)
        if sample_rows == 0:
            raise ValueError(
                "Cannot build a modality batch sampler for an empty cached dataset."
            )

        pattern: list[str] = []
        for idx in range(sample_rows):
            modality = self.dataset[idx].query.modality
            if modality not in pattern:
                pattern.append(modality)
            if len(pattern) >= MAX_MODALITY_PATTERN_SIZE:
                break

        if len(pattern) < MIN_MODALITY_PATTERN_SIZE:
            raise ValueError(
                "Cached dataset does not expose enough query-modality variation for modality-aware batching."
            )

        verification_rows = min(len(self.dataset), len(pattern) * 64)
        for idx in range(verification_rows):
            expected = pattern[idx % len(pattern)]
            observed = self.dataset[idx].query.modality
            if observed != expected:
                raise ValueError(
                    "Cached dataset query modalities are not globally interleaved; disable modality-aware batching "
                    "or regenerate the cache with stable ordering."
                )

        generator = random.Random(self.seed)
        sampled_indices = {
            generator.randrange(len(self.dataset))
            for _ in range(min(len(self.dataset), 256))
        }
        for idx in sampled_indices:
            expected = pattern[idx % len(pattern)]
            observed = self.dataset[idx].query.modality
            if observed != expected:
                raise ValueError(
                    "Cached dataset query modalities are not globally interleaved; disable modality-aware batching "
                    "or regenerate the cache with stable ordering."
                )
        return pattern

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        shuffled_positions = {
            modality: torch.randperm(count, generator=generator)
            for modality, count in self.modality_counts.items()
            if count > 0
        }

        batch_descriptors: list[tuple[str, int]] = []
        for modality, count in self.modality_counts.items():
            if count <= 0:
                continue
            num_batches = (
                count // self.batch_size
                if self.drop_last
                else math.ceil(count / self.batch_size)
            )
            for batch_idx in range(num_batches):
                batch_descriptors.append((modality, batch_idx))

        if not batch_descriptors:
            return

        batch_order = torch.randperm(
            len(batch_descriptors), generator=generator
        ).tolist()
        for order_idx in batch_order:
            modality, batch_idx = batch_descriptors[order_idx]
            positions = shuffled_positions[modality]
            start = batch_idx * self.batch_size
            end = start + self.batch_size
            batch_positions = positions[start:end]
            if len(batch_positions) < self.batch_size and self.drop_last:
                continue
            offset = self.modality_offsets[modality]
            yield (offset + batch_positions * self.period).tolist()

    def __len__(self) -> int:
        total_batches = 0
        for count in self.modality_counts.values():
            total_batches += (
                count // self.batch_size
                if self.drop_last
                else math.ceil(count / self.batch_size)
            )
        return total_batches


def _tri_encoder_checkpoint_state_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, "trainer_state.json")


def save_tri_encoder_checkpoint(
    accelerator: Accelerator,
    checkpoint_dir: str,
    epoch: int,
    micro_step_in_epoch: int,
    global_step: int,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    accelerator.save_state(checkpoint_dir)
    if accelerator.is_main_process:
        with open(
            _tri_encoder_checkpoint_state_path(checkpoint_dir), "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    "epoch": epoch,
                    "micro_step_in_epoch": micro_step_in_epoch,
                    "global_step": global_step,
                },
                handle,
                indent=2,
            )


def load_tri_encoder_checkpoint_state(checkpoint_dir: str) -> dict[str, int]:
    state_path = _tri_encoder_checkpoint_state_path(checkpoint_dir)
    if not os.path.exists(state_path):
        return {"epoch": 0, "micro_step_in_epoch": 0, "global_step": 0}

    with open(state_path, encoding="utf-8") as handle:
        raw = json.load(handle)
    return {
        "epoch": int(raw.get("epoch", 0)),
        "micro_step_in_epoch": int(raw.get("micro_step_in_epoch", 0)),
        "global_step": int(raw.get("global_step", 0)),
    }


def evaluate_tri_encoder_model(
    model: torch.nn.Module,
    eval_loader: DataLoader | None,
    accelerator: Accelerator,
    scale: float,
) -> dict[str, float]:
    if eval_loader is None:
        return {}

    model.eval()
    eval_losses = []
    eval_top1 = []
    eval_progress = None
    if accelerator.is_main_process:
        eval_progress = tqdm(
            eval_loader,
            desc="eval",
            leave=False,
            dynamic_ncols=True,
        )
    eval_iterable = eval_progress if eval_progress is not None else eval_loader
    with torch.no_grad():
        for batch in eval_iterable:
            anchor, positive = _encode_query_positive_batch(
                model, batch["query"], batch["positive"]
            )
            loss = multiple_negatives_ranking_loss(anchor, positive, scale=scale)
            scores = torch.matmul(anchor, positive.T)
            labels = torch.arange(scores.shape[0], device=scores.device)
            top1 = (scores.argmax(dim=1) == labels).float().mean()

            gathered_loss = accelerator.gather_for_metrics(loss.detach().reshape(1))
            gathered_top1 = accelerator.gather_for_metrics(top1.detach().reshape(1))
            eval_losses.append(gathered_loss.float().mean().item())
            eval_top1.append(gathered_top1.float().mean().item())

            if eval_progress is not None:
                eval_progress.set_postfix(
                    loss=f"{sum(eval_losses) / len(eval_losses):.4f}",
                    top1=f"{sum(eval_top1) / len(eval_top1):.4f}",
                )

    if eval_progress is not None:
        eval_progress.close()

    model.train()
    if not eval_losses:
        return {}
    return {
        "eval_loss": float(sum(eval_losses) / len(eval_losses)),
        "eval_top1": float(sum(eval_top1) / len(eval_top1)),
    }


def save_tri_encoder_final_artifacts(
    accelerator: Accelerator, model: torch.nn.Module, cfg: dict[str, Any]
) -> None:
    final_dir = os.path.join(cfg["output_dir"], "final")
    os.makedirs(final_dir, exist_ok=True)

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), os.path.join(final_dir, "model.pt"))
        with open(
            os.path.join(final_dir, "config.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(cfg, handle, indent=2)
