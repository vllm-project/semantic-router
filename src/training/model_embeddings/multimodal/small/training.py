"""Contrastive epoch training and retrieval evaluation."""

from __future__ import annotations

import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .checkpoints import save_batch_checkpoint

RETRIEVAL_CUTOFFS = (1, 5, 10)


@dataclass(frozen=True)
class EpochResult:
    loss: float
    last_batch: int | None
    batches: int


def _wrapper(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _autocast(device: torch.device, enabled: bool) -> AbstractContextManager[Any]:
    if not enabled or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.version.hip is not None else torch.float16
    return torch.amp.autocast("cuda", dtype=dtype)


def _move_batch(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[str, tuple[Any, Any, Any]]:
    if "input_features" in batch:
        return "audio", (
            batch["input_features"].to(device, non_blocking=True),
            batch["input_ids"].to(device, non_blocking=True),
            batch["attention_mask"].to(device, non_blocking=True),
        )
    if "pixel_values" in batch:
        return "image", (
            batch["pixel_values"].to(device, non_blocking=True),
            batch["input_ids"].to(device, non_blocking=True),
            batch["attention_mask"].to(device, non_blocking=True),
        )
    wrapper = _wrapper(model)
    if not hasattr(wrapper, "preprocess"):
        raise ValueError("Raw image batches require the image-text training wrapper")
    return "image", wrapper.preprocess(batch["images"], batch["captions"], device)


def _is_hardware_exception(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "hip error",
            "cuda error",
            "device-side assert",
            "illegal memory",
            "launch failure",
        )
    )


def _clear_after_hardware_exception(optimizer: Any, wait_seconds: int) -> None:
    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(wait_seconds)


def _train_batch(
    model: torch.nn.Module,
    batch: dict[str, Any],
    loss_fn: Any,
    device: torch.device,
    divisor: int,
    synchronize: bool,
    use_amp: bool,
) -> float:
    sync_context = (
        model.no_sync()
        if hasattr(model, "no_sync") and not synchronize
        else nullcontext()
    )
    with sync_context:
        _, tensors = _move_batch(model, batch, device)
        with _autocast(device, use_amp):
            left_embedding, text_embedding = model(*tensors)
            loss = loss_fn(left_embedding, text_embedding) / divisor
        loss.backward()
    return float(loss.detach().item() * divisor)


def _pad_batches(
    batches: list[dict[str, Any]],
    target_batches: int | None,
) -> list[dict[str, Any]]:
    if target_batches is None or len(batches) >= target_batches:
        return batches
    if not batches:
        return []
    original = list(batches)
    batches.extend(
        original[index % len(original)]
        for index in range(target_batches - len(original))
    )
    return batches


def _train_with_retries(
    model: torch.nn.Module,
    batch: dict[str, Any],
    loss_fn: Any,
    optimizer: Any,
    device: torch.device,
    divisor: int,
    synchronize: bool,
    use_amp: bool,
    max_retries: int,
) -> float:
    for retry in range(max_retries):
        try:
            return _train_batch(
                model,
                batch,
                loss_fn,
                device,
                divisor,
                synchronize,
                use_amp,
            )
        except RuntimeError as exc:
            if not _is_hardware_exception(exc) or retry + 1 >= max_retries:
                raise
            _clear_after_hardware_exception(optimizer, 10 * (retry + 1))
    raise RuntimeError("Training batch retry loop terminated unexpectedly")


def _maybe_save_batch_checkpoint(
    model: torch.nn.Module,
    optimizer: Any,
    scheduler: Any,
    checkpoint_dir: str | None,
    epoch: int,
    batch_index: int,
    shard_idx: int | None,
    total_loss: float,
    completed: int,
    synchronize: bool,
    save_every_n_batches: int,
    rank: int,
) -> None:
    should_save = (
        synchronize
        and checkpoint_dir is not None
        and save_every_n_batches > 0
        and (batch_index + 1) % save_every_n_batches == 0
        and rank == 0
    )
    if not should_save:
        return
    save_batch_checkpoint(
        model,
        optimizer,
        scheduler,
        Path(checkpoint_dir).with_name(Path(checkpoint_dir).name + "_batch"),
        {
            "epoch": epoch,
            "batch_idx": batch_index,
            "shard_idx": shard_idx,
            "total_loss": total_loss,
            "num_batches": completed,
        },
    )


def train_epoch(
    model: torch.nn.Module,
    dataloader: Any,
    loss_fn: Any,
    optimizer: Any,
    scheduler: Any,
    device: torch.device,
    epoch: int,
    *,
    grad_accum: int = 1,
    rank: int = 0,
    checkpoint_dir: str | None = None,
    start_batch: int = 0,
    save_every_n_batches: int = 0,
    max_retries: int = 3,
    shard_idx: int | None = None,
    use_amp: bool = False,
    target_batches: int | None = None,
    max_steps: int | None = None,
) -> EpochResult:
    """Train one shard/epoch with DDP-safe accumulation and bounded retries."""
    if grad_accum < 1:
        raise ValueError("grad_accum must be at least 1")
    batches = _pad_batches(list(dataloader), target_batches)
    if max_steps is not None:
        batches = batches[:max_steps]
    if not batches:
        return EpochResult(0.0, None, 0)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    completed = 0
    last_batch: int | None = None
    iterator = tqdm(
        enumerate(batches),
        total=len(batches),
        desc=f"Epoch {epoch}",
        disable=rank != 0,
    )
    for batch_index, batch in iterator:
        if batch_index < start_batch:
            continue
        group_start = batch_index - batch_index % grad_accum
        group_end = min(group_start + grad_accum, len(batches))
        divisor = group_end - group_start
        synchronize = batch_index + 1 == group_end

        batch_loss = _train_with_retries(
            model,
            batch,
            loss_fn,
            optimizer,
            device,
            divisor,
            synchronize,
            use_amp,
            max_retries,
        )

        if synchronize:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += batch_loss
        completed += 1
        last_batch = batch_index
        if rank == 0 and batch_index % 10 == 0:
            learning_rate = optimizer.param_groups[0]["lr"]
            iterator.set_postfix(loss=f"{batch_loss:.4f}", lr=f"{learning_rate:.2e}")

        _maybe_save_batch_checkpoint(
            model,
            optimizer,
            scheduler,
            checkpoint_dir,
            epoch,
            batch_index,
            shard_idx,
            total_loss,
            completed,
            synchronize,
            save_every_n_batches,
            rank,
        )

    return EpochResult(total_loss / max(completed, 1), last_batch, completed)


def evaluate(
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device,
    *,
    rank: int = 0,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Compute paired retrieval ranks over image-text or audio-text batches."""
    model.eval()
    left_embeddings: list[torch.Tensor] = []
    text_embeddings: list[torch.Tensor] = []
    with torch.no_grad():
        for index, batch in enumerate(
            tqdm(dataloader, desc="Evaluating", disable=rank != 0)
        ):
            if max_batches is not None and index >= max_batches:
                break
            _, tensors = _move_batch(model, batch, device)
            left, text = model(*tensors)
            left_embeddings.append(left.cpu())
            text_embeddings.append(text.cpu())

    if not left_embeddings:
        raise ValueError("Evaluation DataLoader produced no batches")
    left = torch.cat(left_embeddings)
    text = torch.cat(text_embeddings)
    similarity = text @ left.T
    diagonal = similarity.diagonal().unsqueeze(1)
    ranks = (similarity > diagonal).sum(dim=1).to(torch.float32) + 1
    recall = {
        f"R@{cutoff}": float((ranks <= cutoff).float().mean().item() * 100)
        for cutoff in RETRIEVAL_CUTOFFS
    }
    return {**recall, "MeanRank": float(ranks.mean().item())}
