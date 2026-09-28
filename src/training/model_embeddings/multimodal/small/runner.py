"""End-to-end training orchestration kept separate from the CLI."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from transformers import get_cosine_schedule_with_warmup

from .checkpoints import save_epoch_checkpoint, save_model, write_metrics
from .distributed import DistributedContext, barrier, rank0_print, wrap_distributed
from .losses import InfoNCELoss, MatryoshkaLoss
from .pipeline import (
    DataBundle,
    create_data_bundle,
    create_loader,
    estimate_optimizer_steps,
    synchronized_batch_count,
)
from .stages import apply_stage, create_model, load_weights
from .training import EpochResult, evaluate, train_epoch
from .wrappers import AudioContrastiveWrapper, ContrastiveTrainingWrapper


def _seed_everything(seed: int, rank: int) -> None:
    effective_seed = seed + rank
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)


def _trainable_summary(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return trainable, total


def _create_optimizer_and_scheduler(
    model: torch.nn.Module,
    training_config: dict[str, Any],
    bundle: DataBundle,
    context: DistributedContext,
) -> tuple[Any, Any]:
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.01)),
    )
    steps = estimate_optimizer_steps(
        bundle,
        int(training_config["batch_size"]),
        int(training_config["num_epochs"]),
        int(training_config["grad_accum"]),
        context.world_size,
    )
    warmup = math.ceil(steps * float(training_config.get("warmup_ratio", 0.1)))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, steps)
    return optimizer, scheduler


def _create_loss(training_config: dict[str, Any]) -> torch.nn.Module:
    loss_type = training_config.get("loss_type", "infonce")
    temperature = float(training_config.get("temperature", 0.07))
    if loss_type == "infonce":
        return InfoNCELoss(temperature=temperature)
    if loss_type == "matryoshka":
        return MatryoshkaLoss(
            dim_schedule=[int(dim) for dim in training_config["matryoshka_dims"]],
            temperature=temperature,
        )
    raise ValueError(f"Unsupported training loss: {loss_type}")


def _restore_training_state(
    resume: str | None,
    optimizer: Any,
    scheduler: Any,
) -> tuple[int, float, float]:
    if not resume:
        return 0, 0.0, float("inf")
    path = Path(resume)
    state_path = (
        path / "training_state.pt"
        if path.is_dir()
        else path.parent / "training_state.pt"
    )
    if not state_path.is_file():
        return 0, 0.0, float("inf")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and state.get("scheduler_state_dict"):
        scheduler.load_state_dict(state["scheduler_state_dict"])
    return (
        int(state.get("epoch", -1)) + 1,
        float(state.get("best_r1", 0.0)),
        float(state.get("best_loss", float("inf"))),
    )


def _run_standard_epoch(
    model: torch.nn.Module,
    bundle: DataBundle,
    config: dict[str, Any],
    context: DistributedContext,
    loss_fn: Any,
    optimizer: Any,
    scheduler: Any,
    epoch: int,
) -> EpochResult:
    training = config["training"]
    loader = create_loader(
        bundle.train,
        bundle.collate,
        config["data"],
        context,
        batch_size=int(training["batch_size"]),
        training=True,
    )
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)
    return train_epoch(
        model,
        loader,
        loss_fn,
        optimizer,
        scheduler,
        context.device,
        epoch,
        grad_accum=int(training["grad_accum"]),
        rank=context.rank,
        checkpoint_dir=config["runtime"]["output_dir"],
        save_every_n_batches=int(training.get("save_every_n_batches", 0)),
        max_retries=int(training.get("hardware_exception_retries", 3)),
        use_amp=bool(training.get("use_amp", False)),
        max_steps=training.get("max_steps"),
    )


def _run_sequential_epoch(
    model: torch.nn.Module,
    bundle: DataBundle,
    config: dict[str, Any],
    context: DistributedContext,
    loss_fn: Any,
    optimizer: Any,
    scheduler: Any,
    epoch: int,
) -> EpochResult:
    training = config["training"]
    if epoch > 0:
        bundle.train.reset()
    losses = 0.0
    batches = 0
    last_batch = None
    remaining = training.get("max_steps")
    for shard_index in range(bundle.train.num_shards):
        loader = create_loader(
            bundle.train,
            bundle.collate,
            config["data"],
            context,
            batch_size=int(training["batch_size"]),
            training=True,
            sequential=True,
        )
        target_batches = synchronized_batch_count(len(loader), context)
        result = train_epoch(
            model,
            loader,
            loss_fn,
            optimizer,
            scheduler,
            context.device,
            epoch,
            grad_accum=int(training["grad_accum"]),
            rank=context.rank,
            checkpoint_dir=config["runtime"]["output_dir"],
            save_every_n_batches=int(training.get("save_every_n_batches", 0)),
            max_retries=int(training.get("hardware_exception_retries", 3)),
            shard_idx=shard_index,
            use_amp=bool(training.get("use_amp", False)),
            target_batches=target_batches,
            max_steps=remaining,
        )
        losses += result.loss * result.batches
        batches += result.batches
        last_batch = result.last_batch
        if remaining is not None:
            remaining -= result.batches
            if remaining <= 0:
                break
        if shard_index + 1 < bundle.train.num_shards:
            bundle.train.next_shard()
    return EpochResult(losses / max(1, batches), last_batch, batches)


def _evaluate_rank0(
    model: torch.nn.Module,
    bundle: DataBundle,
    config: dict[str, Any],
    context: DistributedContext,
) -> dict[str, float] | None:
    barrier(context)
    metrics = None
    if context.is_main:
        evaluation_model = model.module if hasattr(model, "module") else model
        loader = create_loader(
            bundle.validation,
            bundle.collate,
            config["data"],
            DistributedContext(0, 1, 0, False, context.device),
            batch_size=int(config["training"]["batch_size"]),
            training=False,
        )
        metrics = evaluate(
            evaluation_model,
            loader,
            context.device,
            max_batches=config["training"].get("evaluation_max_batches"),
        )
    barrier(context)
    return metrics


def _broadcast_stop(stop: bool, context: DistributedContext) -> bool:
    if not context.distributed:
        return stop
    value = torch.tensor([int(stop)], device=context.device)
    dist.broadcast(value, src=0)
    return bool(value.item())


def _write_run_manifest(config: dict[str, Any], output_dir: Path) -> None:
    manifest = {
        "schema_version": 1,
        "artifact": config["artifact"],
        "model": config["model"],
        "training": config["training"],
        "data_contract": config["data"].get("dataset", {}),
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def run(config: dict[str, Any], context: DistributedContext) -> None:
    """Execute the configured compact multimodal training workflow."""
    training = config["training"]
    output_dir = Path(config["runtime"]["output_dir"]).expanduser().resolve()
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_run_manifest(config, output_dir)
    barrier(context)
    _seed_everything(int(training.get("seed", 42)), context.rank)

    base_model = create_model(config["model"])
    loaded = load_weights(base_model, config["runtime"].get("resume"))
    stage_description = apply_stage(base_model, int(training["stage"]))
    wrapper_type = (
        AudioContrastiveWrapper
        if int(training["stage"]) in (5, 6, 7)
        else ContrastiveTrainingWrapper
    )
    model = wrap_distributed(wrapper_type(base_model).to(context.device), context)
    trainable, total = _trainable_summary(model)
    rank0_print(
        context,
        f"Stage {training['stage']}: {stage_description}; "
        f"{trainable:,}/{total:,} parameters trainable",
    )
    if loaded is not None:
        rank0_print(context, f"Loaded weights from {loaded}")

    bundle = create_data_bundle(config["data"], context)
    optimizer, scheduler = _create_optimizer_and_scheduler(
        model,
        training,
        bundle,
        context,
    )
    start_epoch, best_r1, best_loss = _restore_training_state(
        config["runtime"].get("resume"),
        optimizer,
        scheduler,
    )
    loss_fn = _create_loss(training)
    epochs_without_improvement = 0

    for epoch in range(start_epoch, int(training["num_epochs"])):
        epoch_runner = (
            _run_sequential_epoch if bundle.sequential else _run_standard_epoch
        )
        result = epoch_runner(
            model,
            bundle,
            config,
            context,
            loss_fn,
            optimizer,
            scheduler,
            epoch,
        )
        metrics = _evaluate_rank0(model, bundle, config, context)
        stop = False
        if context.is_main and metrics is not None:
            rank0_print(
                context,
                f"Epoch {epoch}: loss={result.loss:.4f}, R@1={metrics['R@1']:.2f}",
            )
            if metrics["R@1"] > best_r1:
                best_r1 = metrics["R@1"]
                save_model(model, output_dir / "best")
                write_metrics(
                    output_dir / "best",
                    {"epoch": epoch, "train_loss": result.loss, **metrics},
                )
            if result.loss < best_loss - float(
                training.get("early_stop_min_delta", 0.01)
            ):
                best_loss = result.loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            patience = int(training.get("early_stop_patience", 0))
            stop = patience > 0 and epochs_without_improvement >= patience
            if training.get("save_every_epoch", True):
                save_epoch_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    output_dir / "latest",
                    {
                        "epoch": epoch,
                        "best_r1": best_r1,
                        "best_loss": best_loss,
                        "train_loss": result.loss,
                    },
                )
        if _broadcast_stop(stop, context):
            break
        barrier(context)

    rank0_print(context, f"Training complete; best R@1={best_r1:.2f}")
