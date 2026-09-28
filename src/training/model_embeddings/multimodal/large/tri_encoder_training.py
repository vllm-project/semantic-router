"""Orchestration for the cached datacenter tri-encoder training path."""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Any

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from .data import SequentialShardDataset, collate_records
from .model import multiple_negatives_ranking_loss
from .runtime import log_progress, normalize_mixed_precision, write_status
from .tri_encoder import (
    InterleavedModalityBatchSampler,
    _build_cached_loader_kwargs,
    _encode_query_positive_batch,
    build_datacenter_tri_encoder_model,
    evaluate_tri_encoder_model,
    load_datacenter_tri_encoder_datasets,
    load_tri_encoder_checkpoint_state,
    save_tri_encoder_checkpoint,
    save_tri_encoder_final_artifacts,
)


@dataclass
class LoaderBundle:
    train_dataset: Any
    train_info: dict[str, Any]
    eval_dataset: Any
    eval_info: dict[str, Any] | None
    train_loader: DataLoader | None
    eval_loader: DataLoader | None
    sequential: bool
    homogeneous_pattern: list[str] | None


@dataclass
class TrainingState:
    accelerator: Accelerator
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any
    loaders: LoaderBundle
    cfg: dict[str, Any]
    max_train_steps: int
    start_epoch: int = 0
    resume_micro_step: int = 0
    global_step: int = 0
    progress_bar: Any = None


def _make_accelerator(training_cfg: dict[str, Any]) -> Accelerator:
    return Accelerator(
        gradient_accumulation_steps=int(training_cfg.get("grad_accum_steps", 1)),
        mixed_precision=normalize_mixed_precision(
            training_cfg.get("mixed_precision", "bf16")
        ),
    )


def _make_train_loader(
    dataset: Any, cfg: dict[str, Any]
) -> tuple[DataLoader | None, bool, list[str] | None]:
    training_cfg = cfg["training"]
    sequential = isinstance(dataset, SequentialShardDataset)
    if sequential:
        return None, True, None

    kwargs = _build_cached_loader_kwargs(
        int(training_cfg.get("num_workers", 4)),
        int(training_cfg.get("prefetch_factor", 4)),
    )
    pattern = None
    if bool(training_cfg.get("modality_homogeneous_batches", False)):
        sampler = InterleavedModalityBatchSampler(
            dataset,
            batch_size=int(training_cfg["batch_size"]),
            drop_last=bool(training_cfg.get("drop_last", True)),
            seed=int(cfg.get("seed", 42)),
        )
        kwargs["batch_sampler"] = sampler
        pattern = list(sampler.pattern)
    else:
        kwargs.update(
            batch_size=int(training_cfg["batch_size"]),
            shuffle=True,
            drop_last=bool(training_cfg.get("drop_last", True)),
        )
    return DataLoader(dataset, collate_fn=collate_records, **kwargs), False, pattern


def _make_eval_loader(dataset: Any, cfg: dict[str, Any]) -> DataLoader | None:
    if dataset is None:
        return None
    training_cfg = cfg["training"]
    validation_cfg = cfg.get("validation", {})
    workers = int(
        validation_cfg.get(
            "num_workers", max(1, int(training_cfg.get("num_workers", 4)) // 2)
        )
    )
    return DataLoader(
        dataset,
        batch_size=int(validation_cfg.get("batch_size", training_cfg["batch_size"])),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_records,
        **_build_cached_loader_kwargs(
            workers, int(training_cfg.get("prefetch_factor", 4))
        ),
    )


def _loaders(cfg: dict[str, Any]) -> LoaderBundle:
    log_progress("[startup] loading cached datacenter tri-encoder datasets")
    train_dataset, train_info, eval_dataset, eval_info = (
        load_datacenter_tri_encoder_datasets(cfg)
    )
    train_loader, sequential, pattern = _make_train_loader(train_dataset, cfg)
    bundle = LoaderBundle(
        train_dataset=train_dataset,
        train_info=train_info,
        eval_dataset=eval_dataset,
        eval_info=eval_info,
        train_loader=train_loader,
        eval_loader=_make_eval_loader(eval_dataset, cfg),
        sequential=sequential,
        homogeneous_pattern=pattern,
    )
    _log_dataset_summary(bundle)
    return bundle


def _log_dataset_summary(loaders: LoaderBundle) -> None:
    info = loaders.train_info
    log_progress(
        "[startup] loaded cached train dataset with "
        f"{info['num_rows']} rows and observed modalities {info['modalities']}"
    )
    if loaders.eval_info is not None:
        log_progress(
            "[startup] loaded cached eval dataset with "
            f"{loaders.eval_info['num_rows']} rows and observed modalities "
            f"{loaders.eval_info['modalities']}"
        )


def _updates_per_epoch(loaders: LoaderBundle, training_cfg: dict[str, Any]) -> int:
    accumulation = max(int(training_cfg.get("grad_accum_steps", 1)), 1)
    if loaders.sequential:
        micro_steps = loaders.train_dataset.estimated_num_batches(
            batch_size=int(training_cfg["batch_size"]),
            drop_last=bool(training_cfg.get("drop_last", True)),
        )
    else:
        if loaders.train_loader is None:
            raise RuntimeError("Non-sequential training requires a DataLoader")
        micro_steps = len(loaders.train_loader)
    return max(1, math.ceil(micro_steps / accumulation))


def _make_schedule(
    optimizer: torch.optim.Optimizer,
    loaders: LoaderBundle,
    training_cfg: dict[str, Any],
) -> tuple[Any, int]:
    updates = _updates_per_epoch(loaders, training_cfg)
    max_steps = int(
        training_cfg.get("max_steps")
        or math.ceil(float(training_cfg["epochs"]) * updates)
    )
    warmup = training_cfg.get("warmup_steps")
    if warmup is None:
        warmup = math.ceil(max_steps * float(training_cfg.get("warmup_ratio", 0.0)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup),
        num_training_steps=max_steps,
    )
    return scheduler, max_steps


def _prepare_state(cfg: dict[str, Any]) -> TrainingState:
    training_cfg = cfg["training"]
    accelerator = _make_accelerator(training_cfg)
    loaders = _loaders(cfg)
    log_progress("[startup] building datacenter tri-encoder model")
    model = build_datacenter_tri_encoder_model(cfg)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
    )
    scheduler, max_steps = _make_schedule(optimizer, loaders, training_cfg)

    if loaders.sequential:
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    else:
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, loaders.train_loader, scheduler
        )
        loaders.train_loader = train_loader
    if loaders.eval_loader is not None:
        loaders.eval_loader = accelerator.prepare(loaders.eval_loader)
    accelerator.register_for_checkpointing(scheduler)
    return TrainingState(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        cfg=cfg,
        max_train_steps=max_steps,
    )


def _restore_state(state: TrainingState, resume_path: str | None) -> None:
    if not resume_path:
        return
    log_progress(
        f"[startup] resuming datacenter tri-encoder checkpoint from {resume_path}"
    )
    state.accelerator.load_state(resume_path)
    receipt = load_tri_encoder_checkpoint_state(resume_path)
    state.start_epoch = int(receipt["epoch"])
    state.resume_micro_step = int(receipt["micro_step_in_epoch"])
    state.global_step = int(receipt["global_step"])


def _start_progress(state: TrainingState) -> None:
    training_cfg = state.cfg["training"]
    loaders = state.loaders
    workers = int(training_cfg.get("num_workers", 4))
    eval_workers = int(
        state.cfg.get("validation", {}).get("num_workers", max(1, workers // 2))
    )
    log_progress(
        "[startup] cached dataloader workers "
        f"train={workers} eval={eval_workers} "
        f"prefetch_factor={int(training_cfg.get('prefetch_factor', 4))}"
    )
    if loaders.homogeneous_pattern:
        log_progress(
            "[startup] modality-homogeneous batching enabled with interleave pattern "
            f"{loaders.homogeneous_pattern}"
        )
    if loaders.sequential:
        log_progress(
            "[startup] sequential shard loading enabled; train dataloader workers forced to 0"
        )
    if state.accelerator.is_main_process:
        state.progress_bar = tqdm(
            total=state.max_train_steps,
            initial=state.global_step,
            desc="train",
            unit="step",
            dynamic_ncols=True,
            smoothing=0.1,
            mininterval=5.0,
            file=sys.stdout,
        )


def _train_microbatch(state: TrainingState, batch: dict[str, Any]) -> Any:
    training_cfg = state.cfg["training"]
    with state.accelerator.accumulate(state.model):
        anchor, positive = _encode_query_positive_batch(
            state.model, batch["query"], batch["positive"]
        )
        loss = multiple_negatives_ranking_loss(
            anchor,
            positive,
            scale=float(state.cfg.get("loss", {}).get("scale", 20.0)),
        )
        state.accelerator.backward(loss)
        if state.accelerator.sync_gradients:
            state.accelerator.clip_grad_norm_(
                state.model.parameters(),
                float(training_cfg.get("max_grad_norm", 1.0)),
            )
            state.optimizer.step()
            state.scheduler.step()
            state.optimizer.zero_grad(set_to_none=True)
            state.global_step += 1
            if state.progress_bar is not None:
                state.progress_bar.update(1)
    return loss


def _step_events(state: TrainingState, loss: Any, epoch: int, micro_step: int) -> None:
    if not state.accelerator.sync_gradients:
        return
    training_cfg = state.cfg["training"]
    log_every = int(training_cfg.get("log_every", 10))
    save_every = int(training_cfg.get("save_every", 1000))
    eval_every = int(training_cfg.get("eval_every", save_every))
    current = state.global_step

    if log_every > 0 and current % log_every == 0:
        mean_loss = (
            state.accelerator.gather_for_metrics(loss.detach().reshape(1))
            .float()
            .mean()
            .item()
        )
        lr = state.scheduler.get_last_lr()[0]
        if state.progress_bar is not None:
            state.progress_bar.set_postfix(
                epoch=epoch + 1, loss=f"{mean_loss:.4f}", lr=f"{lr:.3e}"
            )
        log_progress(
            f"[train] epoch={epoch + 1} step={current}/{state.max_train_steps} "
            f"loss={mean_loss:.5f} lr={lr:.3e}"
        )
    if save_every > 0 and current % save_every == 0:
        checkpoint_dir = os.path.join(state.cfg["output_dir"], f"checkpoint-{current}")
        save_tri_encoder_checkpoint(
            state.accelerator,
            checkpoint_dir,
            epoch=epoch,
            micro_step_in_epoch=micro_step,
            global_step=current,
        )
        log_progress(
            f"[checkpoint] saved datacenter tri-encoder checkpoint to {checkpoint_dir}"
        )
    if (
        state.loaders.eval_loader is not None
        and eval_every > 0
        and current % eval_every == 0
    ):
        metrics = evaluate_tri_encoder_model(
            state.model,
            state.loaders.eval_loader,
            state.accelerator,
            float(state.cfg.get("loss", {}).get("scale", 20.0)),
        )
        if metrics:
            log_progress(
                f"[eval] step={current} eval_loss={metrics['eval_loss']:.5f} "
                f"eval_top1={metrics['eval_top1']:.4f}"
            )


def _run_loader(
    state: TrainingState,
    loader: DataLoader,
    epoch: int,
    micro_step: int,
) -> int:
    for batch in loader:
        if epoch == state.start_epoch and micro_step < state.resume_micro_step:
            micro_step += 1
            continue
        loss = _train_microbatch(state, batch)
        micro_step += 1
        _step_events(state, loss, epoch, micro_step)
        if state.global_step >= state.max_train_steps:
            break
    return micro_step


def _sequential_loader(state: TrainingState) -> DataLoader:
    training_cfg = state.cfg["training"]
    return DataLoader(
        state.loaders.train_dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=bool(training_cfg.get("shuffle_within_shard", True)),
        drop_last=bool(training_cfg.get("drop_last", True)),
        collate_fn=collate_records,
        num_workers=0,
        pin_memory=True,
    )


def _run_epoch(state: TrainingState, epoch: int) -> None:
    loaders = state.loaders
    if not loaders.sequential:
        if loaders.train_loader is None:
            raise RuntimeError("Training DataLoader was not prepared")
        sampler = getattr(loaders.train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        _run_loader(state, loaders.train_loader, epoch, 0)
        return

    has_shard = loaders.train_dataset.reset(epoch)
    micro_step = 0
    while has_shard and state.global_step < state.max_train_steps:
        micro_step = _run_loader(state, _sequential_loader(state), epoch, micro_step)
        has_shard = loaders.train_dataset.next_shard()


def _run_epochs(state: TrainingState) -> None:
    state.model.train()
    state.optimizer.zero_grad(set_to_none=True)
    epochs = math.ceil(float(state.cfg["training"]["epochs"]))
    for epoch in range(state.start_epoch, epochs):
        _run_epoch(state, epoch)
        state.resume_micro_step = 0
        if state.global_step >= state.max_train_steps:
            break


def _finalize(state: TrainingState, st_version: str) -> None:
    state.accelerator.wait_for_everyone()
    if state.progress_bar is not None:
        state.progress_bar.close()
    save_tri_encoder_final_artifacts(state.accelerator, state.model, state.cfg)
    metrics = {
        "sentence_transformers_version": st_version,
        "train_rows": state.loaders.train_info["num_rows"],
        "train_modalities": state.loaders.train_info["modalities"],
        "global_step": state.global_step,
        "max_train_steps": state.max_train_steps,
        "datacenter_tri_encoder_training": True,
    }
    if state.loaders.eval_info is not None:
        metrics.update(
            eval_rows=state.loaders.eval_info["num_rows"],
            eval_modalities=state.loaders.eval_info["modalities"],
        )
    if state.accelerator.is_main_process:
        write_status(state.cfg["output_dir"], metrics)
    log_progress(
        f"[done] wrote status to {os.path.join(state.cfg['output_dir'], 'train_status.json')}"
    )


def run_datacenter_tri_encoder_training(
    cfg: dict[str, Any], resume_path: str | None, st_version: str
) -> None:
    state = _prepare_state(cfg)
    _restore_state(state, resume_path)
    _start_progress(state)
    _run_epochs(state)
    _finalize(state, st_version)
