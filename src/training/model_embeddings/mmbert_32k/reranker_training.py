"""Narrow training loop for the 2D Matryoshka reranker."""
# Derived from Model-training@3bc41e1322ee5a53e08d18eb940855dec53c1539.
# Upstream blob: 36951a954bf2be62ea4d6536fecfd3ce0aad6d5c.

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from .reranker_data import RerankerDataset, collate_fn
from .reranker_evaluation import evaluate_model
from .reranker_loss import Matryoshka2DLoss
from .reranker_model import Matryoshka2DReranker

logger = logging.getLogger(__name__)


@dataclass
class _TrainingState:
    global_step: int = 0
    total_loss: float = 0.0


def _select_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("GPU Memory: %.1f GB", memory)
    return device


def _prepare_output(args) -> None:
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"./outputs/reranker-2d-matryoshka-{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as handle:
        json.dump(vars(args), handle, indent=2)


def _parse_indices(value: str | None) -> list[int] | None:
    return [int(item) for item in value.split(",")] if value else None


def _load_model_and_tokenizer(args, device):
    logger.info("Loading tokenizer from %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, revision=args.model_revision, trust_remote_code=True
    )
    logger.info("Loading model from %s", args.model_name)
    model = Matryoshka2DReranker(
        model_name_or_path=args.model_name,
        model_revision=args.model_revision,
        layer_indices=_parse_indices(args.layer_indices),
        dim_indices=_parse_indices(args.dim_indices),
        use_flash_attn=args.use_flash_attn,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        pooling_strategy=args.pooling_strategy,
    ).to(device)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    logger.info("Parameters: %s total, %s trainable", f"{total:,}", f"{trainable:,}")
    if args.gradient_checkpointing and hasattr(
        model.encoder, "gradient_checkpointing_enable"
    ):
        model.encoder.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    return model, tokenizer


def _build_loader(args, tokenizer):
    logger.info("%s", "=" * 60)
    logger.info("Loading datasets (BGE-reranker-v2-m3 style)")
    logger.info("%s", "=" * 60)
    dataset = RerankerDataset(
        args.train_data,
        tokenizer,
        max_length=args.max_length,
        max_samples=args.max_samples,
        use_quora=args.use_quora,
        use_fever=args.use_fever,
        max_quora_samples=args.max_quora_samples,
        max_fever_samples=args.max_fever_samples,
        negatives_per_query=args.negatives_per_query,
    )
    if not dataset:
        raise ValueError("No reranker examples were loaded from train_data")
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def _build_optimization(args, model, loader):
    total_steps = len(loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    logger.info("Training steps: %s (%s warmup)", total_steps, warmup_steps)
    effective_batch = args.batch_size * args.gradient_accumulation_steps
    logger.info("Effective batch size: %s", effective_batch)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss = None
    if args.use_2d_matryoshka:
        loss = Matryoshka2DLoss(model.layer_indices, model.dim_indices)
        logger.info("Using 2D Matryoshka loss")
    return optimizer, scheduler, loss


def _forward_loss(args, model, batch, matryoshka_loss):
    with torch.amp.autocast(
        "cuda", dtype=torch.bfloat16 if args.bf16 else torch.float32
    ):
        if args.use_2d_matryoshka:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_all_scores=True,
            )
            loss = matryoshka_loss(outputs["all_scores"], batch["labels"])
        else:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs["loss"]
        return loss / args.gradient_accumulation_steps


def _log_progress(args, state, scheduler, progress) -> None:
    values = {
        "loss": f"{state.total_loss / args.logging_steps:.4f}",
        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
    }
    if torch.cuda.is_available():
        values["mem"] = f"{torch.cuda.memory_allocated() / 1e9:.1f}GB"
    progress.set_postfix(values)
    state.total_loss = 0.0


def _optimizer_step(args, model, tokenizer, optimizer, scheduler, progress, state):
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    state.global_step += 1
    if state.global_step % args.logging_steps == 0:
        _log_progress(args, state, scheduler, progress)
    if args.save_steps > 0 and state.global_step % args.save_steps == 0:
        checkpoint = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        logger.info("Saved checkpoint: %s", checkpoint)


def _run_training(args, model, tokenizer, loader, optimizer, scheduler, loss):
    logger.info("Starting training...")
    model.train()
    state = _TrainingState()
    optimizer.zero_grad()
    device = next(model.parameters()).device
    for epoch in range(args.epochs):
        logger.info("Epoch %s/%s", epoch + 1, args.epochs)
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}")
        for step, raw_batch in enumerate(progress):
            batch = {key: value.to(device) for key, value in raw_batch.items()}
            batch_loss = _forward_loss(args, model, batch, loss)
            batch_loss.backward()
            state.total_loss += batch_loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                _optimizer_step(
                    args,
                    model,
                    tokenizer,
                    optimizer,
                    scheduler,
                    progress,
                    state,
                )


def train(args):
    """Train and export the source-compatible Matryoshka reranker."""
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _select_device()
    _prepare_output(args)
    model, tokenizer = _load_model_and_tokenizer(args, device)
    loader = _build_loader(args, tokenizer)
    optimizer, scheduler, loss = _build_optimization(args, model, loader)
    _run_training(args, model, tokenizer, loader, optimizer, scheduler, loss)
    logger.info("Saving final model: %s", args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Training complete!")
    evaluate_model(model, tokenizer, device)
    return model
