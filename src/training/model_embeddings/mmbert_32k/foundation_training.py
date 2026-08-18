"""Narrow orchestration for config-first mmBERT 32K foundation training."""
# Derived from Model-training@3bc41e1322ee5a53e08d18eb940855dec53c1539.
# Upstream blob: a8ef9416fb4ce6e6374e5d92f5fefb4dd27221e0.

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_rope_utils import rope_config_validation

from .foundation_collators import RetrievalMaskingCollator, StandardMLMCollator
from .foundation_ewc import EWCRegularizer
from .foundation_integrity import validate_packing_manifest, write_training_receipt
from .foundation_optimization import (
    accumulation_window_size,
    optimizer_steps_per_epoch,
    should_optimizer_step,
)
from .rope_config import (
    assert_yarn_config,
    configure_modernbert_yarn,
    verify_loaded_modernbert_yarn,
)

logger = logging.getLogger(__name__)


@dataclass
class _TrainingState:
    global_step: int = 0
    total_loss: float = 0.0


def load_dataset_from_path(
    dataset_path: str,
    max_samples: int | None = None,
    dataset_revision: str | None = None,
):
    """Load a local or revision-pinned Hugging Face training dataset."""
    if os.path.exists(dataset_path):
        logger.info("Loading dataset from disk: %s", dataset_path)
        dataset = load_from_disk(dataset_path)
    else:
        logger.info("Loading dataset from HuggingFace: %s", dataset_path)
        dataset = load_dataset(dataset_path, split="train", revision=dataset_revision)
    logger.info("Dataset size: %s examples", len(dataset))
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info("Truncated to %s samples", len(dataset))
    return dataset


def _validate_dataset(args):
    dataset = load_dataset_from_path(args.dataset_path, None, args.dataset_revision)
    validation = validate_packing_manifest(
        args.dataset_path,
        dataset,
        expected_source_repo_id=args.packing_source_repo_id,
        expected_dataset_revision=args.dataset_revision,
        expected_tokenizer_repo_id=args.model_name_or_path,
        expected_tokenizer_revision=args.model_revision,
        expected_target_length=args.model_max_length,
        expected_sequence_count=args.expected_train_samples,
        expected_languages=args.packing_languages.split(","),
        expected_source_etags=args.packing_source_etags.split(","),
        expected_source_content_lengths=(
            int(value) for value in args.packing_source_content_lengths.split(",")
        ),
        expected_max_document_bytes=args.packing_max_document_bytes,
        expected_max_document_tokens=args.packing_max_document_tokens,
        acknowledge_cc100_license_unknown=args.acknowledge_cc100_license_unknown,
    )
    write_training_receipt(
        args.output_dir,
        validation=validation,
        model_repo_id=args.model_name_or_path,
        model_revision=args.model_revision,
        dataset_revision=args.dataset_revision,
        status="data-validated",
        optimizer_steps=0,
    )
    if args.max_train_samples:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))
        logger.info("Truncated training view to %s samples", len(dataset))
    return dataset, validation


def _load_tokenizer(args):
    logger.info("Loading tokenizer from %s", args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, revision=args.model_revision
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    if tokenizer.model_max_length < args.model_max_length:
        logger.warning(
            "Updating tokenizer.model_max_length: %s -> %s",
            tokenizer.model_max_length,
            args.model_max_length,
        )
        tokenizer.model_max_length = args.model_max_length
    return tokenizer


def _configure_model(args):
    logger.info("Loading model config from %s", args.model_name_or_path)
    model_config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        revision=args.model_revision,
        trust_remote_code=True,
    )
    if args.rope_scaling_type == "yarn":
        if args.yarn_extrapolation_factor != 1.0:
            raise ValueError(
                "Transformers 4.57.6 official YaRN has no extrapolation_factor; "
                "remove the override or set it to 1.0"
            )
        configure_modernbert_yarn(
            model_config,
            original_max_position_embeddings=args.rope_original_max_position_embeddings,
            target_max_position_embeddings=args.model_max_length,
            beta_fast=args.yarn_beta_fast,
            beta_slow=args.yarn_beta_slow,
            attention_implementation=args.attn_implementation,
        )
        rope_config_validation(model_config)
    elif args.model_max_length > model_config.max_position_embeddings:
        raise ValueError(
            "refusing to extend context without an explicit supported RoPE scaling"
        )
    return model_config


def _load_model(args, tokenizer, device):
    model_config = _configure_model(args)
    logger.info("Loading model weights from %s", args.model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name_or_path,
        revision=args.model_revision,
        config=model_config,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
    )
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    if args.rope_scaling_type == "yarn":
        rotary_count = verify_loaded_modernbert_yarn(
            model,
            original_max_position_embeddings=args.rope_original_max_position_embeddings,
            target_max_position_embeddings=args.model_max_length,
            beta_fast=args.yarn_beta_fast,
            beta_slow=args.yarn_beta_slow,
            attention_implementation=args.attn_implementation,
        )
        logger.info("Validated config-driven YaRN in %s attention layers", rotary_count)
    model = model.to(device)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    logger.info("Parameters: %s total, %s trainable", f"{total:,}", f"{trainable:,}")
    return model


def _build_collator(args, tokenizer):
    if not args.use_retrieval_masking:
        return StandardMLMCollator(tokenizer, mlm_probability=args.mlm_probability)
    logger.info("%s", "=" * 60)
    logger.info("Using RETRIEVAL MASKING (recommended for long context)")
    logger.info("  Retrieval probability: %s", args.retrieval_probability)
    logger.info("  Min distance: %s", args.min_distance_for_retrieval)
    logger.info("%s", "=" * 60)
    return RetrievalMaskingCollator(
        tokenizer,
        mlm_probability=args.mlm_probability,
        retrieval_probability=args.retrieval_probability,
        min_distance_for_retrieval=args.min_distance_for_retrieval,
    )


def _build_scheduler(args, optimizer, warmup_steps: int, total_steps: int):
    if args.lr_scheduler_type == "constant_with_warmup":
        logger.info("Using CONSTANT LR schedule (recommended for long context)")
        return get_constant_schedule_with_warmup(optimizer, warmup_steps)
    if args.lr_scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)


def _build_training_components(args, dataset, model, collator):
    train_loader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    steps_per_epoch = optimizer_steps_per_epoch(
        len(train_loader), args.gradient_accumulation_steps
    )
    total_steps = steps_per_epoch * args.num_train_epochs
    warmup_steps = (
        int(total_steps * args.warmup_ratio)
        if args.warmup_ratio > 0
        else args.warmup_steps
    )
    logger.info("Training steps: %s (%s warmup)", total_steps, warmup_steps)
    effective_batch = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps
    )
    logger.info("Effective batch size: %s", effective_batch)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )
    scheduler = _build_scheduler(args, optimizer, warmup_steps, total_steps)
    return train_loader, optimizer, scheduler, total_steps


def _build_ewc(args, dataset, model, collator, device):
    if not args.use_ewc:
        return None
    logger.info("%s", "=" * 60)
    logger.info("Setting up EWC Regularization")
    logger.info("  Lambda: %s", args.ewc_lambda)
    logger.info("  Samples: %s", args.ewc_samples)
    logger.info("%s", "=" * 60)
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    return EWCRegularizer(model, loader, device, args.ewc_samples, args.ewc_lambda)


def _forward_loss(args, model, raw_batch, device, window_size: int, ewc):
    raw_batch.pop("_retrieval_count", 0)
    raw_batch.pop("_total_masked", 0)
    batch = {
        key: value.to(device)
        for key, value in raw_batch.items()
        if isinstance(value, torch.Tensor)
    }
    with torch.amp.autocast(
        "cuda", dtype=torch.bfloat16 if args.bf16 else torch.float32
    ):
        loss = model(**batch).loss / window_size
        if ewc is not None:
            loss = loss + ewc.penalty(model) / window_size
    return loss


def _assert_and_save_model(args, model, tokenizer, output_dir: str) -> None:
    if args.rope_scaling_type == "yarn":
        assert_yarn_config(
            model.config,
            original_max_position_embeddings=args.rope_original_max_position_embeddings,
            target_max_position_embeddings=args.model_max_length,
            beta_fast=args.yarn_beta_fast,
            beta_slow=args.yarn_beta_slow,
        )
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def _log_progress(args, state, scheduler, progress) -> None:
    average_loss = state.total_loss / args.logging_steps
    values = {
        "loss": f"{average_loss:.4f}",
        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
    }
    if torch.cuda.is_available():
        values["mem"] = f"{torch.cuda.memory_allocated() / 1e9:.1f}GB"
    progress.set_postfix(values)
    state.total_loss = 0


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
        logger.info("Saving checkpoint: %s", checkpoint)
        _assert_and_save_model(args, model, tokenizer, checkpoint)


def _run_training_loop(
    args, model, tokenizer, loader, optimizer, scheduler, ewc, device
):
    logger.info("Starting training...")
    model.train()
    state = _TrainingState()
    optimizer.zero_grad()
    for epoch in range(args.num_train_epochs):
        logger.info("Epoch %s/%s", epoch + 1, args.num_train_epochs)
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}")
        for step, raw_batch in enumerate(progress):
            window_size = accumulation_window_size(
                step, len(loader), args.gradient_accumulation_steps
            )
            loss = _forward_loss(args, model, raw_batch, device, window_size, ewc)
            loss.backward()
            state.total_loss += loss.item()
            if should_optimizer_step(
                step, len(loader), args.gradient_accumulation_steps
            ):
                _optimizer_step(
                    args,
                    model,
                    tokenizer,
                    optimizer,
                    scheduler,
                    progress,
                    state,
                )
    return state.global_step


def _select_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("GPU Memory: %.1f GB", memory)
    return device


def _save_final(args, model, tokenizer, validation, global_step: int) -> None:
    logger.info("Saving final model: %s", args.output_dir)
    _assert_and_save_model(args, model, tokenizer, args.output_dir)
    write_training_receipt(
        args.output_dir,
        validation=validation,
        model_repo_id=args.model_name_or_path,
        model_revision=args.model_revision,
        dataset_revision=args.dataset_revision,
        status="complete",
        optimizer_steps=global_step,
    )
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as handle:
        json.dump(vars(args), handle, indent=2)
    logger.info("Training complete!")


def train(args) -> None:
    """Execute config-first foundation training without changing its artifacts."""
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _select_device()
    dataset, validation = _validate_dataset(args)
    tokenizer = _load_tokenizer(args)
    model = _load_model(args, tokenizer, device)
    collator = _build_collator(args, tokenizer)
    loader, optimizer, scheduler, total_steps = _build_training_components(
        args, dataset, model, collator
    )
    ewc = _build_ewc(args, dataset, model, collator, device)
    global_step = _run_training_loop(
        args, model, tokenizer, loader, optimizer, scheduler, ewc, device
    )
    if global_step != total_steps:
        raise RuntimeError(
            f"optimizer step count mismatch: {global_step} != planned {total_steps}"
        )
    _save_final(args, model, tokenizer, validation, global_step)
