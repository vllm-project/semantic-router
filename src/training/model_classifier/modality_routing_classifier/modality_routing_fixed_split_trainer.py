"""
Fixed-Split Trainer for the Modality Routing Classifier (Issue #3198 candidate work)
=====================================================================================

Trains directly from export_modality_dataset.py's deterministic, stratified
train.jsonl / validation.jsonl (70/15/15 split, random_state=42) -- NOT from
modality_routing_bert_finetuning_lora.py's own ModalityRoutingDataset builder,
which re-splits its dynamically-assembled pool with a different, non-deterministic
80/20 split and discards the exporter's own held-out test set entirely.

Every model this script trains, trains on the exact complement of test.jsonl.
Do not "fix" this back to calling ModalityRoutingDataset.prepare_datasets() --
that would reintroduce the split mismatch this script exists to avoid.

Two modes, selected by whether --teacher-model-path is passed:

  - No teacher -> plain fine-tune mode. Used to train a leakage-free "clean
    baseline" reproduction of the production mmBERT-32K classifier: trains on
    hard labels only (Focal Loss), on this exporter's own train/val split, so
    it can be compared fairly against a candidate trained on the same split
    (the published production checkpoint's actual training data is NOT
    guaranteed to exclude test.jsonl -- see DECISION_RECORD.md).

  - --teacher-model-path given -> knowledge-distillation mode. Used to train
    the compact student candidate (DistilBERT-base-uncased) against a frozen
    teacher (e.g. the published mmBERT-32K checkpoint), combining a
    temperature-scaled soft-label KL loss (Hinton et al., 2015) with the same
    hard-label Focal Loss used elsewhere in this pipeline.

Usage:
    # Clean baseline (mmBERT-32K, no teacher, hard labels only)
    python modality_routing_fixed_split_trainer.py \\
        --model mmbert-32k \\
        --train-file exported_modality_routing_dataset/train.jsonl \\
        --val-file exported_modality_routing_dataset/validation.jsonl \\
        --output-dir lora_modality_router_mmbert32k_clean_baseline \\
        --merge-output-dir models/mmbert32k-modality-router-clean-merged

    # Candidate (DistilBERT student, distilled from the published baseline)
    python modality_routing_fixed_split_trainer.py \\
        --model distilbert-base-uncased \\
        --teacher-model-path llm-semantic-router/mmbert32k-modality-router-merged \\
        --train-file exported_modality_routing_dataset/train.jsonl \\
        --val-file exported_modality_routing_dataset/validation.jsonl \\
        --temperature 3.0 --kd-alpha 0.5 \\
        --output-dir lora_modality_router_distilbert_candidate \\
        --merge-output-dir models/distilbert-modality-router-candidate-merged
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import Trainer, TrainingArguments

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_THIS_DIR)  # modality_routing_bert_finetuning_lora
sys.path.append(os.path.dirname(_THIS_DIR))  # common_lora_utils

from common_lora_utils import (
    clear_gpu_memory,
    create_lora_config,
    load_sequence_classifier_for_inference,
    log_memory_usage,
    resolve_model_path,
    set_gpu_device,
    setup_logging,
)
from modality_routing_bert_finetuning_lora import (
    MODALITY_LABELS,
    FocalLoss,
    compute_modality_metrics,
    create_lora_modality_routing_model,
    create_tokenizer_for_model,
    merge_lora_adapter_to_full_model,
    recommend_lora_rank,
    tokenize_modality_data,
)

logger = setup_logging()

LABEL_TO_ID = {label: idx for idx, label in enumerate(MODALITY_LABELS)}


def load_jsonl(path: str) -> List[Dict]:
    """Load {"text", "label", "label_name"} rows written by export_modality_dataset.py."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_class_weights_and_focal_gamma(train_data: List[Dict], num_classes: int = 3):
    """
    Mirrors modality_routing_bert_finetuning_lora.main()'s class-weight/focal-gamma
    logic exactly (inverse-frequency weighting with sqrt dampening, clamped [0.5, 3.0];
    focal gamma adapted to imbalance severity), duplicated here since the baseline
    script does not expose it as a standalone function.
    """
    train_labels = [item["label"] for item in train_data]
    label_counts: Dict[int, int] = {}
    for label in train_labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    total = len(train_labels)
    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)
        raw_weight = total / (num_classes * count)
        weights.append(max(0.5, min(raw_weight**0.5, 3.0)))
    class_weights = torch.tensor(weights, dtype=torch.float32)

    max_count = max(label_counts.values())
    min_count = min(label_counts.values())
    imbalance_ratio = max_count / max(min_count, 1)

    logger.info("Class distribution in training data:")
    for i in range(num_classes):
        label_name = MODALITY_LABELS[i] if i < len(MODALITY_LABELS) else f"class_{i}"
        count = label_counts.get(i, 0)
        logger.info(
            f"  {label_name}: {count} ({count / total * 100:.1f}%), weight={weights[i]:.3f}"
        )
    logger.info(f"Imbalance ratio: {imbalance_ratio:.1f}:1")

    if imbalance_ratio > 3.0:
        focal_gamma = 3.0
    elif imbalance_ratio > 1.5:
        focal_gamma = 2.0
    else:
        focal_gamma = 1.5
    logger.info(f"Focal gamma: {focal_gamma}")

    return class_weights, focal_gamma, label_counts


def oversample_minority_classes(train_data: List[Dict], label_counts: Dict[int, int]) -> List[Dict]:
    """Mirrors modality_routing_bert_finetuning_lora.main()'s oversampling logic."""
    max_count = max(label_counts.values())
    min_count = min(label_counts.values())
    if max_count / max(min_count, 1) <= 2.0:
        return train_data

    class_buckets: Dict[int, List[Dict]] = {}
    for item in train_data:
        class_buckets.setdefault(item["label"], []).append(item)

    oversampled: List[Dict] = []
    for label, items in class_buckets.items():
        if len(items) < max_count:
            repeats = max_count // len(items)
            remainder = max_count % len(items)
            oversampled.extend(items * repeats + random.sample(items, remainder))
            logger.info(
                f"  {MODALITY_LABELS[label]}: {len(items)} -> "
                f"{repeats * len(items) + remainder} (oversampled)"
            )
        else:
            oversampled.extend(items)
    random.shuffle(oversampled)
    return oversampled


class FixedSplitFocalTrainer(Trainer):
    """Plain-fine-tune-mode trainer: hard-label Focal Loss only (no teacher)."""

    def __init__(self, class_weights=None, focal_gamma: float = 2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma
        self._focal_loss: Optional[FocalLoss] = None

    def _get_loss_fn(self, device: torch.device) -> FocalLoss:
        if self._focal_loss is None:
            alpha = self.class_weights.to(device) if self.class_weights is not None else None
            self._focal_loss = FocalLoss(alpha=alpha, gamma=self.focal_gamma, reduction="mean")
        return self._focal_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss_fn = self._get_loss_fn(outputs.logits.device)
        loss = loss_fn(outputs.logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class ModalityDistillationTrainer(Trainer):
    """
    Distillation-mode trainer: combines temperature-scaled soft-label KL loss
    (Hinton et al., 2015, with the standard T^2 gradient-scale correction) against
    precomputed frozen-teacher logits, with the same hard-label Focal Loss used
    elsewhere in this pipeline, weighted by kd_alpha.
    """

    def __init__(
        self,
        class_weights=None,
        focal_gamma: float = 2.0,
        temperature: float = 3.0,
        kd_alpha: float = 0.5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma
        self.temperature = temperature
        self.kd_alpha = kd_alpha
        self._focal_loss: Optional[FocalLoss] = None

    def _get_loss_fn(self, device: torch.device) -> FocalLoss:
        if self._focal_loss is None:
            alpha = self.class_weights.to(device) if self.class_weights is not None else None
            self._focal_loss = FocalLoss(alpha=alpha, gamma=self.focal_gamma, reduction="mean")
        return self._focal_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        # Only train_dataset carries "teacher_logits" (precomputed once, up front);
        # val_dataset intentionally doesn't, since eval-set KD loss isn't needed for
        # the graduation gate -- fall back to hard-loss-only during evaluation.
        teacher_logits = inputs.pop("teacher_logits", None)
        outputs = model(**inputs)
        student_logits = outputs.logits

        loss_fn = self._get_loss_fn(student_logits.device)
        hard_loss = loss_fn(student_logits, labels)

        if teacher_logits is None:
            loss = hard_loss
        else:
            T = self.temperature
            soft_loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=-1),
                F.softmax(teacher_logits / T, dim=-1),
                reduction="batchmean",
            ) * (T**2)
            loss = self.kd_alpha * soft_loss + (1 - self.kd_alpha) * hard_loss
        return (loss, outputs) if return_outputs else loss


def compute_teacher_logits(
    teacher_model, teacher_tokenizer, texts: List[str], batch_size: int, max_length: int, device: str
) -> np.ndarray:
    """Frozen-teacher forward pass over `texts`, in the same order, returning raw logits."""
    teacher_model.to(device)
    teacher_model.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = teacher_tokenizer(
                batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
            ).to(device)
            logits = teacher_model(**enc).logits
            all_logits.append(logits.detach().cpu().numpy())
    return np.concatenate(all_logits, axis=0).astype(np.float32)


def main(
    model_name: str,
    train_file: str,
    val_file: str,
    teacher_model_path: Optional[str] = None,
    lora_rank: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    lora_dropout: float = 0.1,
    num_epochs: int = 8,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    temperature: float = 3.0,
    kd_alpha: float = 0.5,
    max_length: int = 256,
    output_dir: Optional[str] = None,
    merge_output_dir: Optional[str] = None,
    gpu_id: Optional[int] = None,
    use_class_weights: bool = True,
):
    mode = "distillation" if teacher_model_path else "plain fine-tune"
    logger.info("=" * 70)
    logger.info(f"Fixed-Split Modality Routing Trainer ({mode} mode)")
    logger.info(f"  Student model: {model_name}")
    logger.info(f"  Teacher model: {teacher_model_path or 'N/A'}")
    logger.info(f"  Train file: {train_file}")
    logger.info(f"  Val file: {val_file}")
    logger.info("=" * 70)

    if gpu_id is not None:
        device, _ = set_gpu_device(gpu_id=gpu_id, auto_select=False)
    else:
        device, _ = set_gpu_device(gpu_id=None, auto_select=True)
    clear_gpu_memory()
    log_memory_usage("Pre-training")

    train_data = load_jsonl(train_file)
    val_data = load_jsonl(val_file)
    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")

    model_path = resolve_model_path(model_name)
    logger.info(f"Using model: {model_name} -> {model_path}")

    effective_rank = recommend_lora_rank(
        num_train_samples=len(train_data), num_classes=len(MODALITY_LABELS), user_rank=lora_rank
    )
    effective_alpha = lora_alpha if lora_alpha is not None else 2 * effective_rank
    lora_config = create_lora_config(model_name, effective_rank, effective_alpha, lora_dropout)

    class_weights = None
    focal_gamma = 2.0
    if use_class_weights:
        class_weights, focal_gamma, label_counts = compute_class_weights_and_focal_gamma(
            train_data, num_classes=len(MODALITY_LABELS)
        )
        train_data = oversample_minority_classes(train_data, label_counts)
        logger.info(f"Training set after oversampling: {len(train_data)} samples")

    model, tokenizer = create_lora_modality_routing_model(model_path, len(MODALITY_LABELS), lora_config)

    train_dataset = tokenize_modality_data(train_data, tokenizer, max_length=max_length)
    val_dataset = tokenize_modality_data(val_data, tokenizer, max_length=max_length)

    if output_dir is None:
        output_dir = f"lora_modality_router_{model_name}_fixed_split_r{effective_rank}_model"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Model will be saved to: {output_dir}")

    # transformers >=5.0 dropped TrainingArguments(warmup_ratio=...) entirely (only
    # warmup_steps remains) -- the baseline script's own `warmup_ratio=0.06` would hit
    # the same TypeError on that version despite its requirements.txt allowing it
    # (transformers>=4.40.0, no upper bound). Compute the equivalent step count
    # ourselves so this works across both old and new transformers releases.
    gradient_accumulation_steps = 2
    steps_per_epoch = -(-len(train_dataset) // (batch_size * gradient_accumulation_steps))
    warmup_steps = max(1, round(0.06 * steps_per_epoch * num_epochs))

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=3,
        report_to=[],
        fp16=torch.cuda.is_available(),
        dataloader_drop_last=False,
        eval_accumulation_steps=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # Distillation mode's train_dataset carries a "teacher_logits" column that
        # AutoModelForSequenceClassification.forward() doesn't accept -- the default
        # remove_unused_columns=True silently drops it before compute_loss ever sees
        # it (a well-known HF Trainer gotcha), so disable that filtering here.
        remove_unused_columns=False,
    )

    if teacher_model_path:
        logger.info(f"Loading frozen teacher from: {teacher_model_path}")
        teacher_model, teacher_tokenizer, _ = load_sequence_classifier_for_inference(
            teacher_model_path, num_labels=len(MODALITY_LABELS)
        )
        train_texts = [item["text"] for item in train_data]
        teacher_logits = compute_teacher_logits(
            teacher_model, teacher_tokenizer, train_texts, batch_size, max_length, device
        )
        train_dataset = train_dataset.add_column("teacher_logits", teacher_logits.tolist())
        del teacher_model
        clear_gpu_memory()

        trainer = ModalityDistillationTrainer(
            class_weights=class_weights,
            focal_gamma=focal_gamma,
            temperature=temperature,
            kd_alpha=kd_alpha,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_modality_metrics,
        )
    else:
        trainer = FixedSplitFocalTrainer(
            class_weights=class_weights,
            focal_gamma=focal_gamma,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_modality_metrics,
        )

    logger.info("Starting training...")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    label_mapping_data = {
        "label_to_idx": LABEL_TO_ID,
        "idx_to_label": {str(v): k for k, v in LABEL_TO_ID.items()},
    }
    with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
        json.dump(label_mapping_data, f, indent=2)
    with open(os.path.join(output_dir, "modality_mapping.json"), "w") as f:
        json.dump(label_mapping_data, f, indent=2)
    with open(os.path.join(output_dir, "lora_config.json"), "w") as f:
        json.dump(lora_config, f, indent=2)

    logger.info("Final evaluation on validation set...")
    eval_results = trainer.evaluate()
    logger.info(f"  Accuracy: {eval_results['eval_accuracy']:.4f}")
    logger.info(f"  F1 (weighted): {eval_results['eval_f1']:.4f}")
    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(
            {k: float(v) for k, v in eval_results.items() if isinstance(v, (int, float))},
            f,
            indent=2,
        )

    logger.info(f"Model saved to: {output_dir} (LoRA adapter, base model kept separate)")

    if merge_output_dir:
        logger.info(f"Merging LoRA adapter into a servable checkpoint at: {merge_output_dir}")
        merge_lora_adapter_to_full_model(output_dir, merge_output_dir, model_path)
        logger.info(f"Merged, servable checkpoint ready at: {merge_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fixed-split modality routing trainer: plain fine-tune (clean "
        "baseline) or knowledge distillation (candidate), both trained on "
        "export_modality_dataset.py's deterministic train/validation split."
    )
    parser.add_argument(
        "--model",
        choices=["mmbert-32k", "distilbert-base-uncased"],
        required=True,
        help="Student/model to train. mmbert-32k for a clean baseline reproduction; "
        "distilbert-base-uncased for the candidate.",
    )
    parser.add_argument(
        "--teacher-model-path",
        type=str,
        default=None,
        help="Local dir or HF Hub repo id of a frozen teacher checkpoint. Omit for "
        "plain fine-tune mode (clean baseline); set for distillation mode (candidate).",
    )
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--val-file", type=str, required=True)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--temperature", type=float, default=3.0, help="KD softmax temperature (distillation mode only)."
    )
    parser.add_argument(
        "--kd-alpha",
        type=float,
        default=0.5,
        help="Weight on the soft-label KD loss vs. hard-label loss (distillation mode only).",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--merge-output-dir",
        type=str,
        default=None,
        help="If set, merge the trained LoRA adapter into a full servable checkpoint here.",
    )
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--no-class-weights", action="store_true")

    args = parser.parse_args()
    main(
        model_name=args.model,
        train_file=args.train_file,
        val_file=args.val_file,
        teacher_model_path=args.teacher_model_path,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        kd_alpha=args.kd_alpha,
        max_length=args.max_length,
        output_dir=args.output_dir,
        merge_output_dir=args.merge_output_dir,
        gpu_id=args.gpu_id,
        use_class_weights=not args.no_class_weights,
    )
