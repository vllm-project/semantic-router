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

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812  (PyTorch convention)
from transformers import Trainer, TrainingArguments, set_seed

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_THIS_DIR)  # modality_routing_bert_finetuning_lora
sys.path.append(os.path.dirname(_THIS_DIR))  # common_lora_utils

from common_lora_utils import (  # noqa: E402
    clear_gpu_memory,
    create_lora_config,
    load_sequence_classifier_for_inference,
    log_memory_usage,
    resolve_model_path,
    set_gpu_device,
    setup_logging,
)
from modality_label_mapping import (  # noqa: E402
    build_label_remap,
    check_output_size,
    logits_to_canonical_order,
)
from modality_routing_bert_finetuning_lora import (  # noqa: E402
    MODALITY_LABELS,
    FocalLoss,
    compute_modality_metrics,
    create_lora_modality_routing_model,
    merge_lora_adapter_to_full_model,
    recommend_lora_rank,
    tokenize_modality_data,
)

logger = setup_logging()

LABEL_TO_ID = {label: idx for idx, label in enumerate(MODALITY_LABELS)}

# Same thresholds as modality_routing_bert_finetuning_lora.main().
SEVERE_IMBALANCE_RATIO = 3.0
MILD_IMBALANCE_RATIO = 1.5
OVERSAMPLE_IMBALANCE_RATIO = 2.0
GRADIENT_ACCUMULATION_STEPS = 2
WARMUP_RATIO = 0.06


def load_jsonl(path: str) -> list[dict]:
    """Load rows written by export_modality_dataset.py.

    Args:
        path: Path to a JSONL file with one {"text", "label", "label_name"} object per line.

    Returns:
        The parsed rows, in file order.
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_class_weights_and_focal_gamma(train_data: list[dict], num_classes: int = 3):
    """Compute class weights and the focal-loss gamma from the training labels.

    Mirrors modality_routing_bert_finetuning_lora.main() exactly: inverse-frequency
    weights with sqrt dampening, clamped to [0.5, 3.0], and a gamma chosen from the
    imbalance ratio. Duplicated here because the baseline script does not expose it
    as a function.

    Args:
        train_data: Training rows with an integer "label".
        num_classes: Number of classes.

    Returns:
        (class_weights, focal_gamma, label_counts).
    """
    train_labels = [item["label"] for item in train_data]
    label_counts: dict[int, int] = {}
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

    if imbalance_ratio > SEVERE_IMBALANCE_RATIO:
        focal_gamma = 3.0
    elif imbalance_ratio > MILD_IMBALANCE_RATIO:
        focal_gamma = 2.0
    else:
        focal_gamma = 1.5
    logger.info(f"Focal gamma: {focal_gamma}")

    return class_weights, focal_gamma, label_counts


def oversample_minority_classes(
    train_data: list[dict], label_counts: dict[int, int]
) -> list[dict]:
    """Repeat minority-class rows until every class matches the largest one.

    Mirrors the oversampling in modality_routing_bert_finetuning_lora.main(). Does
    nothing if the imbalance ratio is 2.0 or less. Uses the `random` module, so seed
    it first (main() does) for a reproducible result.

    Args:
        train_data: Training rows with an integer "label".
        label_counts: Number of rows per label id in train_data.

    Returns:
        The rows to train on, shuffled if any class was oversampled.
    """
    max_count = max(label_counts.values())
    min_count = min(label_counts.values())
    if max_count / max(min_count, 1) <= OVERSAMPLE_IMBALANCE_RATIO:
        return train_data

    class_buckets: dict[int, list[dict]] = {}
    for item in train_data:
        class_buckets.setdefault(item["label"], []).append(item)

    oversampled: list[dict] = []
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
    """Plain fine-tune trainer: hard-label Focal Loss only, no teacher."""

    def __init__(self, class_weights=None, focal_gamma: float = 2.0, *args, **kwargs):
        """Store the loss settings and initialise the Trainer.

        Args:
            class_weights: Per-class weights for the Focal Loss, or None.
            focal_gamma: Focusing parameter of the Focal Loss.
            *args: Passed to transformers.Trainer.
            **kwargs: Passed to transformers.Trainer.
        """
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma
        self._focal_loss: FocalLoss | None = None

    def _get_loss_fn(self, device: torch.device) -> FocalLoss:
        """Build the Focal Loss on first use, with the weights on the model's device.

        Args:
            device: Device the model's logits are on.

        Returns:
            The cached FocalLoss instance.
        """
        if self._focal_loss is None:
            alpha = (
                self.class_weights.to(device)
                if self.class_weights is not None
                else None
            )
            self._focal_loss = FocalLoss(
                alpha=alpha, gamma=self.focal_gamma, reduction="mean"
            )
        return self._focal_loss

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Compute the Focal Loss of the model's logits against the hard labels.

        Args:
            model: The model being trained.
            inputs: Batch with input_ids, attention_mask and labels.
            return_outputs: Whether to also return the model outputs.
            num_items_in_batch: Unused; kept for the Trainer interface.

        Returns:
            The loss, or (loss, outputs) if return_outputs is set.
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss_fn = self._get_loss_fn(outputs.logits.device)
        loss = loss_fn(
            outputs.logits.view(-1, self.model.config.num_labels), labels.view(-1)
        )
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
        """Store the loss settings and initialise the Trainer.

        Args:
            class_weights: Per-class weights for the hard-label Focal Loss, or None.
            focal_gamma: Focusing parameter of the Focal Loss.
            temperature: Softmax temperature for the soft-label KL term.
            kd_alpha: Weight of the soft-label term against the hard-label term.
            *args: Passed to transformers.Trainer.
            **kwargs: Passed to transformers.Trainer.
        """
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma
        self.temperature = temperature
        self.kd_alpha = kd_alpha
        self._focal_loss: FocalLoss | None = None

    def _get_loss_fn(self, device: torch.device) -> FocalLoss:
        """Build the Focal Loss on first use, with the weights on the model's device.

        Args:
            device: Device the model's logits are on.

        Returns:
            The cached FocalLoss instance.
        """
        if self._focal_loss is None:
            alpha = (
                self.class_weights.to(device)
                if self.class_weights is not None
                else None
            )
            self._focal_loss = FocalLoss(
                alpha=alpha, gamma=self.focal_gamma, reduction="mean"
            )
        return self._focal_loss

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Combine the soft-label KL loss with the hard-label Focal Loss.

        Without teacher logits in the batch (the validation set) it returns the
        hard-label loss only.

        Args:
            model: The student model being trained.
            inputs: Batch with input_ids, attention_mask, labels and, when training,
                teacher_logits in canonical label order.
            return_outputs: Whether to also return the model outputs.
            num_items_in_batch: Unused; kept for the Trainer interface.

        Returns:
            The loss, or (loss, outputs) if return_outputs is set.
        """
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
            temp = self.temperature
            soft_loss = F.kl_div(
                F.log_softmax(student_logits / temp, dim=-1),
                F.softmax(teacher_logits / temp, dim=-1),
                reduction="batchmean",
            ) * (temp**2)
            loss = self.kd_alpha * soft_loss + (1 - self.kd_alpha) * hard_loss
        return (loss, outputs) if return_outputs else loss


def compute_teacher_logits(
    teacher_model,
    teacher_tokenizer,
    texts: list[str],
    *,
    batch_size: int,
    max_length: int,
    device: str,
) -> np.ndarray:
    """Run the frozen teacher over the texts and return its raw logits.

    The columns are in the teacher's own class order. Reorder them with
    logits_to_canonical_order before using them as targets.

    Args:
        teacher_model: The teacher classifier.
        teacher_tokenizer: The teacher's tokenizer.
        texts: Prompt texts, in the order of the training set.
        batch_size: Number of texts per forward pass.
        max_length: Token limit; longer texts are truncated.
        device: Device to run the teacher on.

    Returns:
        Array of shape [N, C] with float32 logits.
    """
    teacher_model.to(device)
    teacher_model.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = teacher_tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = teacher_model(**enc).logits
            all_logits.append(logits.detach().cpu().numpy())
    return np.concatenate(all_logits, axis=0).astype(np.float32)


def build_training_args(
    *,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    num_train_rows: int,
    seed: int,
) -> TrainingArguments:
    """Build the TrainingArguments shared by both modes.

    transformers >=5.15 removed TrainingArguments(warmup_ratio=...) and
    logging_dir (only warmup_steps remains; 5.14.1 still accepts both). The
    baseline script's own `warmup_ratio=0.06` hits the same TypeError on that
    version despite its requirements.txt allowing it (transformers>=4.40.0, no
    upper bound). The equivalent step count is computed here so this works across
    both old and new transformers releases.

    Args:
        output_dir: Where the Trainer writes checkpoints.
        num_epochs: Number of training epochs.
        batch_size: Per-device batch size.
        learning_rate: Peak learning rate.
        num_train_rows: Number of rows in the (oversampled) training set.
        seed: Seed for the Trainer and the data order.

    Returns:
        The TrainingArguments for the run.
    """
    steps_per_epoch = -(-num_train_rows // (batch_size * GRADIENT_ACCUMULATION_STEPS))
    warmup_steps = max(1, round(WARMUP_RATIO * steps_per_epoch * num_epochs))
    return TrainingArguments(
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
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        # Distillation mode's train_dataset carries a "teacher_logits" column that
        # AutoModelForSequenceClassification.forward() doesn't accept -- the default
        # remove_unused_columns=True silently drops it before compute_loss ever sees
        # it (a well-known HF Trainer gotcha), so disable that filtering here.
        remove_unused_columns=False,
        seed=seed,
        data_seed=seed,
    )


def add_teacher_logits(
    train_dataset,
    train_texts: list[str],
    teacher_model_path: str,
    *,
    batch_size: int,
    max_length: int,
    device: str,
):
    """Add the frozen teacher's logits to the training set as a column.

    The student is trained in canonical label order, so the teacher's logits are
    reordered to it too. This fails closed on a teacher whose labels do not map onto
    AR/DIFFUSION/BOTH, instead of distilling from the wrong columns.

    Args:
        train_dataset: Tokenized training set.
        train_texts: Prompt texts in the same order as train_dataset.
        teacher_model_path: Local directory or Hub repo id of the teacher.
        batch_size: Number of texts per teacher forward pass.
        max_length: Token limit; longer texts are truncated.
        device: Device to run the teacher on.

    Returns:
        train_dataset with an added "teacher_logits" column.

    Raises:
        ValueError: If the teacher's labels do not map onto the canonical labels.
    """
    logger.info(f"Loading frozen teacher from: {teacher_model_path}")
    teacher_model, teacher_tokenizer, teacher_id2label = (
        load_sequence_classifier_for_inference(
            teacher_model_path, num_labels=len(MODALITY_LABELS)
        )
    )
    teacher_remap = build_label_remap(teacher_id2label, teacher_model_path)
    check_output_size(
        teacher_remap, teacher_model.config.num_labels, teacher_model_path
    )
    logger.info(
        f"Teacher label mapping (checkpoint id -> canonical id): {teacher_remap}"
    )
    teacher_logits = logits_to_canonical_order(
        compute_teacher_logits(
            teacher_model,
            teacher_tokenizer,
            train_texts,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        ),
        teacher_remap,
    )
    del teacher_model
    clear_gpu_memory()
    return train_dataset.add_column("teacher_logits", teacher_logits.tolist())


def save_run_artifacts(model, tokenizer, trainer, output_dir: str, lora_config) -> None:
    """Save the adapter, tokenizer, label mappings, LoRA config and final eval.

    Args:
        model: The trained model (LoRA adapter).
        tokenizer: The student's tokenizer.
        trainer: The finished Trainer, used for the final validation pass.
        output_dir: Directory to write into.
        lora_config: The LoRA configuration the run used.
    """
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    label_mapping_data = {
        "label_to_idx": LABEL_TO_ID,
        "idx_to_label": {str(v): k for k, v in LABEL_TO_ID.items()},
    }
    for name, payload in (
        ("label_mapping.json", label_mapping_data),
        ("modality_mapping.json", label_mapping_data),
        ("lora_config.json", lora_config),
    ):
        with open(os.path.join(output_dir, name), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    logger.info("Final evaluation on validation set...")
    eval_results = trainer.evaluate()
    logger.info(f"  Accuracy: {eval_results['eval_accuracy']:.4f}")
    logger.info(f"  F1 (weighted): {eval_results['eval_f1']:.4f}")
    with open(
        os.path.join(output_dir, "eval_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                k: float(v)
                for k, v in eval_results.items()
                if isinstance(v, (int, float))
            },
            f,
            indent=2,
        )


def log_run_header(
    model_name: str,
    teacher_model_path: str | None,
    train_file: str,
    val_file: str,
    seed: int,
) -> None:
    """Log the mode and inputs of the run.

    Args:
        model_name: Student model name.
        teacher_model_path: Teacher checkpoint, or None for plain fine-tuning.
        train_file: Path to train.jsonl.
        val_file: Path to validation.jsonl.
        seed: Seed for the run.
    """
    mode = "distillation" if teacher_model_path else "plain fine-tune"
    logger.info("=" * 70)
    logger.info(f"Fixed-Split Modality Routing Trainer ({mode} mode)")
    logger.info(f"  Student model: {model_name}")
    logger.info(f"  Teacher model: {teacher_model_path or 'N/A'}")
    logger.info(f"  Train file: {train_file}")
    logger.info(f"  Val file: {val_file}")
    logger.info(f"  Seed: {seed}")
    logger.info("=" * 70)


def main(
    model_name: str,
    train_file: str,
    val_file: str,
    *,
    teacher_model_path: str | None = None,
    lora_rank: int | None = None,
    lora_alpha: int | None = None,
    lora_dropout: float = 0.1,
    num_epochs: int = 8,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    temperature: float = 3.0,
    kd_alpha: float = 0.5,
    max_length: int = 256,
    output_dir: str | None = None,
    merge_output_dir: str | None = None,
    gpu_id: int | None = None,
    use_class_weights: bool = True,
    seed: int = 42,
):
    """Train the student in plain fine-tune or distillation mode.

    Distillation mode is used when teacher_model_path is set. Both modes train on
    the exporter's fixed train/validation files and never see the test set.

    Args:
        model_name: Student to train, "mmbert-32k" or "distilbert-base-uncased".
        train_file: Path to train.jsonl.
        val_file: Path to validation.jsonl.
        teacher_model_path: Local directory or Hub repo id of the frozen teacher, or
            None for plain fine-tuning.
        lora_rank: LoRA rank, or None to pick one from the training set size.
        lora_alpha: LoRA alpha, or None for twice the rank.
        lora_dropout: LoRA dropout.
        num_epochs: Number of training epochs.
        batch_size: Per-device batch size.
        learning_rate: Peak learning rate.
        temperature: Softmax temperature for the soft-label term (distillation only).
        kd_alpha: Weight of the soft-label term (distillation only).
        max_length: Token limit; longer texts are truncated.
        output_dir: Where to save the LoRA adapter, or None for a default name.
        merge_output_dir: If set, also write a merged, servable checkpoint here.
        gpu_id: GPU to use, or None to pick a free one.
        use_class_weights: Whether to weight and oversample by class frequency.
        seed: Seed for LoRA and head initialisation, oversampling and data order.
    """
    log_run_header(model_name, teacher_model_path, train_file, val_file, seed)

    # Seed before anything random happens: the LoRA adapter and the classifier head
    # are initialised when the model is created (before the Trainer exists, so before
    # the Trainer's own seeding), and oversampling uses the `random` module.
    set_seed(seed)

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
        num_train_samples=len(train_data),
        num_classes=len(MODALITY_LABELS),
        user_rank=lora_rank,
    )
    effective_alpha = lora_alpha if lora_alpha is not None else 2 * effective_rank
    lora_config = create_lora_config(
        model_name, effective_rank, effective_alpha, lora_dropout
    )

    class_weights = None
    focal_gamma = 2.0
    if use_class_weights:
        class_weights, focal_gamma, label_counts = (
            compute_class_weights_and_focal_gamma(
                train_data, num_classes=len(MODALITY_LABELS)
            )
        )
        train_data = oversample_minority_classes(train_data, label_counts)
        logger.info(f"Training set after oversampling: {len(train_data)} samples")

    model, tokenizer = create_lora_modality_routing_model(
        model_path, len(MODALITY_LABELS), lora_config
    )

    train_dataset = tokenize_modality_data(train_data, tokenizer, max_length=max_length)
    val_dataset = tokenize_modality_data(val_data, tokenizer, max_length=max_length)

    if output_dir is None:
        output_dir = (
            f"lora_modality_router_{model_name}_fixed_split_r{effective_rank}_model"
        )
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Model will be saved to: {output_dir}")

    training_args = build_training_args(
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_rows=len(train_dataset),
        seed=seed,
    )

    if teacher_model_path:
        train_dataset = add_teacher_logits(
            train_dataset,
            [item["text"] for item in train_data],
            teacher_model_path,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )

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

    save_run_artifacts(model, tokenizer, trainer, output_dir, lora_config)

    logger.info(
        f"Model saved to: {output_dir} (LoRA adapter, base model kept separate)"
    )

    if merge_output_dir:
        logger.info(
            f"Merging LoRA adapter into a servable checkpoint at: {merge_output_dir}"
        )
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
        "--temperature",
        type=float,
        default=3.0,
        help="KD softmax temperature (distillation mode only).",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for LoRA/head init, oversampling and data order. Runs differ by seed, so compare models over several seeds before reading a small gap as real.",
    )

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
        seed=args.seed,
    )
