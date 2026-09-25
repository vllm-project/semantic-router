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
from dataclasses import dataclass

import numpy as np
import torch
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
from modality_data import (  # noqa: E402
    ClassStats,
    compute_class_stats,
    load_jsonl,
    oversample_minority_classes,
)
from modality_label_mapping import (  # noqa: E402
    LABEL_TO_ID,
    MODALITY_LABELS,
    build_label_remap,
    check_output_size,
    logits_to_canonical_order,
)
from modality_losses import DistillationSettings, distillation_loss  # noqa: E402
from modality_routing_bert_finetuning_lora import (  # noqa: E402
    FocalLoss,
    compute_modality_metrics,
    create_lora_modality_routing_model,
    merge_lora_adapter_to_full_model,
    recommend_lora_rank,
    tokenize_modality_data,
)
from training_args_compat import create_training_arguments  # noqa: E402

logger = setup_logging()

GRADIENT_ACCUMULATION_STEPS = 2
WARMUP_RATIO = 0.06
DEFAULT_FOCAL_GAMMA = 2.0


@dataclass(frozen=True)
class TrainConfig:
    """Everything a training run needs, so main() takes one argument.

    Attributes:
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

    model_name: str
    train_file: str
    val_file: str
    teacher_model_path: str | None = None
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float = 0.1
    num_epochs: int = 8
    batch_size: int = 32
    learning_rate: float = 2e-5
    temperature: float = 3.0
    kd_alpha: float = 0.5
    max_length: int = 256
    output_dir: str | None = None
    merge_output_dir: str | None = None
    gpu_id: int | None = None
    use_class_weights: bool = True
    seed: int = 42

    @property
    def distillation(self) -> DistillationSettings | None:
        """Distillation settings, or None when there is no teacher."""
        if not self.teacher_model_path:
            return None
        return DistillationSettings(self.temperature, self.kd_alpha)


class ModalityTrainer(Trainer):
    """Trainer with a Focal Loss on the hard labels and optional distillation.

    Batches that carry a "teacher_logits" column (the training set in distillation
    mode) also get the soft-label term. Batches without one, which is every batch in
    plain fine-tune mode and the validation set in both modes, use the hard-label
    loss only.
    """

    def __init__(
        self,
        *args,
        class_weights: list[float] | None = None,
        focal_gamma: float = DEFAULT_FOCAL_GAMMA,
        distillation: DistillationSettings | None = None,
        **kwargs,
    ):
        """Store the loss settings and initialise the Trainer.

        Args:
            *args: Passed to transformers.Trainer.
            class_weights: Per-class weights for the Focal Loss, or None.
            focal_gamma: Focusing parameter of the Focal Loss.
            distillation: Distillation settings, or None for hard labels only.
            **kwargs: Passed to transformers.Trainer.
        """
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma
        self.distillation = distillation
        self._focal_loss: FocalLoss | None = None

    def _hard_loss_fn(self, device: torch.device) -> FocalLoss:
        """Build the Focal Loss on first use, with the weights on the model's device.

        Args:
            device: Device the model's logits are on.

        Returns:
            The cached FocalLoss instance.
        """
        if self._focal_loss is None:
            alpha = (
                torch.tensor(self.class_weights, dtype=torch.float32, device=device)
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
        """Compute the hard-label loss, plus the soft-label term when teacher logits are present.

        Args:
            model: The model being trained.
            inputs: Batch with input_ids, attention_mask, labels and, when training
                in distillation mode, teacher_logits in canonical label order.
            return_outputs: Whether to also return the model outputs.
            num_items_in_batch: Unused; kept for the Trainer interface.

        Returns:
            The loss, or (loss, outputs) if return_outputs is set.
        """
        labels = inputs.pop("labels")
        teacher_logits = inputs.pop("teacher_logits", None)
        outputs = model(**inputs)
        logits = outputs.logits

        loss = self._hard_loss_fn(logits.device)(logits, labels)
        if teacher_logits is not None and self.distillation is not None:
            loss = distillation_loss(logits, teacher_logits, loss, self.distillation)
        return (loss, outputs) if return_outputs else loss


def log_run_header(config: TrainConfig) -> None:
    """Log the mode and inputs of the run.

    Args:
        config: The run configuration.
    """
    mode = "distillation" if config.teacher_model_path else "plain fine-tune"
    logger.info("=" * 70)
    logger.info(f"Fixed-Split Modality Routing Trainer ({mode} mode)")
    logger.info(f"  Student model: {config.model_name}")
    logger.info(f"  Teacher model: {config.teacher_model_path or 'N/A'}")
    logger.info(f"  Train file: {config.train_file}")
    logger.info(f"  Val file: {config.val_file}")
    logger.info(f"  Seed: {config.seed}")
    logger.info("=" * 70)


def log_class_stats(stats: ClassStats) -> None:
    """Log the class balance and the loss settings derived from it.

    Args:
        stats: Statistics of the training set.
    """
    total = sum(stats.label_counts.values())
    logger.info("Class distribution in training data:")
    for i, label in enumerate(MODALITY_LABELS):
        count = stats.label_counts.get(i, 0)
        logger.info(
            f"  {label}: {count} ({count / total * 100:.1f}%), "
            f"weight={stats.class_weights[i]:.3f}"
        )
    logger.info(f"Imbalance ratio: {stats.imbalance_ratio:.1f}:1")
    logger.info(f"Focal gamma: {stats.focal_gamma}")


def prepare_training_rows(
    config: TrainConfig,
) -> tuple[list[dict], list[dict], ClassStats | None]:
    """Load the fixed train and validation splits and balance the training rows.

    Args:
        config: The run configuration.

    Returns:
        (training rows after any oversampling, validation rows, class statistics or
        None when class weighting is off).
    """
    train_rows = load_jsonl(config.train_file)
    val_rows = load_jsonl(config.val_file)
    logger.info(f"Training samples: {len(train_rows)}")
    logger.info(f"Validation samples: {len(val_rows)}")
    if not config.use_class_weights:
        return train_rows, val_rows, None

    stats = compute_class_stats(
        [row["label"] for row in train_rows], num_classes=len(MODALITY_LABELS)
    )
    log_class_stats(stats)
    train_rows = oversample_minority_classes(
        train_rows, stats.label_counts, random.Random(config.seed)
    )
    logger.info(f"Training set after oversampling: {len(train_rows)} samples")
    return train_rows, val_rows, stats


def build_training_args(config: TrainConfig, output_dir: str) -> TrainingArguments:
    """Build the TrainingArguments shared by both modes.

    Args:
        config: The run configuration.
        output_dir: Where the Trainer writes checkpoints.

    Returns:
        The TrainingArguments for the run.
    """
    # Shared with the other training scripts: transformers >=5.15 removed
    # warmup_ratio, and this helper maps it onto warmup_steps (a fraction there).
    return create_training_arguments(
        TrainingArguments,
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
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
        seed=config.seed,
        data_seed=config.seed,
    )


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
            all_logits.append(teacher_model(**enc).logits.detach().cpu().numpy())
    return np.concatenate(all_logits, axis=0).astype(np.float32)


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


def save_run_artifacts(
    model, tokenizer, trainer: Trainer, output_dir: str, lora_config
) -> None:
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
    numeric = {
        k: float(v) for k, v in eval_results.items() if isinstance(v, (int, float))
    }
    with open(
        os.path.join(output_dir, "eval_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(numeric, f, indent=2)


def main(config: TrainConfig) -> None:
    """Train the student in plain fine-tune or distillation mode.

    Distillation mode is used when config.teacher_model_path is set. Both modes train
    on the exporter's fixed train/validation files and never see the test set.

    Args:
        config: The run configuration.
    """
    log_run_header(config)
    # Seed before anything random happens: the LoRA adapter and the classifier head
    # are initialised when the model is created (before the Trainer exists, so before
    # the Trainer's own seeding).
    set_seed(config.seed)
    device, _ = set_gpu_device(gpu_id=config.gpu_id, auto_select=config.gpu_id is None)
    clear_gpu_memory()
    log_memory_usage("Pre-training")

    train_rows, val_rows, stats = prepare_training_rows(config)
    model_path = resolve_model_path(config.model_name)
    logger.info(f"Using model: {config.model_name} -> {model_path}")

    rank = recommend_lora_rank(
        num_train_samples=len(train_rows),
        num_classes=len(MODALITY_LABELS),
        user_rank=config.lora_rank,
    )
    lora_config = create_lora_config(
        config.model_name,
        rank,
        config.lora_alpha if config.lora_alpha is not None else 2 * rank,
        config.lora_dropout,
    )
    model, tokenizer = create_lora_modality_routing_model(
        model_path, len(MODALITY_LABELS), lora_config
    )
    train_dataset = tokenize_modality_data(
        train_rows, tokenizer, max_length=config.max_length
    )
    val_dataset = tokenize_modality_data(
        val_rows, tokenizer, max_length=config.max_length
    )
    if config.teacher_model_path:
        train_dataset = add_teacher_logits(
            train_dataset,
            [row["text"] for row in train_rows],
            config.teacher_model_path,
            batch_size=config.batch_size,
            max_length=config.max_length,
            device=device,
        )

    output_dir = config.output_dir or (
        f"lora_modality_router_{config.model_name}_fixed_split_r{rank}_model"
    )
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Model will be saved to: {output_dir}")

    trainer = ModalityTrainer(
        class_weights=stats.class_weights if stats else None,
        focal_gamma=stats.focal_gamma if stats else DEFAULT_FOCAL_GAMMA,
        distillation=config.distillation,
        model=model,
        args=build_training_args(config, output_dir),
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
    if config.merge_output_dir:
        logger.info(
            f"Merging LoRA adapter into a servable checkpoint at: {config.merge_output_dir}"
        )
        merge_lora_adapter_to_full_model(
            output_dir, config.merge_output_dir, model_path
        )
        logger.info(f"Merged, servable checkpoint ready at: {config.merge_output_dir}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The parser for the trainer's options.
    """
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
        help="Seed for LoRA/head init, oversampling and data order. Runs differ by "
        "seed, so compare models over several seeds before reading a small gap as real.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    """Turn parsed command-line arguments into a TrainConfig.

    Args:
        args: Result of build_parser().parse_args().

    Returns:
        The run configuration.
    """
    return TrainConfig(
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


if __name__ == "__main__":
    main(config_from_args(build_parser().parse_args()))
