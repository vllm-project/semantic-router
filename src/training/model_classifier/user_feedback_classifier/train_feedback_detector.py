#!/usr/bin/env python3
"""
Feedback Detector Training Script

Trains a 4-class user satisfaction classifier compatible with:
https://huggingface.co/llm-semantic-router/feedback-detector

Labels:
  - SAT: User is satisfied
  - NEED_CLARIFICATION: User needs more explanation
  - WRONG_ANSWER: System provided incorrect information
  - WANT_DIFFERENT: User wants alternative options

Supports both full fine-tuning and LoRA training.

Optimized Hyperparameters (validated 2026-02-02):
  - LoRA rank: 64 (higher capacity needed for 4-class distinction)
  - LoRA alpha: 128 (2x rank recommended)
  - Epochs: 10 (with early stopping patience=3)
  - Learning rate: 2e-5
  - Batch size: 16
  - Historical validation overlapped training SAT templates; those scores are
    not a generalization claim. New runs require explicit isolated validation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

if __package__:
    from .data_contract import (
        ID2LABEL,
        LABEL2ID,
        balanced_class_weights,
        merged_output_directory,
        parse_example,
        validate_splits,
        write_label_mapping,
    )
else:
    from data_contract import (
        ID2LABEL,
        LABEL2ID,
        balanced_class_weights,
        merged_output_directory,
        parse_example,
        validate_splits,
        write_label_mapping,
    )

# Optional LoRA imports
try:
    from peft import LoraConfig, TaskType, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print(
        "Warning: PEFT not installed. LoRA training unavailable. Install with: pip install peft"
    )


NUM_LABELS = len(LABEL2ID)


def load_feedback_data(
    data_source: str, max_samples: int | None = None, revision: str | None = None
):
    """Load explicit train/validation splits without synthetic or train fallbacks."""
    result = []
    local = Path(data_source)
    if not local.exists():
        if not revision:
            raise ValueError(
                "Remote feedback datasets require a fixed --dataset_revision"
            )
        dataset = load_dataset(data_source, revision=revision)
        for split in ("train", "validation"):
            if split not in dataset:
                raise ValueError(f"Dataset is missing the {split} split")
            rows = list(dataset[split])
            result.append([parse_example(row) for row in rows[:max_samples]])
    else:
        for split, names in (
            ("train", ("train.jsonl",)),
            ("validation", ("validation.jsonl", "val.jsonl")),
        ):
            path = next(
                (local / name for name in names if (local / name).is_file()), None
            )
            if path is None:
                raise FileNotFoundError(f"Missing explicit {split} JSONL in {local}")
            with path.open(encoding="utf-8") as handle:
                rows = [
                    parse_example(json.loads(line)) for line in handle if line.strip()
                ]
            result.append(rows[:max_samples])
    return tuple(result)


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
    }

    # Per-class F1
    f1_per_class = f1_score(labels, predictions, average=None, labels=range(NUM_LABELS))
    for i, f1 in enumerate(f1_per_class):
        metrics[f"f1_{ID2LABEL[i]}"] = f1

    return metrics


class WeightedTrainer(Trainer):
    """Trainer with class weights for imbalanced data."""

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        # This method returns a microbatch mean and does not normalize with
        # num_items_in_batch. Trainer must apply gradient-accumulation scaling.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = torch.tensor(
                self.class_weights, device=logits.device, dtype=logits.dtype
            )
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser(description="Train feedback detector (4-class)")

    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        default="llm-semantic-router/mmbert-32k-yarn",
        help="Base model (default: mmBERT-32K YaRN with 32K context)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="models/mmbert32k_feedback_detector"
    )

    # Data
    parser.add_argument(
        "--data_source",
        type=str,
        default="llm-semantic-router/feedback-detector-dataset",
        help="HuggingFace dataset ID or local data directory",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Max samples per split"
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--dataset_revision")
    parser.add_argument("--model_revision")
    parser.add_argument("--seed", type=int, default=42)

    # Training (optimized for 4-class feedback detection, validated 2026-02-02)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Maximum epochs; selection uses isolated validation",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (2e-5 optimal for feedback)",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--use_class_weights", action="store_true", default=True)

    # LoRA
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA training")
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="LoRA rank (higher for 4-class feedback)",
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=128, help="LoRA alpha (2x rank recommended)"
    )
    parser.add_argument(
        "--merge_lora", action="store_true", help="Merge LoRA weights after training"
    )

    args = parser.parse_args()
    set_seed(args.seed)
    if not Path(args.model_name).exists() and not args.model_revision:
        parser.error("Remote base models require a fixed --model_revision")

    print("=" * 70)
    print("FEEDBACK DETECTOR TRAINING (4-class)")
    print("=" * 70)
    print(f"\nBase Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print(
        f"LoRA: {f'Yes (rank={args.lora_rank}, alpha={args.lora_alpha})' if args.use_lora else 'No (full fine-tune)'}"
    )
    print("\nLabels:")
    for label, idx in LABEL2ID.items():
        print(f"  {idx}: {label}")

    # Load data
    print(f"\n[1/5] Loading data from: {args.data_source}")
    train_examples, val_examples = load_feedback_data(
        args.data_source, args.max_samples, args.dataset_revision
    )

    data_manifest = validate_splits(train_examples, val_examples)

    # Count labels
    train_labels = [ex["label"] for ex in train_examples]
    label_counts = {}
    for ex in train_examples:
        label_counts[ex["label_name"]] = label_counts.get(ex["label_name"], 0) + 1

    print(f"\n  Train: {len(train_examples):,} examples")
    for label in LABEL2ID:
        count = label_counts.get(label, 0)
        pct = count / len(train_examples) * 100 if train_examples else 0
        print(f"    {label}: {count:,} ({pct:.1f}%)")
    print(f"\n  Val: {len(val_examples):,} examples")

    # Compute class weights
    class_weights = None
    if args.use_class_weights and len(set(train_labels)) > 1:
        class_weights = balanced_class_weights(train_labels)
        print(f"\n  Class weights: {dict(zip(LABEL2ID, class_weights, strict=True))}")

    # Create datasets
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)

    # Load tokenizer and model
    print(f"\n[2/5] Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, revision=args.model_revision
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    model.config.reference_compile = False
    if args.max_length <= 0 or args.max_length > model.config.max_position_embeddings:
        raise ValueError("max_length must fit the actual backbone context")

    # Apply LoRA if requested
    if args.use_lora:
        if not PEFT_AVAILABLE:
            print("ERROR: PEFT not installed. Cannot use LoRA.")
            return

        print(f"\n  Applying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            target_modules=[
                "attn.Wqkv",
                "attn.Wo",
                "mlp.Wi",
                "mlp.Wo",
            ],  # mmBERT/ModernBERT
            bias="none",
            revision=args.model_revision,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Tokenize
    print(f"\n[3/5] Tokenizing (max_length={args.max_length})...")

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    train_dataset = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=[
            column for column in train_dataset.column_names if column != "label"
        ],
    )
    val_dataset = val_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=[
            column for column in val_dataset.column_names if column != "label"
        ],
    )

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    # Training arguments
    print("\n[4/5] Setting up training...")
    output_dir = args.output_dir + ("_lora" if args.use_lora else "")

    # Check if running on AMD GPU (ROCm) - disable fp16 AMP which has issues with BFloat16
    use_fp16 = False
    use_bf16 = False
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0).lower()
        if "amd" in device_name or "mi300" in device_name or "instinct" in device_name:
            # AMD GPUs: use bf16 without gradient scaling, or disable mixed precision
            print(f"  Detected AMD GPU ({device_name}), using bf16=True")
            use_bf16 = True
        else:
            # NVIDIA GPUs: use fp16 with AMP
            use_fp16 = True

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        seed=args.seed,
        data_seed=args.seed,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=100,
        report_to="none",
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=4,
    )

    # Create trainer (use processing_class for newer transformers, fallback to tokenizer)
    try:
        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,  # New API in transformers 5.x
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
    except TypeError:
        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,  # Old API
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

    # Train
    print("\n[5/5] Training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {training_args.device}")
    print()

    trainer.train()

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    results = trainer.evaluate()
    for key, value in sorted(results.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    # Classification report
    print("\n" + "-" * 70)
    predictions = trainer.predict(val_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids
    print(
        classification_report(
            labels, preds, target_names=list(LABEL2ID.keys()), digits=4
        )
    )

    # Save model
    print(f"\nSaving model to {output_dir}")

    if args.use_lora:
        # Save LoRA adapter
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Merge and save if requested
        if args.merge_lora:
            merged_dir = str(merged_output_directory(output_dir))
            print(f"Merging LoRA weights to {merged_dir}")
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            write_label_mapping(merged_dir)

            # Save config for merged model
            config = {
                "model_type": "feedback_detector",
                "labels": list(LABEL2ID.keys()),
                "label2id": LABEL2ID,
                "id2label": ID2LABEL,
                "base_model": args.model_name,
                "base_revision": args.model_revision,
                "max_length": args.max_length,
                "lora_merged": True,
                "metrics": {
                    k: float(v)
                    for k, v in results.items()
                    if isinstance(v, (int, float))
                },
            }
            with open(Path(merged_dir) / "training_config.json", "w") as f:
                json.dump(config, f, indent=2)
    else:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

    # Save training config
    config = {
        "model_type": "feedback_detector",
        "labels": list(LABEL2ID.keys()),
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
        "base_model": args.model_name,
        "base_revision": args.model_revision,
        "max_length": args.max_length,
        "use_lora": args.use_lora,
        "lora_rank": args.lora_rank if args.use_lora else None,
        "lora_alpha": args.lora_alpha if args.use_lora else None,
        "class_weights": class_weights,
        "metrics": {
            k: float(v) for k, v in results.items() if isinstance(v, (int, float))
        },
    }
    with open(Path(output_dir) / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    write_label_mapping(output_dir)
    for directory in [output_dir] + (
        [merged_dir] if args.use_lora and args.merge_lora else []
    ):
        (Path(directory) / "data_manifest.json").write_text(
            json.dumps(data_manifest, indent=2) + "\n", encoding="utf-8"
        )

    print("\nTraining complete!")
    print(f"   Output: {output_dir}")
    if args.use_lora and args.merge_lora:
        print(f"   Merged: {merged_dir}")


if __name__ == "__main__":
    main()
