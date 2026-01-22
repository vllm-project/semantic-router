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
"""

import os
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

# Optional LoRA imports
try:
    from peft import LoraConfig, get_peft_model, TaskType

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print(
        "Warning: PEFT not installed. LoRA training unavailable. Install with: pip install peft"
    )


# Label mapping matching feedback-detector
LABEL2ID = {
    "SAT": 0,
    "NEED_CLARIFICATION": 1,
    "WRONG_ANSWER": 2,
    "WANT_DIFFERENT": 3,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


# SAT example templates for synthetic data generation
SAT_TEMPLATES = [
    "Thanks, that's helpful!",
    "Perfect, exactly what I needed!",
    "Great, thank you!",
    "That's exactly right, thanks!",
    "Awesome, this is perfect!",
    "Yes, that's what I was looking for!",
    "Great explanation, thanks!",
    "That works perfectly, thank you!",
    "Excellent, much appreciated!",
    "Yes, that's correct!",
    "Thanks for the help!",
    "Perfect answer, thanks!",
    "That's great, exactly what I needed!",
    "Wonderful, thank you so much!",
    "Yes, this is helpful!",
    "Got it, thanks!",
    "That makes sense, thanks!",
    "Okay, thanks for clarifying!",
    "Great info, appreciate it!",
    "Yes, thank you!",
    "That's perfect!",
    "Excellent work, thanks!",
    "This is exactly what I wanted!",
    "Thank you, that's very helpful!",
    "Perfect, I understand now!",
    "Yes, that answers my question!",
    "Great, this is useful!",
    "Thanks a lot!",
    "Awesome, appreciate the help!",
    "That's right, thank you!",
]

# Variations to expand SAT examples
SAT_VARIATIONS = [
    "{} 👍",
    "{}",
    "{} :)",
    "Ok, {}",
    "Alright, {}",
    "{} Much appreciated.",
    "{} I'll try that.",
    "{} This helps a lot.",
]


def generate_sat_examples(num_examples: int) -> list:
    """Generate synthetic SAT (satisfied) examples."""
    import random

    examples = []

    for i in range(num_examples):
        template = random.choice(SAT_TEMPLATES)
        variation = random.choice(SAT_VARIATIONS)
        text = variation.format(template) if "{}" in variation else template

        examples.append(
            {
                "text": text,
                "label": LABEL2ID["SAT"],
                "label_name": "SAT",
            }
        )

    return examples


def load_feedback_data(data_source: str, max_samples: int = None):
    """
    Load feedback data from HuggingFace dataset or local files.

    Args:
        data_source: HuggingFace dataset ID or local directory path
        max_samples: Maximum samples to load (None = all)

    Returns:
        train_examples, val_examples
    """
    train_examples = []
    val_examples = []

    # Try loading from HuggingFace
    if "/" in data_source and not Path(data_source).exists():
        print(f"Loading from HuggingFace: {data_source}")
        try:
            # Try loading directly first (for llm-semantic-router/feedback-detector-dataset)
            try:
                dataset = load_dataset(data_source)
            except Exception:
                # Fallback to specific files for older datasets
                dataset = load_dataset(
                    data_source,
                    data_files={"train": "train.jsonl", "validation": "val.jsonl"},
                )

            for split, examples in [
                ("train", train_examples),
                ("validation", val_examples),
            ]:
                if split in dataset:
                    for ex in dataset[split]:
                        text = ex.get("text", ex.get("followup", ""))
                        label = ex.get("label_name", ex.get("label", ""))

                        # Handle numeric or string labels
                        if isinstance(label, int):
                            label_name = ID2LABEL.get(label, "SAT")
                        else:
                            label_name = label if label in LABEL2ID else None

                        if label_name is None:
                            continue  # Skip unknown labels

                        examples.append(
                            {
                                "text": text,
                                "label": LABEL2ID[label_name],
                                "label_name": label_name,
                            }
                        )

                        if max_samples and len(examples) >= max_samples:
                            break

            # Add SAT examples if missing (for backwards compatibility with older datasets)
            train_sat_count = sum(
                1 for ex in train_examples if ex["label_name"] == "SAT"
            )
            if train_sat_count == 0:
                print("  No SAT examples found, generating synthetic SAT data...")
                sat_examples = generate_sat_examples(len(train_examples) // 3)
                train_examples.extend(sat_examples)
                val_sat = generate_sat_examples(len(val_examples) // 3)
                val_examples.extend(val_sat)

            return train_examples, val_examples
        except Exception as e:
            print(f"Could not load from HuggingFace: {e}")

    # Load from local JSONL files
    data_dir = Path(data_source)
    for split_name, examples in [("train", train_examples), ("val", val_examples)]:
        # Try multiple file patterns
        for pattern in [
            f"{split_name}.jsonl",
            f"{split_name}_feedback.jsonl",
            f"feedback_{split_name}.jsonl",
        ]:
            path = data_dir / pattern
            if path.exists():
                print(f"Loading from: {path}")
                with open(path, "r") as f:
                    for line in f:
                        ex = json.loads(line)
                        text = ex.get("text", ex.get("followup", ""))
                        label = ex.get("label_name", ex.get("label", "SAT"))

                        if isinstance(label, int):
                            label_name = ID2LABEL.get(label, "SAT")
                        elif label in LABEL2ID:
                            label_name = label
                        else:
                            # Map old labels to new
                            label_map = {
                                "SATISFIED": "SAT",
                                "DISSAT": "NEED_CLARIFICATION",  # Default DISSAT
                            }
                            label_name = label_map.get(label, "SAT")

                        examples.append(
                            {
                                "text": text,
                                "label": LABEL2ID[label_name],
                                "label_name": label_name,
                            }
                        )

                        if max_samples and len(examples) >= max_samples:
                            break
                break

    return train_examples, val_examples


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
        default="jhu-clsp/mmBERT-base",
        help="Base model (default: mmBERT)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="models/mmbert_feedback_detector"
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

    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--use_class_weights", action="store_true", default=True)

    # LoRA
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA training")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument(
        "--merge_lora", action="store_true", help="Merge LoRA weights after training"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("FEEDBACK DETECTOR TRAINING (4-class)")
    print("=" * 70)
    print(f"\nBase Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print(
        f"LoRA: {'Yes (rank={}, alpha={})'.format(args.lora_rank, args.lora_alpha) if args.use_lora else 'No (full fine-tune)'}"
    )
    print("\nLabels:")
    for label, idx in LABEL2ID.items():
        print(f"  {idx}: {label}")

    # Load data
    print(f"\n[1/5] Loading data from: {args.data_source}")
    train_examples, val_examples = load_feedback_data(
        args.data_source, args.max_samples
    )

    if not train_examples:
        print("ERROR: No training data found!")
        return

    # Count labels
    train_labels = [ex["label"] for ex in train_examples]
    label_counts = {}
    for ex in train_examples:
        label_counts[ex["label_name"]] = label_counts.get(ex["label_name"], 0) + 1

    print(f"\n  Train: {len(train_examples):,} examples")
    for label in LABEL2ID.keys():
        count = label_counts.get(label, 0)
        pct = count / len(train_examples) * 100 if train_examples else 0
        print(f"    {label}: {count:,} ({pct:.1f}%)")
    print(f"\n  Val: {len(val_examples):,} examples")

    # Compute class weights
    class_weights = None
    if args.use_class_weights and len(set(train_labels)) > 1:
        unique_labels = sorted(set(train_labels))
        class_weights = compute_class_weight(
            "balanced", classes=np.array(unique_labels), y=np.array(train_labels)
        ).tolist()
        # Pad to NUM_LABELS if needed
        while len(class_weights) < NUM_LABELS:
            class_weights.append(1.0)
        print(f"\n  Class weights: {dict(zip(LABEL2ID.keys(), class_weights))}")

    # Create datasets
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = (
        Dataset.from_list(val_examples)
        if val_examples
        else train_dataset.select(range(min(100, len(train_examples))))
    )

    # Load tokenizer and model
    print(f"\n[2/5] Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

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
            padding="max_length",
        )

    train_dataset = train_dataset.map(
        tokenize_fn, batched=True, remove_columns=["text", "label_name"]
    )
    val_dataset = val_dataset.map(
        tokenize_fn, batched=True, remove_columns=["text", "label_name"]
    )

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    # Training arguments
    print(f"\n[4/5] Setting up training...")
    output_dir = args.output_dir + ("_lora" if args.use_lora else "")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=100,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
    )

    # Create trainer
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train
    print(f"\n[5/5] Training...")
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
            merged_dir = output_dir.replace("_lora", "_merged")
            print(f"Merging LoRA weights to {merged_dir}")
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)

            # Save config for merged model
            config = {
                "model_type": "feedback_detector",
                "labels": list(LABEL2ID.keys()),
                "label2id": LABEL2ID,
                "id2label": ID2LABEL,
                "base_model": args.model_name,
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

    print("\n✅ Training complete!")
    print(f"   Output: {output_dir}")
    if args.use_lora and args.merge_lora:
        print(f"   Merged: {merged_dir}")


if __name__ == "__main__":
    main()
