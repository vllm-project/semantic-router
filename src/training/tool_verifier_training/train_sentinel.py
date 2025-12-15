#!/usr/bin/env python3
"""
Train Stage 1 FunctionCallSentinel model for jailbreak/injection detection.

This is a binary sequence classifier that detects potentially risky prompts:
- SAFE (0): Normal, benign prompts
- INJECTION_RISK (1): Jailbreak, prompt injection, or tool-abuse attempts

Usage:
    python train_sentinel.py \
        --train-data data/generated/stage1_train.json \
        --dev-data data/generated/stage1_dev.json \
        --output-dir output/function_call_sentinel
"""

import argparse
import json
import random
from collections import Counter
from datetime import timedelta
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from data_generation import (
    SentinelDataset,
    SentinelSample,
    load_sentinel_samples,
    load_sentinel_label_config,
)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(
    train_samples: list[SentinelSample], label2id: dict
) -> torch.Tensor:
    """
    Compute class weights for imbalanced jailbreak detection.

    Jailbreak datasets are often imbalanced, with more benign than malicious samples.
    """
    label_counts = Counter(s.label for s in train_samples)

    total = sum(label_counts.values())
    num_labels = len(label2id)

    weights = []
    for i in range(num_labels):
        label = [k for k, v in label2id.items() if v == i][0]
        count = label_counts.get(label, 1)
        # Inverse frequency weighting
        weight = total / (num_labels * count)
        weights.append(weight)

    weights = torch.tensor(weights, dtype=torch.float32)

    # Normalize to have mean of 1.0
    weights = weights / weights.mean()

    print(f"\nClass weights:")
    for i, w in enumerate(weights):
        label = [k for k, v in label2id.items() if v == i][0]
        count = label_counts.get(label, 0)
        print(f"  {label}: {w:.4f} (count: {count})")

    return weights


def compute_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    id2label: dict,
    label2id: dict,
) -> dict:
    """Compute evaluation metrics for sequence classification."""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Prob of INJECTION_RISK

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    num_labels = len(id2label)
    target_names = [id2label[i] for i in range(num_labels)]

    report = classification_report(
        all_labels,
        all_preds,
        labels=list(range(num_labels)),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_labels)))

    # ROC-AUC for binary classification
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = 0.0

    # Precision/Recall/F1 for INJECTION_RISK class (main metric)
    injection_idx = label2id["INJECTION_RISK"]
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=[injection_idx],
        average="binary",
        pos_label=injection_idx,
    )

    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "injection_precision": precision,
        "injection_recall": recall,
        "injection_f1": f1,
        "roc_auc": roc_auc,
    }


def print_metrics(metrics: dict, id2label: dict):
    """Print evaluation metrics."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    report = metrics["classification_report"]

    print(
        f"\n{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}"
    )
    print("-" * 70)

    for i in range(len(id2label)):
        label = id2label[i]
        if label in report:
            r = report[label]
            print(
                f"{label:<20} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1-score']:<12.4f} {int(r['support']):<10}"
            )

    print("-" * 70)
    print(f"{'Accuracy':<20} {'':<12} {'':<12} {metrics['accuracy']:<12.4f}")
    print(
        f"{'Macro Avg':<20} {report['macro avg']['precision']:<12.4f} {report['macro avg']['recall']:<12.4f} {metrics['macro_f1']:<12.4f}"
    )
    print(
        f"{'Weighted Avg':<20} {report['weighted avg']['precision']:<12.4f} {report['weighted avg']['recall']:<12.4f} {metrics['weighted_f1']:<12.4f}"
    )

    print(f"\n🎯 Key Metrics (INJECTION_RISK class):")
    print(f"   Precision: {metrics['injection_precision']:.4f}")
    print(f"   Recall:    {metrics['injection_recall']:.4f}")
    print(f"   F1-Score:  {metrics['injection_f1']:.4f}")
    print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(f"{'':>15}", end="")
    for i in range(len(id2label)):
        print(f"{id2label[i][:8]:>12}", end="")
    print()

    cm = metrics["confusion_matrix"]
    for i in range(len(cm)):
        print(f"{id2label[i]:<15}", end="")
        for j in range(len(cm[i])):
            print(f"{cm[i][j]:>12}", end="")
        print()

    print("=" * 80)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    save_path: str,
    id2label: dict,
    label2id: dict,
    class_weights: torch.Tensor = None,
) -> dict:
    """Train the sentinel model."""

    best_f1 = 0.0
    best_metrics = None

    # Use class-weighted CrossEntropyLoss
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using class-weighted CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss")

    print(f"\nStarting training on {device}")
    print(f"Train: {len(train_loader.dataset)}, Dev: {len(dev_loader.dataset)}")

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print("=" * 80)

        model.train()
        total_loss = 0.0
        num_batches = 0

        progress = tqdm(train_loader, desc="Training")
        for batch in progress:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = criterion(logits, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️ Skipping batch {num_batches} due to NaN/Inf loss")
                continue

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            progress.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg": f"{total_loss/num_batches:.4f}",
                }
            )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - epoch_start
        print(
            f"\nEpoch {epoch+1}: avg_loss={avg_loss:.4f}, time={timedelta(seconds=int(epoch_time))}"
        )

        # Evaluate
        metrics = compute_metrics(model, dev_loader, device, id2label, label2id)
        print_metrics(metrics, id2label)

        # Save best model based on INJECTION_RISK F1 (key metric)
        current_f1 = metrics["injection_f1"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_metrics = metrics

            print(f"\n✅ New best Injection F1: {best_f1:.4f}")
            model.save_pretrained(save_path)
            train_loader.dataset.tokenizer.save_pretrained(save_path)

            with open(Path(save_path) / "best_metrics.json", "w") as f:
                json.dump(
                    {k: v for k, v in metrics.items() if k != "confusion_matrix"},
                    f,
                    indent=2,
                )

            print(f"Model saved to {save_path}")

    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Training complete: {timedelta(seconds=int(total_time))}")
    print(f"Best Injection F1: {best_f1:.4f}")
    print("=" * 80)

    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Stage 1 FunctionCallSentinel")
    parser.add_argument(
        "--train-data", type=str, default="data/generated/stage1_train.json"
    )
    parser.add_argument(
        "--dev-data", type=str, default="data/generated/stage1_dev.json"
    )
    parser.add_argument(
        "--label-config", type=str, default="data/generated/sentinel_label_config.json"
    )
    parser.add_argument("--model-name", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument(
        "--output-dir", type=str, default="output/function_call_sentinel"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-class-weights", action="store_true", default=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load label config
    label2id, id2label = load_sentinel_label_config(args.label_config)
    num_labels = len(label2id)
    print(f"Labels ({num_labels}): {list(label2id.keys())}")

    # Load data
    print(f"\nLoading data...")
    train_samples = load_sentinel_samples(args.train_data)
    dev_samples = load_sentinel_samples(args.dev_data)
    print(f"Train: {len(train_samples)}, Dev: {len(dev_samples)}")

    # Print class distribution
    train_dist = Counter(s.label for s in train_samples)
    dev_dist = Counter(s.label for s in dev_samples)
    print(f"Train distribution: {dict(train_dist)}")
    print(f"Dev distribution: {dict(dev_dist)}")

    # Initialize tokenizer and model
    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True,
        attn_implementation="sdpa",  # Use SDPA which enables Flash Attention on ROCm automatically
    )
    model.to(device)

    # Create datasets
    train_dataset = SentinelDataset(train_samples, tokenizer, label2id, args.max_length)
    dev_dataset = SentinelDataset(dev_samples, tokenizer, label2id, args.max_length)

    # Compute class weights
    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(train_samples, label2id)

    # Data loaders
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save training config
    config = {
        "model_name": args.model_name,
        "num_labels": num_labels,
        "label2id": label2id,
        "id2label": id2label,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "use_class_weights": args.use_class_weights,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
    }
    (output_dir / "training_config.json").write_text(json.dumps(config, indent=2))

    # Train
    best_metrics = train(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        save_path=str(output_dir),
        id2label=id2label,
        label2id=label2id,
        class_weights=class_weights,
    )

    print(f"\nTraining complete! Model saved to: {output_dir}")

    # Save final report
    if best_metrics:
        final_report = {
            "accuracy": best_metrics["accuracy"],
            "injection_precision": best_metrics["injection_precision"],
            "injection_recall": best_metrics["injection_recall"],
            "injection_f1": best_metrics["injection_f1"],
            "roc_auc": best_metrics["roc_auc"],
            "macro_f1": best_metrics["macro_f1"],
        }
        (output_dir / "final_report.json").write_text(
            json.dumps(final_report, indent=2)
        )

        print("\n📊 Final Results:")
        for k, v in final_report.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
