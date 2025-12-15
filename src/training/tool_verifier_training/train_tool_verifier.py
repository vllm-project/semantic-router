#!/usr/bin/env python3
"""
Train Stage 2 ToolCallVerifier model for unauthorized tool-call detection.

This is a token-level classifier that verifies tool calls against user intent:
- AUTHORIZED (0): Tool call aligns with user request
- UNAUTHORIZED (1): Violates policy/intent - block it
- UNAUTHORIZED (2): Clearly violates policy/intent

Usage:
    python train_tool_verifier.py \
        --train-data data/generated/stage2_train.json \
        --dev-data data/generated/stage2_dev.json \
        --output-dir output/tool_call_verifier
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
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)
from sklearn.metrics import classification_report, confusion_matrix

from data_generation import (
    ToolCallVerifierDataset,
    ToolCallSample,
    load_tool_call_samples,
    load_verifier_label_config,
)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(
    train_samples: list[ToolCallSample], label2id: dict
) -> torch.Tensor:
    """
    Compute class weights for imbalanced tool-call verification.

    Most tokens are AUTHORIZED, while UNAUTHORIZED are rare.
    """
    label_counts = Counter()

    for sample in train_samples:
        for label_info in sample.labels:
            label_counts[label_info["label"]] += 1

    num_labels = len(label2id)
    id2label = {v: k for k, v in label2id.items()}

    # Fixed weights designed for tool-call verification:
    # - AUTHORIZED: downweight (dominant class)
    # - UNAUTHORIZED: boost (rare but important for security)
    fixed_weights = {
        "AUTHORIZED": 0.5,  # Dominant class - downweight
        "UNAUTHORIZED": 3.0,  # Very important - strong boost
    }

    weights = []
    for i in range(num_labels):
        label = id2label[i]
        weight = fixed_weights.get(label, 1.0)
        weights.append(weight)

    weights = torch.tensor(weights, dtype=torch.float32)

    print(f"\nClass weights (fixed for tool-call verification):")
    for i, w in enumerate(weights):
        label = id2label[i]
        count = label_counts.get(label, 0)
        print(f"  {label}: {w:.4f} (spans: {count})")

    return weights


def compute_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    id2label: dict,
    label2id: dict,
) -> dict:
    """Compute evaluation metrics for token classification."""
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)

            for pred_seq, label_seq in zip(preds, labels):
                for pred, label in zip(pred_seq, label_seq):
                    if label.item() != -100:
                        all_preds.append(pred.item())
                        all_labels.append(label.item())

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

    # Calculate unauthorized detection F1 (key metric)
    unauthorized_classes = [k for k in label2id.keys() if k != "AUTHORIZED"]
    unauthorized_f1_scores = []
    for cls in unauthorized_classes:
        if cls in report and report[cls]["support"] > 0:
            unauthorized_f1_scores.append(report[cls]["f1-score"])

    unauthorized_f1 = (
        sum(unauthorized_f1_scores) / len(unauthorized_f1_scores)
        if unauthorized_f1_scores
        else 0.0
    )

    # Calculate UNAUTHORIZED-specific metrics (most important class)
    unauthorized_idx = label2id.get("UNAUTHORIZED", 2)
    if "UNAUTHORIZED" in report:
        unauthorized_precision = report["UNAUTHORIZED"]["precision"]
        unauthorized_recall = report["UNAUTHORIZED"]["recall"]
        unauthorized_specific_f1 = report["UNAUTHORIZED"]["f1-score"]
    else:
        unauthorized_precision = 0.0
        unauthorized_recall = 0.0
        unauthorized_specific_f1 = 0.0

    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "unauthorized_avg_f1": unauthorized_f1,
        "unauthorized_precision": unauthorized_precision,
        "unauthorized_recall": unauthorized_recall,
        "unauthorized_f1": unauthorized_specific_f1,
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

    print(f"\n🎯 Key Metrics (Unauthorized Detection):")
    print(f"   UNAUTHORIZED Precision: {metrics['unauthorized_precision']:.4f}")
    print(f"   UNAUTHORIZED Recall:    {metrics['unauthorized_recall']:.4f}")
    print(f"   UNAUTHORIZED F1:        {metrics['unauthorized_f1']:.4f}")
    print(f"   Non-AUTHORIZED Avg F1:  {metrics['unauthorized_avg_f1']:.4f}")

    # Confusion matrix
    print("\nConfusion Matrix:")
    abbrev = {
        "AUTHORIZED": "AUTH",
        "UNAUTHORIZED": "UNAUTH",
    }

    print(f"{'':>12}", end="")
    for i in range(len(id2label)):
        label = id2label[i]
        print(f"{abbrev.get(label, label[:6]):>10}", end="")
    print()

    cm = metrics["confusion_matrix"]
    for i in range(len(cm)):
        label = id2label[i]
        print(f"{abbrev.get(label, label[:6]):<12}", end="")
        for j in range(len(cm[i])):
            print(f"{cm[i][j]:>10}", end="")
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
    """Train the tool-call verifier model."""

    best_f1 = 0.0
    best_metrics = None

    # Use class-weighted CrossEntropyLoss
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        print("Using class-weighted CrossEntropyLoss for tool-call verification")
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
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

            # Compute loss with class weights
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

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

        # Save best model based on UNAUTHORIZED F1 (key metric)
        current_f1 = metrics["unauthorized_f1"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_metrics = metrics

            print(f"\n✅ New best UNAUTHORIZED F1: {best_f1:.4f}")
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
    print(f"Best UNAUTHORIZED F1: {best_f1:.4f}")
    print("=" * 80)

    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Stage 2 ToolCallVerifier")
    parser.add_argument(
        "--train-data", type=str, default="data/generated/stage2_train.json"
    )
    parser.add_argument(
        "--dev-data", type=str, default="data/generated/stage2_dev.json"
    )
    parser.add_argument(
        "--label-config", type=str, default="data/generated/verifier_label_config.json"
    )
    parser.add_argument("--model-name", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--output-dir", type=str, default="output/tool_call_verifier")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load label config
    label2id, id2label, severity = load_verifier_label_config(args.label_config)
    num_labels = len(label2id)
    print(f"Labels ({num_labels}): {list(label2id.keys())}")
    print(f"Severity levels: {severity}")

    # Load data
    print(f"\nLoading data...")
    train_samples = load_tool_call_samples(args.train_data)
    dev_samples = load_tool_call_samples(args.dev_data)
    print(f"Train: {len(train_samples)}, Dev: {len(dev_samples)}")

    # Print label distribution
    train_unauthorized = sum(
        1
        for s in train_samples
        if any(ann.get("label") == "UNAUTHORIZED" for ann in s.labels)
    )
    dev_unauthorized = sum(
        1
        for s in dev_samples
        if any(ann.get("label") == "UNAUTHORIZED" for ann in s.labels)
    )
    print(f"Train unauthorized samples: {train_unauthorized}/{len(train_samples)}")
    print(f"Dev unauthorized samples: {dev_unauthorized}/{len(dev_samples)}")

    # Initialize tokenizer and model
    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True,
        attn_implementation="sdpa",  # Use SDPA which enables Flash Attention on ROCm automatically
    )
    model.to(device)

    # Create datasets
    train_dataset = ToolCallVerifierDataset(
        train_samples, tokenizer, label2id, args.max_length
    )
    dev_dataset = ToolCallVerifierDataset(
        dev_samples, tokenizer, label2id, args.max_length
    )

    # Compute class weights
    class_weights = compute_class_weights(train_samples, label2id)

    # Data loaders
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, label_pad_token_id=-100
    )
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save training config
    config = {
        "model_name": args.model_name,
        "num_labels": num_labels,
        "label2id": label2id,
        "id2label": id2label,
        "severity": severity,
        "loss": "CrossEntropyLoss (class-weighted)",
        "class_weights": class_weights.tolist(),
        "optimization_target": "unauthorized_f1",
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
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
            "unauthorized_precision": best_metrics["unauthorized_precision"],
            "unauthorized_recall": best_metrics["unauthorized_recall"],
            "unauthorized_f1": best_metrics["unauthorized_f1"],
            "unauthorized_avg_f1": best_metrics["unauthorized_avg_f1"],
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
