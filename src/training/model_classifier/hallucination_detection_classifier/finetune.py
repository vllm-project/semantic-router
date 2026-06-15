#!/usr/bin/env python3
"""
Standalone fine-tuning script for hallucination detection using ModernBERT-32K.

This script trains a token classification model to detect hallucinated spans
in LLM-generated responses.

Model: llm-semantic-router/modernbert-base-32k (32K context window)
Task: Token Classification (2 classes: Supported=0, Hallucinated=1)

Best Configuration (from research):
- max_length: 8192
- batch_size: 8
- learning_rate: 1e-5
- epochs: 6
- loss: CrossEntropyLoss (no class weights)
- scheduler: None (constant LR)
- early_stopping: 4 epochs

Usage:
    # Basic training
    python finetune.py \
        --train-path ./data/train.json \
        --dev-path ./data/dev.json \
        --output-dir ./output/haldetect-32k

    # Full training with test evaluation
    python finetune.py \
        --train-path ./data/train.json \
        --dev-path ./data/dev.json \
        --test-path ./data/test.json \
        --output-dir ./output/haldetect-32k \
        --epochs 6 \
        --batch-size 8 \
        --learning-rate 1e-5

Dependencies:
    pip install torch transformers scikit-learn tqdm
"""

import argparse
import json
import random
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
)
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class HallucinationSample:
    """A single hallucination detection sample."""

    prompt: str
    answer: str
    labels: list[dict]  # List of {"start": int, "end": int, "label": str}
    split: str
    task_type: str
    dataset: str
    language: str

    @classmethod
    def from_dict(cls, data: dict) -> "HallucinationSample":
        return cls(
            prompt=data["prompt"],
            answer=data["answer"],
            labels=data.get("labels", []),
            split=data.get("split", "train"),
            task_type=data.get("task_type", "unknown"),
            dataset=data.get("dataset", "unknown"),
            language=data.get("language", "en"),
        )


# ============================================================================
# Dataset Class
# ============================================================================


class HallucinationDataset(Dataset):
    """PyTorch Dataset for hallucination detection."""

    def __init__(
        self,
        samples: list[HallucinationSample],
        tokenizer: AutoTokenizer,
        max_length: int = 8192,
    ):
        """
        Initialize the dataset.

        Args:
            samples: List of HallucinationSample objects
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.

        Returns:
            Dictionary with input_ids, attention_mask, and labels tensors
        """
        sample = self.samples[idx]

        # Tokenize context and answer together
        encoding = self.tokenizer(
            sample.prompt,
            sample.answer,
            truncation="only_first",  # Only truncate context if needed
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=True,
        )

        # Get offset mappings and remove from encoding
        offsets = encoding.pop("offset_mapping")[0]  # Shape: (seq_length, 2)

        # Find where the answer starts
        # Tokenize just the context to find the boundary
        context_only = self.tokenizer(
            sample.prompt, add_special_tokens=True, return_tensors="pt"
        )
        answer_start_token = context_only["input_ids"].shape[1]

        # Handle edge case where we land on a special token
        if (
            answer_start_token < offsets.size(0)
            and offsets[answer_start_token][0] == offsets[answer_start_token][1]
        ):
            answer_start_token += 1

        # Initialize labels: -100 for context tokens (ignored in loss)
        seq_length = encoding["input_ids"].shape[1]
        labels = [-100] * seq_length

        # Get the character offset where the answer starts
        if answer_start_token < len(offsets):
            answer_char_offset = offsets[answer_start_token][0].item()
        else:
            answer_char_offset = 0

        # Label answer tokens based on hallucination spans
        for i in range(answer_start_token, seq_length):
            token_start, token_end = offsets[i]
            token_start = token_start.item()
            token_end = token_end.item()

            # Adjust to be relative to answer text
            token_rel_start = token_start - answer_char_offset
            token_rel_end = token_end - answer_char_offset

            # Default: supported (0)
            token_label = 0

            # Check if token overlaps any hallucination span
            for ann in sample.labels:
                if token_rel_end > ann["start"] and token_rel_start < ann["end"]:
                    token_label = 1  # Hallucinated
                    break

            labels[i] = token_label

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ============================================================================
# Evaluation Functions
# ============================================================================


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """
    Evaluate model on token-level hallucination detection.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on

    Returns:
        Dictionary with precision, recall, F1 for each class
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Token-Level Eval", leave=False):
            outputs = model(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            predictions = torch.argmax(outputs.logits, dim=-1)

            # Only evaluate on tokens that have labels (not -100)
            mask = batch["labels"] != -100
            predictions = predictions[mask].cpu().numpy()
            labels = batch["labels"][mask].cpu().numpy()

            all_preds.extend(predictions.tolist())
            all_labels.extend(labels.tolist())

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1], average=None, zero_division=0
    )

    return {
        "supported": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
        },
        "hallucinated": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
        },
    }


def evaluate_example_level(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """
    Evaluate model at example level (any hallucinated token = hallucinated example).

    This matches the RAGTruth benchmark evaluation methodology.
    """
    model.eval()
    example_preds = []
    example_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Example-Level Eval", leave=False):
            outputs = model(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            predictions = torch.argmax(outputs.logits, dim=-1)

            # Process each example in batch
            for i in range(batch["labels"].size(0)):
                sample_labels = batch["labels"][i]
                sample_preds = predictions[i].cpu()

                # Mask for valid labels
                valid_mask = sample_labels != -100

                if valid_mask.sum().item() == 0:
                    example_labels.append(0)
                    example_preds.append(0)
                    continue

                sample_labels = sample_labels[valid_mask]
                sample_preds = sample_preds[valid_mask]

                # If any token is hallucinated, example is hallucinated
                true_label = 1 if (sample_labels == 1).any().item() else 0
                pred_label = 1 if (sample_preds == 1).any().item() else 0

                example_labels.append(true_label)
                example_preds.append(pred_label)

    precision, recall, f1, _ = precision_recall_fscore_support(
        example_labels, example_preds, labels=[0, 1], average=None, zero_division=0
    )

    return {
        "supported": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
        },
        "hallucinated": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
        },
    }


def print_metrics(metrics: dict, title: str = "Evaluation Results"):
    """Print evaluation metrics in a readable format."""
    print(f"\n{title}:")
    print(
        f"  Hallucinated - P: {metrics['hallucinated']['precision']:.4f}, "
        f"R: {metrics['hallucinated']['recall']:.4f}, "
        f"F1: {metrics['hallucinated']['f1']:.4f}"
    )
    print(
        f"  Supported    - P: {metrics['supported']['precision']:.4f}, "
        f"R: {metrics['supported']['recall']:.4f}, "
        f"F1: {metrics['supported']['f1']:.4f}"
    )


def print_benchmark_comparison(token_f1: float, example_f1: float):
    """Print comparison with benchmark models."""
    print("\n" + "-" * 40)
    print("BENCHMARK COMPARISON")
    print("-" * 40)
    print(f"Token-Level F1:")
    print(f"model-f1:           {token_f1*100:.2f}%")
    print(f"\nExample-Level F1:")
    print(f"model-f1:           {example_f1*100:.2f}%")
    print("-" * 40)


# ============================================================================
# Training
# ============================================================================


class Trainer:
    """Trainer for hallucination detection with per-epoch benchmark evaluation."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        benchmark_loader: Optional[DataLoader] = None,
        benchmark_name: str = "RAGTruth",
        epochs: int = 6,
        learning_rate: float = 1e-5,
        save_path: str = "output/haldetect",
        early_stopping_patience: int = 4,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: Model to train
            tokenizer: Tokenizer for the model
            train_loader: DataLoader for training data
            dev_loader: DataLoader for development data (mixed)
            test_loader: Optional DataLoader for test data
            benchmark_loader: Optional DataLoader for benchmark evaluation (RAGTruth test)
            benchmark_name: Name of the benchmark dataset
            epochs: Number of training epochs
            learning_rate: Learning rate
            save_path: Path to save the best model
            early_stopping_patience: Epochs without improvement before stopping
            device: Device to train on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.benchmark_loader = benchmark_loader
        self.benchmark_name = benchmark_name
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.save_path = Path(save_path)
        self.early_stopping_patience = early_stopping_patience
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Setup optimizer (no scheduler - constant LR works best)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        self.model.to(self.device)
        self.save_path.mkdir(parents=True, exist_ok=True)

    def train(self) -> float:
        """
        Train the model.

        Returns:
            Best hallucinated F1 score achieved during training
        """
        best_f1 = 0.0
        best_benchmark_token_f1 = 0.0
        best_benchmark_example_f1 = 0.0
        epochs_without_improvement = 0
        start_time = time.time()

        print(f"\n{'=' * 60}")
        print("TRAINING STARTED")
        print(f"{'=' * 60}")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Dev samples: {len(self.dev_loader.dataset)}")
        if self.test_loader:
            print(f"Test samples: {len(self.test_loader.dataset)}")
        if self.benchmark_loader:
            print(
                f"{self.benchmark_name} benchmark samples: {len(self.benchmark_loader.dataset)}"
            )
        print(f"Epochs: {self.epochs}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Early stopping patience: {self.early_stopping_patience}")
        print(f"{'=' * 60}\n")

        for epoch in range(self.epochs):
            epoch_start = time.time()

            # Training
            self.model.train()
            total_loss = 0
            num_batches = 0

            progress_bar = tqdm(
                self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=True
            )

            for batch in progress_bar:
                self.optimizer.zero_grad()

                outputs = self.model(
                    batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    labels=batch["labels"].to(self.device),
                )

                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "avg": f"{total_loss / num_batches:.4f}",
                    }
                )

            avg_loss = total_loss / num_batches
            epoch_time = time.time() - epoch_start

            print(
                f"\nEpoch {epoch + 1} completed in {timedelta(seconds=int(epoch_time))}"
            )
            print(f"Average loss: {avg_loss:.4f}")

            # ================================================================
            # Evaluation on Dev Set (Mixed)
            # ================================================================
            print("\n" + "=" * 50)
            print("DEV SET EVALUATION (Mixed)")
            print("=" * 50)

            dev_token_metrics = evaluate_model(self.model, self.dev_loader, self.device)
            dev_example_metrics = evaluate_example_level(
                self.model, self.dev_loader, self.device
            )

            print_metrics(dev_token_metrics, "Dev Token-Level")
            print_metrics(dev_example_metrics, "Dev Example-Level")

            # ================================================================
            # Evaluation on RAGTruth Benchmark (if provided)
            # ================================================================
            if self.benchmark_loader:
                print("\n" + "=" * 50)
                print(f"{self.benchmark_name} BENCHMARK EVALUATION")
                print("=" * 50)

                bench_token_metrics = evaluate_model(
                    self.model, self.benchmark_loader, self.device
                )
                bench_example_metrics = evaluate_example_level(
                    self.model, self.benchmark_loader, self.device
                )

                print_metrics(bench_token_metrics, f"{self.benchmark_name} Token-Level")
                print_metrics(
                    bench_example_metrics, f"{self.benchmark_name} Example-Level"
                )

                # Print benchmark comparison
                print_benchmark_comparison(
                    bench_token_metrics["hallucinated"]["f1"],
                    bench_example_metrics["hallucinated"]["f1"],
                )

                # Track best benchmark scores
                if (
                    bench_example_metrics["hallucinated"]["f1"]
                    > best_benchmark_example_f1
                ):
                    best_benchmark_token_f1 = bench_token_metrics["hallucinated"]["f1"]
                    best_benchmark_example_f1 = bench_example_metrics["hallucinated"][
                        "f1"
                    ]

            # ================================================================
            # Check for improvement (using dev example-level F1)
            # ================================================================
            current_f1 = dev_example_metrics["hallucinated"]["f1"]

            if current_f1 > best_f1:
                best_f1 = current_f1
                epochs_without_improvement = 0

                # Save best model
                self.model.save_pretrained(self.save_path)
                self.tokenizer.save_pretrained(self.save_path)

                print(f"\n🎯 New best Dev Example-Level F1: {best_f1:.4f}")
                print(f"   Model saved to: {self.save_path}")
            else:
                epochs_without_improvement += 1
                print(f"\nNo improvement for {epochs_without_improvement} epoch(s)")

                if epochs_without_improvement >= self.early_stopping_patience:
                    print(f"\n⚠️ Early stopping triggered after {epoch + 1} epochs")
                    break

            print("\n" + "-" * 60)

        # ================================================================
        # Final evaluation on test set with best model
        # ================================================================
        if self.test_loader or self.benchmark_loader:
            print(f"\n{'=' * 60}")
            print("FINAL EVALUATION WITH BEST MODEL")
            print(f"{'=' * 60}")

            # Load best model
            self.model = AutoModelForTokenClassification.from_pretrained(self.save_path)
            self.model.to(self.device)

            if self.test_loader:
                print("\n--- Test Set ---")
                test_token = evaluate_model(self.model, self.test_loader, self.device)
                test_example = evaluate_example_level(
                    self.model, self.test_loader, self.device
                )
                print_metrics(test_token, "Test Token-Level")
                print_metrics(test_example, "Test Example-Level")

            if self.benchmark_loader:
                print(f"\n--- {self.benchmark_name} Benchmark ---")
                bench_token = evaluate_model(
                    self.model, self.benchmark_loader, self.device
                )
                bench_example = evaluate_example_level(
                    self.model, self.benchmark_loader, self.device
                )
                print_metrics(bench_token, f"{self.benchmark_name} Token-Level")
                print_metrics(bench_example, f"{self.benchmark_name} Example-Level")

                # Final benchmark comparison
                print_benchmark_comparison(
                    bench_token["hallucinated"]["f1"],
                    bench_example["hallucinated"]["f1"],
                )

                # Detailed classification report
                print("\nDetailed Token-Level Classification Report:")
                self._print_detailed_report(self.benchmark_loader)

        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print("TRAINING COMPLETE")
        print(f"{'=' * 60}")
        print(f"Total time: {timedelta(seconds=int(total_time))}")
        print(f"Best Dev Example-Level F1: {best_f1:.4f}")
        if self.benchmark_loader:
            print(
                f"Best {self.benchmark_name} Token-Level F1: {best_benchmark_token_f1:.4f}"
            )
            print(
                f"Best {self.benchmark_name} Example-Level F1: {best_benchmark_example_f1:.4f}"
            )
        print(f"Model saved to: {self.save_path}")
        print(f"{'=' * 60}")

        return best_f1

    def _print_detailed_report(self, dataloader: DataLoader):
        """Print detailed classification report."""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model(
                    batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                )
                predictions = torch.argmax(outputs.logits, dim=-1)

                mask = batch["labels"] != -100
                predictions = predictions[mask].cpu().numpy()
                labels = batch["labels"][mask].cpu().numpy()

                all_preds.extend(predictions.tolist())
                all_labels.extend(labels.tolist())

        report = classification_report(
            all_labels,
            all_preds,
            target_names=["Supported", "Hallucinated"],
            digits=4,
        )
        print(report)


# ============================================================================
# Main
# ============================================================================


def set_seed(seed: int = 42):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_samples(path: Path) -> list[HallucinationSample]:
    """Load samples from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return [HallucinationSample.from_dict(d) for d in data]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune hallucination detector with 32K ModernBERT"
    )

    # Data arguments
    parser.add_argument(
        "--train-path",
        type=str,
        required=True,
        help="Path to training data JSON file",
    )
    parser.add_argument(
        "--dev-path",
        type=str,
        required=True,
        help="Path to development data JSON file",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        help="Path to test data JSON file (RAGTruth test set for benchmark)",
    )

    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="llm-semantic-router/modernbert-base-32k",
        help="Name or path of the pretrained model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/haldetect-32k",
        help="Directory to save the trained model",
    )

    # Training arguments
    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Maximum sequence length (up to 32768 for 32K model)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=6,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=4,
        help="Epochs without improvement before early stopping",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # Memory optimization
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print("\n" + "=" * 60)
    print("HALLUCINATION DETECTOR FINE-TUNING")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Max length: {args.max_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_samples = load_samples(Path(args.train_path))
    dev_samples = load_samples(Path(args.dev_path))
    test_samples = None
    if args.test_path:
        test_samples = load_samples(Path(args.test_path))

    print(f"Training samples: {len(train_samples)}")
    print(f"Dev samples: {len(dev_samples)}")
    if test_samples:
        print(f"Test samples: {len(test_samples)}")

    # Filter RAGTruth-only samples for benchmark evaluation
    ragtruth_test_samples = None
    if test_samples:
        ragtruth_test_samples = [s for s in test_samples if s.dataset == "ragtruth"]
        if ragtruth_test_samples:
            print(f"RAGTruth benchmark samples: {len(ragtruth_test_samples)}")

    # Load tokenizer
    print(f"\nLoading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    # Create datasets
    print("Creating datasets...")
    train_dataset = HallucinationDataset(
        train_samples, tokenizer, max_length=args.max_length
    )
    dev_dataset = HallucinationDataset(
        dev_samples, tokenizer, max_length=args.max_length
    )
    test_dataset = None
    if test_samples:
        test_dataset = HallucinationDataset(
            test_samples, tokenizer, max_length=args.max_length
        )

    # Create RAGTruth benchmark dataset
    benchmark_dataset = None
    if ragtruth_test_samples:
        benchmark_dataset = HallucinationDataset(
            ragtruth_test_samples, tokenizer, max_length=args.max_length
        )

    # Create data loaders
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )

    # Create benchmark loader (RAGTruth only)
    benchmark_loader = None
    if benchmark_dataset:
        benchmark_loader = DataLoader(
            benchmark_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )

    # Load model
    print(f"\nLoading model from {args.model_name}...")
    config = AutoConfig.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    # Print model info
    if hasattr(config, "max_position_embeddings"):
        print(f"  Max positions: {config.max_position_embeddings}")

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=2,  # Supported (0), Hallucinated (1)
        trust_remote_code=True,
    )

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Create trainer and train
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        benchmark_loader=benchmark_loader,
        benchmark_name="RAGTruth",
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_path=args.output_dir,
        early_stopping_patience=args.early_stopping_patience,
    )

    best_f1 = trainer.train()

    # Print usage instructions
    print("\n" + "=" * 60)
    print("TO USE THE TRAINED MODEL")
    print("=" * 60)
    print(
        """
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Load model
model = AutoModelForTokenClassification.from_pretrained("{output_dir}")
tokenizer = AutoTokenizer.from_pretrained("{output_dir}")

# Predict
context = "Your context here..."
answer = "LLM response to check..."

inputs = tokenizer(context, answer, return_tensors="pt", truncation="only_first", max_length=8192)
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)

# Find hallucinated spans (where prediction == 1)
""".format(
            output_dir=args.output_dir
        )
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
