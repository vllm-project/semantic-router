#!/usr/bin/env python3
"""
LoRA Training Script for Domain-Specific Cache Embedding Models

Pipeline Position: Step 2 of 2
    Input:  triplets.jsonl (from generate_training_data.py)
    Output: LoRA adapter model files (lightweight domain-specific embeddings)

Trains a domain-specific cache embedding model using LoRA (Low-Rank Adaptation)
with Multiple Negatives Ranking (MNR) loss.

Based on research: arXiv:2504.02268v1
- Small specialized models > Large general models for cache matching
- 1 epoch training is sufficient with good contrastive data
- LoRA enables efficient fine-tuning (only 0.32-0.44% of parameters)

Usage:
    python lora_trainer.py \
        --train-data data/programming/triplets.jsonl \
        --val-data data/programming/val.jsonl \
        --base-model sentence-transformers/all-MiniLM-L12-v2 \
        --output models/programming-cache-lora \
        --epochs 1
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.cache_embeddings.common_utils import (
    set_seed,
    save_jsonl,
    load_jsonl,
    setup_logging,
)

logger = logging.getLogger(__name__)


# === MNR Loss (Multiple Negatives Ranking) ===
# Merged from losses.py - this is the ONLY loss we use for cache embedding training


class MultipleNegativesRankingLoss(nn.Module):
    """
    Multiple Negatives Ranking (MNR) Loss.

    Recommended by the research paper (arXiv:2504.02268v1) for semantic caching.
    Uses in-batch negatives: all other positives in the batch serve as negatives.

    This is efficient and effective for learning semantic similarity.
    """

    def __init__(self, temperature: float = 0.05, reduction: str = "mean"):
        """
        Args:
            temperature: Temperature parameter for scaling
            reduction: "mean", "sum", or "none"
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

        logger.info(
            f"MultipleNegativesRankingLoss initialized: temperature={temperature}"
        )

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        """
        Compute MNR loss using in-batch negatives.

        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive embeddings [batch_size, embedding_dim]

        Returns:
            Loss value
        """
        # Normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)

        # Compute similarity matrix
        # [batch_size, batch_size]
        similarity_matrix = torch.matmul(anchor, positive.T) / self.temperature

        # Labels: diagonal elements are positives (i matches i)
        labels = torch.arange(anchor.size(0), device=anchor.device)

        # Cross-entropy loss: maximize similarity with correct positive
        loss = F.cross_entropy(similarity_matrix, labels, reduction=self.reduction)

        return loss


# Optional dependencies
try:
    from transformers import AutoModel, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "transformers and peft not available. " "Install: pip install transformers peft"
    )


class TripletDataset(Dataset):
    """Dataset for triplet training (anchor, positive, negative)."""

    def __init__(
        self,
        data_file: str,
        tokenizer,
        max_length: int = 128,
    ):
        """
        Initialize triplet dataset.

        Args:
            data_file: Path to JSONL file with triplets
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.data = load_jsonl(data_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

        logger.info(f"Loaded {len(self.data)} triplets from {data_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize anchor, positive, negative
        anchor = self.tokenizer(
            item["anchor"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Handle positive pairs (may not exist for negative samples)
        positive_text = item.get("positive", "")
        positive = self.tokenizer(
            positive_text if positive_text else "[PAD]",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Support both 'negative' and 'hard_negative' field names
        negative_text = item.get("negative") or item.get("hard_negative", "")
        negative = self.tokenizer(
            negative_text if negative_text else "[PAD]",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "anchor_input_ids": anchor["input_ids"].squeeze(0),
            "anchor_attention_mask": anchor["attention_mask"].squeeze(0),
            "positive_input_ids": positive["input_ids"].squeeze(0),
            "positive_attention_mask": positive["attention_mask"].squeeze(0),
            "negative_input_ids": negative["input_ids"].squeeze(0),
            "negative_attention_mask": negative["attention_mask"].squeeze(0),
        }


class CacheEmbeddingModel(nn.Module):
    """Wrapper for BERT-based embedding model with LoRA."""

    def __init__(self, base_model_name: str, use_lora: bool = True):
        """
        Initialize cache embedding model.

        Args:
            base_model_name: HuggingFace model name
            use_lora: Whether to use LoRA (recommended)
        """
        super().__init__()

        self.base_model_name = base_model_name
        self.model = AutoModel.from_pretrained(base_model_name)

        if use_lora:
            # Configure LoRA
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=8,  # LoRA rank (higher = more parameters, better quality)
                lora_alpha=32,  # LoRA scaling factor
                lora_dropout=0.1,
                target_modules=["query", "value"],  # Apply LoRA to attention layers
                bias="none",
            )

            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        logger.info(f"Initialized model: {base_model_name} (LoRA: {use_lora})")

    def forward(self, input_ids, attention_mask):
        """
        Forward pass - generate embeddings.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask

        Returns:
            Normalized embeddings
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Mean pooling
        token_embeddings = outputs.last_hidden_state
        attention_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        sum_embeddings = torch.sum(token_embeddings * attention_mask_expanded, dim=1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask

        # Normalize (important for cosine similarity)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings


def train_epoch(
    model: CacheEmbeddingModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: MultipleNegativesRankingLoss,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: Cache embedding model
        train_loader: Training data loader
        optimizer: Optimizer
        loss_fn: MNR loss function
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Training metrics
    """
    model.train()

    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        # Move to device
        anchor_ids = batch["anchor_input_ids"].to(device)
        anchor_mask = batch["anchor_attention_mask"].to(device)
        positive_ids = batch["positive_input_ids"].to(device)
        positive_mask = batch["positive_attention_mask"].to(device)

        # Forward pass
        anchor_emb = model(anchor_ids, anchor_mask)
        positive_emb = model(positive_ids, positive_mask)

        # Compute MNR loss (uses in-batch negatives)
        loss = loss_fn(anchor_emb, positive_emb)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        num_batches += 1

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches

    return {"train_loss": avg_loss}


def validate(
    model: CacheEmbeddingModel,
    val_loader: DataLoader,
    loss_fn: MultipleNegativesRankingLoss,
    device: torch.device,
) -> Dict[str, float]:
    """
    Validate the model.

    Args:
        model: Cache embedding model
        val_loader: Validation data loader
        loss_fn: MNR loss function
        device: Device

    Returns:
        Validation metrics
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Move to device
            anchor_ids = batch["anchor_input_ids"].to(device)
            anchor_mask = batch["anchor_attention_mask"].to(device)
            positive_ids = batch["positive_input_ids"].to(device)
            positive_mask = batch["positive_attention_mask"].to(device)

            # Forward pass
            anchor_emb = model(anchor_ids, anchor_mask)
            positive_emb = model(positive_ids, positive_mask)

            # Compute loss
            loss = loss_fn(anchor_emb, positive_emb)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches

    return {"val_loss": avg_loss}


def save_model(model: CacheEmbeddingModel, output_dir: str, tokenizer):
    """
    Save trained model and tokenizer.

    Args:
        model: Trained model
        output_dir: Output directory
        tokenizer: Tokenizer
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save LoRA adapters (lightweight!)
    model.model.save_pretrained(output_dir)

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    # Save training info
    info = {
        "base_model": model.base_model_name,
        "architecture": "LoRA fine-tuned BERT",
        "task": "cache embedding",
        "loss": "Multiple Negatives Ranking (MNR)",
    }

    with open(os.path.join(output_dir, "training_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    logger.info(f"Model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train domain-specific cache embedding model with LoRA"
    )

    # Data arguments
    parser.add_argument("--train-data", required=True, help="Training data JSONL file")
    parser.add_argument("--val-data", help="Validation data JSONL file (optional)")

    # Model arguments
    parser.add_argument(
        "--base-model",
        default="sentence-transformers/all-MiniLM-L12-v2",
        help="Base embedding model",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA (train full model - not recommended)",
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs (paper uses 1)"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--max-length", type=int, default=128, help="Max sequence length"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.05, help="MNR loss temperature"
    )

    # Output arguments
    parser.add_argument("--output", required=True, help="Output directory for model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Check dependencies
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Error: transformers and peft required")
        print("Install: pip install transformers peft")
        return 1

    # Setup
    setup_logging()
    set_seed(args.seed)
    # Prefer MPS (Apple Silicon) > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")
    logger.info(f"Training configuration: {vars(args)}")

    # Load tokenizer and model
    logger.info(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = CacheEmbeddingModel(args.base_model, use_lora=not args.no_lora)
    model = model.to(device)

    # Create datasets
    train_dataset = TripletDataset(args.train_data, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging
    )

    val_loader = None
    if args.val_data:
        val_dataset = TripletDataset(args.val_data, tokenizer, args.max_length)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = MultipleNegativesRankingLoss(temperature=args.temperature)

    # Training loop
    logger.info("=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)

    training_history = []

    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch
        )

        logger.info(f"Epoch {epoch} - Train Loss: {train_metrics['train_loss']:.4f}")

        # Validate
        if val_loader:
            val_metrics = validate(model, val_loader, loss_fn, device)
            logger.info(f"Epoch {epoch} - Val Loss: {val_metrics['val_loss']:.4f}")
            train_metrics.update(val_metrics)

        training_history.append({"epoch": epoch, **train_metrics})

    # Save model
    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)

    save_model(model, args.output, tokenizer)

    # Save training history
    history_file = os.path.join(args.output, "training_history.json")
    with open(history_file, "w") as f:
        json.dump(training_history, f, indent=2)

    logger.info(f"Training history saved to {history_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"Model saved to: {args.output}")
    print(f"Base model: {args.base_model}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Epochs: {args.epochs}")
    print(f"Final train loss: {training_history[-1]['train_loss']:.4f}")
    if val_loader:
        print(f"Final val loss: {training_history[-1]['val_loss']:.4f}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
