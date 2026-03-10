import json
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from dual_classifier import DualClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class DualTaskDataset(Dataset):
    """
    Dataset for dual-task learning with category classification and PII detection.
    """

    def __init__(
        self,
        texts: List[str],
        category_labels: List[int],
        pii_labels: List[List[int]],  # Token-level PII labels
        tokenizer,
        max_length: int = 512,
    ):
        self.texts = texts
        self.category_labels = category_labels
        self.pii_labels = pii_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        category_label = self.category_labels[idx]
        pii_label = self.pii_labels[idx]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Prepare PII labels to match tokenized length
        # Note: This is simplified - in practice you'd need proper token alignment
        pii_labels_padded = pii_label[: self.max_length]
        if len(pii_labels_padded) < self.max_length:
            pii_labels_padded.extend([0] * (self.max_length - len(pii_labels_padded)))

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "category_label": torch.tensor(category_label, dtype=torch.long),
            "pii_labels": torch.tensor(pii_labels_padded, dtype=torch.long),
        }


class DualTaskLoss(nn.Module):
    """
    Combined loss function for dual-task learning.
    """

    def __init__(self, category_weight: float = 1.0, pii_weight: float = 1.0):
        super().__init__()
        self.category_weight = category_weight
        self.pii_weight = pii_weight
        self.category_loss_fn = nn.CrossEntropyLoss()
        self.pii_loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100
        )  # Ignore padding tokens

    def forward(
        self,
        category_logits: torch.Tensor,
        pii_logits: torch.Tensor,
        category_labels: torch.Tensor,
        pii_labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate combined loss for both tasks.

        Returns:
            total_loss, category_loss, pii_loss
        """
        # Category classification loss
        category_loss = self.category_loss_fn(category_logits, category_labels)

        # PII detection loss - only compute loss for attended tokens
        # Reshape for loss computation
        pii_logits_flat = pii_logits.view(-1, pii_logits.size(-1))
        pii_labels_flat = pii_labels.view(-1)

        # Mask out padded tokens
        attention_mask_flat = attention_mask.view(-1)
        pii_labels_masked = pii_labels_flat.clone()
        pii_labels_masked[attention_mask_flat == 0] = -100

        pii_loss = self.pii_loss_fn(pii_logits_flat, pii_labels_masked)

        # Combined loss
        total_loss = self.category_weight * category_loss + self.pii_weight * pii_loss

        return total_loss, category_loss, pii_loss


class DualTaskTrainer:
    """
    Trainer for the dual-purpose classifier.
    """

    def __init__(
        self,
        model: DualClassifier,
        train_dataset: DualTaskDataset,
        val_dataset: Optional[DualTaskDataset] = None,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        category_weight: float = 1.0,
        pii_weight: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Set device
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        # Setup loss function and optimizer
        self.loss_fn = DualTaskLoss(category_weight, pii_weight)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues on some systems
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )

        # Training history
        self.history = {
            "train_loss": [],
            "train_category_loss": [],
            "train_pii_loss": [],
            "val_loss": [],
            "val_category_loss": [],
            "val_pii_loss": [],
            "val_category_acc": [],
            "val_pii_f1": [],
        }

    def train_epoch(self) -> Tuple[float, float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_category_loss = 0
        total_pii_loss = 0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc="Training")

        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            category_labels = batch["category_label"].to(self.device)
            pii_labels = batch["pii_labels"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            category_logits, pii_logits = self.model(input_ids, attention_mask)

            # Calculate loss
            loss, cat_loss, pii_loss = self.loss_fn(
                category_logits, pii_logits, category_labels, pii_labels, attention_mask
            )

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            total_category_loss += cat_loss.item()
            total_pii_loss += pii_loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "cat_loss": f"{cat_loss.item():.4f}",
                    "pii_loss": f"{pii_loss.item():.4f}",
                }
            )

        return (
            total_loss / num_batches,
            total_category_loss / num_batches,
            total_pii_loss / num_batches,
        )

    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        if not self.val_dataset:
            return {}

        self.model.eval()
        total_loss = 0
        total_category_loss = 0
        total_pii_loss = 0
        num_batches = 0

        all_category_preds = []
        all_category_labels = []
        all_pii_preds = []
        all_pii_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                category_labels = batch["category_label"].to(self.device)
                pii_labels = batch["pii_labels"].to(self.device)

                # Forward pass
                category_logits, pii_logits = self.model(input_ids, attention_mask)

                # Calculate loss
                loss, cat_loss, pii_loss = self.loss_fn(
                    category_logits,
                    pii_logits,
                    category_labels,
                    pii_labels,
                    attention_mask,
                )

                # Update loss metrics
                total_loss += loss.item()
                total_category_loss += cat_loss.item()
                total_pii_loss += pii_loss.item()
                num_batches += 1

                # Collect predictions for metrics
                category_preds = torch.argmax(category_logits, dim=1)
                all_category_preds.extend(category_preds.cpu().numpy())
                all_category_labels.extend(category_labels.cpu().numpy())

                # PII predictions (only for non-padded tokens)
                pii_preds = torch.argmax(pii_logits, dim=2)
                for i in range(len(batch["input_ids"])):
                    mask = attention_mask[i].cpu().numpy()
                    valid_length = mask.sum()
                    all_pii_preds.extend(pii_preds[i][:valid_length].cpu().numpy())
                    all_pii_labels.extend(pii_labels[i][:valid_length].cpu().numpy())

        # Calculate metrics
        category_acc = accuracy_score(all_category_labels, all_category_preds)
        pii_f1 = f1_score(all_pii_labels, all_pii_preds, average="weighted")

        return {
            "val_loss": total_loss / num_batches,
            "val_category_loss": total_category_loss / num_batches,
            "val_pii_loss": total_pii_loss / num_batches,
            "val_category_acc": category_acc,
            "val_pii_f1": pii_f1,
        }

    def train(self):
        """Train the model for the specified number of epochs."""
        print(f"Training on device: {self.device}")
        print(f"Number of training samples: {len(self.train_dataset)}")
        if self.val_dataset:
            print(f"Number of validation samples: {len(self.val_dataset)}")

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # Train
            train_loss, train_cat_loss, train_pii_loss = self.train_epoch()

            # Log training metrics
            self.history["train_loss"].append(train_loss)
            self.history["train_category_loss"].append(train_cat_loss)
            self.history["train_pii_loss"].append(train_pii_loss)

            print(
                f"Train Loss: {train_loss:.4f}, "
                f"Category Loss: {train_cat_loss:.4f}, "
                f"PII Loss: {train_pii_loss:.4f}"
            )

            # Evaluate
            if self.val_dataset:
                val_metrics = self.evaluate()

                # Log validation metrics
                for key, value in val_metrics.items():
                    self.history[key].append(value)

                print(
                    f"Val Loss: {val_metrics['val_loss']:.4f}, "
                    f"Category Acc: {val_metrics['val_category_acc']:.4f}, "
                    f"PII F1: {val_metrics['val_pii_f1']:.4f}"
                )

    def save_model(self, path: str):
        """Save the trained model."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)

        # Save training history
        with open(f"{path}/training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)
