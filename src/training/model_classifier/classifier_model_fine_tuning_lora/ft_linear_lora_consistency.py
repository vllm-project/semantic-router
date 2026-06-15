"""
CONSISTENCY TRAINING for Noise-Robust Classification

This script implements CONSISTENCY LOSS training that forces the model to:
1. Maintain high accuracy on clean text
2. Give the SAME prediction for noisy/augmented versions of the same text

Key Innovation:
- For each sample, creates BOTH clean AND augmented (typo) versions
- Adds consistency loss (KL divergence) that penalizes different predictions
- Preserves clean accuracy while learning robustness to input noise

Benefits:
- No accuracy regression on clean text (maintains original capability)
- Improved robustness to typos, misspellings, and input noise
- Single training pass (no two-stage training needed)

Tested Training Parameters (v1 - recommended):
    accelerate launch ft_linear_lora_consistency.py \\
        --epochs 15 \\
        --max-samples 25000 \\
        --typo-prob 0.20 \\
        --consistency-weight 1.0 \\
        --learning-rate 2e-5 \\
        --batch-size 16

    Results on 100-sample test set:
    - Clean Accuracy: 73% (+3% vs baseline)
    - Typo Accuracy: 45% (+11% vs baseline)

Relationship to ft_linear_lora.py:
- ft_linear_lora.py: Standard LoRA fine-tuning with optional typo augmentation
- ft_linear_lora_consistency.py (THIS FILE): Adds consistency loss for robustness
  The consistency loss approach was developed to address issue #952 where typo-laden
  prompts caused incorrect domain classification.

Usage:
    # Single GPU
    python ft_linear_lora_consistency.py --epochs 15 --max-samples 25000

    # Multi-GPU with accelerate
    accelerate launch ft_linear_lora_consistency.py --epochs 15 --max-samples 25000
"""

import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Import common LoRA utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_lora_utils import (
    clear_gpu_memory,
    create_lora_config,
    get_all_gpu_info,
    log_memory_usage,
    resolve_model_path,
    set_gpu_device,
    setup_logging,
)

logger = setup_logging()

# Required categories
REQUIRED_CATEGORIES = [
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
]

# Keyboard layout for realistic typos (QWERTY)
KEYBOARD_ADJACENT = {
    "a": "sqwz",
    "b": "vghn",
    "c": "xdfv",
    "d": "serfcx",
    "e": "wrsdf",
    "f": "drtgvc",
    "g": "ftyhbv",
    "h": "gyujnb",
    "i": "ujklo",
    "j": "huikmn",
    "k": "jiolm",
    "l": "kop",
    "m": "njk",
    "n": "bhjm",
    "o": "iklp",
    "p": "ol",
    "q": "wa",
    "r": "edft",
    "s": "awedxz",
    "t": "rfgy",
    "u": "yhji",
    "v": "cfgb",
    "w": "qase",
    "x": "zsdc",
    "y": "tghu",
    "z": "asx",
}

# Common typo patterns (real-world mistakes)
COMMON_TYPOS = {
    "the": "teh",
    "and": "adn",
    "that": "taht",
    "with": "wiht",
    "this": "tihs",
    "from": "form",
    "have": "hvae",
    "were": "weer",
    "their": "thier",
    "there": "tehre",
    "these": "tehse",
    "those": "thsoe",
    "what": "waht",
    "when": "wehn",
    "where": "wehre",
    "which": "whcih",
    "solve": "slove",
    "problem": "prblem",
    "mathematical": "mathemtical",
    "explain": "expalin",
    "describe": "descrbe",
    "calculate": "calculte",
    "following": "follwing",
    "structure": "strcture",
    "process": "procss",
}

# Vowel substitutions (common mistakes)
VOWEL_SUBS = {"a": "eiou", "e": "aiou", "i": "aeou", "o": "aeiu", "u": "aeio"}

# Common character confusions
CHAR_CONFUSIONS = {
    "i": "l1",
    "l": "i1",
    "o": "0",
    "0": "o",
    "1": "il",
    "s": "z5",
    "z": "s2",
    "e": "3",
    "a": "4",
    "g": "9",
}


def apply_typo(text: str, prob: float = 0.20) -> str:
    """
    Apply realistic typos to text with given probability per word.
    Enhanced version with more realistic typo patterns.
    """
    if not text or len(text) < 3:
        return text

    words = text.split()
    result = []

    for word in words:
        original_word = word
        word_lower = word.lower()

        # Skip if too short or not alphabetic
        if len(word) < 3 or not word.isalpha():
            result.append(word)
            continue

        # Skip with probability
        if random.random() >= prob:
            result.append(word)
            continue

        # Check for common typo patterns first (more realistic)
        if word_lower in COMMON_TYPOS and random.random() < 0.3:
            # Use common typo pattern
            typo = COMMON_TYPOS[word_lower]
            # Preserve capitalization
            if word[0].isupper():
                typo = typo.capitalize()
            result.append(typo)
            continue

        word_list = list(word)
        word_lower_list = [c.lower() for c in word_list]

        # Decide number of typos (1-2 for longer words)
        num_typos = 1 if len(word) < 6 else (1 if random.random() < 0.7 else 2)

        for _ in range(num_typos):
            if len(word_list) < 3:
                break

            # Choose augmentation type with weighted probabilities
            # More realistic typos are more common
            aug_weights = [
                ("keyboard", 0.35),  # Most common - keyboard mistakes
                ("swap", 0.25),  # Common - transposition
                ("delete", 0.20),  # Common - missing character
                ("vowel", 0.10),  # Less common - vowel confusion
                ("substitute", 0.05),  # Less common - character substitution
                ("double", 0.05),  # Least common - double character
            ]

            aug_type = random.choices(
                [w[0] for w in aug_weights], weights=[w[1] for w in aug_weights]
            )[0]

            if aug_type == "swap" and len(word_list) > 2:
                # Character swap (common typo)
                idx = random.randint(0, len(word_list) - 2)
                word_list[idx], word_list[idx + 1] = word_list[idx + 1], word_list[idx]
                word_lower_list[idx], word_lower_list[idx + 1] = (
                    word_lower_list[idx + 1],
                    word_lower_list[idx],
                )

            elif aug_type == "delete" and len(word_list) > 3:
                # Delete character (common - missing key)
                idx = random.randint(1, len(word_list) - 2)  # Avoid first/last
                word_list.pop(idx)
                word_lower_list.pop(idx)

            elif aug_type == "keyboard":
                # Keyboard adjacent mistake (most realistic)
                idx = random.randint(0, len(word_list) - 1)
                char = word_lower_list[idx]
                if char in KEYBOARD_ADJACENT:
                    new_char = random.choice(KEYBOARD_ADJACENT[char])
                    # Preserve case
                    if word_list[idx].isupper():
                        new_char = new_char.upper()
                    word_list[idx] = new_char
                    word_lower_list[idx] = new_char.lower()

            elif aug_type == "vowel":
                # Vowel substitution (common mistake)
                vowels = [i for i, c in enumerate(word_lower_list) if c in "aeiou"]
                if vowels:
                    idx = random.choice(vowels)
                    char = word_lower_list[idx]
                    if char in VOWEL_SUBS:
                        new_char = random.choice(VOWEL_SUBS[char])
                        # Preserve case
                        if word_list[idx].isupper():
                            new_char = new_char.upper()
                        word_list[idx] = new_char
                        word_lower_list[idx] = new_char

            elif aug_type == "substitute":
                # Character confusion (less common)
                idx = random.randint(0, len(word_list) - 1)
                char = word_lower_list[idx]
                if char in CHAR_CONFUSIONS:
                    new_char = random.choice(CHAR_CONFUSIONS[char])
                    # Preserve case
                    if word_list[idx].isupper():
                        new_char = new_char.upper()
                    word_list[idx] = new_char
                    word_lower_list[idx] = new_char.lower()

            elif aug_type == "double":
                # Double character (typing too fast)
                idx = random.randint(0, len(word_list) - 1)
                word_list.insert(idx, word_list[idx])
                word_lower_list.insert(idx, word_lower_list[idx])

        result.append("".join(word_list))

    return " ".join(result)


class ConsistencyDataset(torch.utils.data.Dataset):
    """
    Dataset that returns PAIRS of (clean, typo) versions for consistency training.
    """

    def __init__(self, texts, labels, tokenizer, typo_prob=0.20, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.typo_prob = typo_prob
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        clean_text = self.texts[idx]
        typo_text = apply_typo(clean_text, self.typo_prob)
        label = self.labels[idx]

        # Tokenize clean version
        clean_enc = self.tokenizer(
            clean_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Tokenize typo version
        typo_enc = self.tokenizer(
            typo_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "clean_input_ids": clean_enc["input_ids"].squeeze(),
            "clean_attention_mask": clean_enc["attention_mask"].squeeze(),
            "typo_input_ids": typo_enc["input_ids"].squeeze(),
            "typo_attention_mask": typo_enc["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class ConsistencyTrainer(Trainer):
    """
    Custom trainer that adds consistency loss between clean and typo predictions.

    Loss = CrossEntropy(clean) + CrossEntropy(typo) + λ * KL_Divergence(clean, typo)

    This forces the model to:
    1. Classify clean text correctly
    2. Classify typo text correctly
    3. Give SAME predictions for both versions
    """

    def __init__(self, consistency_weight=1.0, *args, **kwargs):
        # Remove columns removal to keep our custom keys
        if (
            "remove_unused_columns"
            not in kwargs.get("args", TrainingArguments(output_dir="tmp")).__dict__
        ):
            pass
        super().__init__(*args, **kwargs)
        self.consistency_weight = consistency_weight

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.get("labels")

        # Forward pass on clean text
        clean_outputs = model(
            input_ids=inputs.get("clean_input_ids"),
            attention_mask=inputs.get("clean_attention_mask"),
        )
        clean_logits = clean_outputs.logits

        # Forward pass on typo text
        typo_outputs = model(
            input_ids=inputs.get("typo_input_ids"),
            attention_mask=inputs.get("typo_attention_mask"),
        )
        typo_logits = typo_outputs.logits

        # Classification loss on both versions
        ce_loss = nn.CrossEntropyLoss()
        clean_loss = ce_loss(clean_logits, labels)
        typo_loss = ce_loss(typo_logits, labels)

        # Consistency loss: KL divergence between predictions
        # This encourages same prediction for clean and typo versions
        clean_probs = F.log_softmax(clean_logits, dim=-1)
        typo_probs = F.softmax(typo_logits, dim=-1)
        consistency_loss = F.kl_div(clean_probs, typo_probs, reduction="batchmean")

        # Total loss
        total_loss = clean_loss + typo_loss + self.consistency_weight * consistency_loss

        return (total_loss, clean_outputs) if return_outputs else total_loss


def load_mmlu_dataset(max_samples=10000):
    """Load MMLU-Pro dataset with balanced category sampling."""
    logger.info("Loading MMLU-Pro dataset...")

    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    all_texts = dataset["test"]["question"]
    all_labels = dataset["test"]["category"]

    # Group by category
    category_samples = {}
    for text, label in zip(all_texts, all_labels):
        if label not in category_samples:
            category_samples[label] = []
        category_samples[label].append(text)

    # Balanced sampling
    available_cats = [c for c in REQUIRED_CATEGORIES if c in category_samples]
    samples_per_cat = max_samples // len(available_cats)

    texts, labels = [], []
    label2id = {cat: idx for idx, cat in enumerate(REQUIRED_CATEGORIES)}

    for cat in available_cats:
        cat_texts = category_samples[cat][:samples_per_cat]
        texts.extend(cat_texts)
        labels.extend([label2id[cat]] * len(cat_texts))

    logger.info(f"Loaded {len(texts)} samples from {len(available_cats)} categories")
    return texts, labels, label2id


def create_model(model_name: str, num_labels: int, lora_rank: int, lora_alpha: int):
    """Create LoRA model."""
    model_path = resolve_model_path(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        torch_dtype=torch.float32,
    )

    lora_config = create_lora_config(model_name, lora_rank, lora_alpha, 0.1)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=lora_config["rank"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        target_modules=lora_config["target_modules"],
        bias="none",
    )

    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


def data_collator(features):
    """Custom collator for consistency training."""
    batch = {
        "clean_input_ids": torch.stack([f["clean_input_ids"] for f in features]),
        "clean_attention_mask": torch.stack(
            [f["clean_attention_mask"] for f in features]
        ),
        "typo_input_ids": torch.stack([f["typo_input_ids"] for f in features]),
        "typo_attention_mask": torch.stack(
            [f["typo_attention_mask"] for f in features]
        ),
        "labels": torch.stack([f["labels"] for f in features]),
    }
    return batch


def main(
    model_name: str = "bert-base-uncased",
    lora_rank: int = 16,
    lora_alpha: int = 32,
    epochs: int = 15,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_samples: int = 25000,
    typo_prob: float = 0.20,
    consistency_weight: float = 1.0,
    output_dir: str = None,
):
    """Main training function with consistency loss."""

    logger.info("=" * 60)
    logger.info("CONSISTENCY TRAINING FOR TYPO ROBUSTNESS")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Max samples: {max_samples}")
    logger.info(f"Typo probability: {typo_prob}")
    logger.info(f"Consistency weight: {consistency_weight}")

    # Load data
    texts, labels, label2id = load_mmlu_dataset(max_samples)
    id2label = {v: k for k, v in label2id.items()}

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )

    logger.info(f"Training samples: {len(train_texts)}")
    logger.info(f"Validation samples: {len(val_texts)}")

    # Create model
    model, tokenizer = create_model(model_name, len(label2id), lora_rank, lora_alpha)

    # Create datasets
    train_dataset = ConsistencyDataset(train_texts, train_labels, tokenizer, typo_prob)
    val_dataset = ConsistencyDataset(val_texts, val_labels, tokenizer, typo_prob)

    # Output directory
    if output_dir is None:
        output_dir = f"consistency_classifier_{model_name}_r{lora_rank}"
    os.makedirs(output_dir, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,  # Use bf16 for L4 GPUs
        remove_unused_columns=False,  # CRITICAL: Keep our custom columns
    )

    # Create trainer
    trainer = ConsistencyTrainer(
        consistency_weight=consistency_weight,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting consistency training...")
    trainer.train()

    # Save model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save label mapping
    mapping = {"category_to_idx": label2id, "idx_to_category": id2label}
    with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
        json.dump(mapping, f, indent=2)
    with open(os.path.join(output_dir, "category_mapping.json"), "w") as f:
        json.dump(mapping, f, indent=2)

    logger.info(f"Model saved to: {output_dir}")

    # Merge for Rust compatibility
    logger.info("Merging LoRA adapter for Rust inference...")
    merged_dir = f"{output_dir}_rust"
    merge_lora_model(output_dir, merged_dir, model_name, label2id, id2label)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Merged model saved to: {merged_dir}")
    logger.info("=" * 60)


def merge_lora_model(lora_path, output_path, model_name, label2id, id2label):
    """Merge LoRA adapter with base model."""
    model_path = resolve_model_path(model_name)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=len(label2id), torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    lora_model = PeftModel.from_pretrained(base_model, lora_path)
    merged_model = lora_model.merge_and_unload()

    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Save config with labels
    config_path = os.path.join(output_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    config["id2label"] = id2label
    config["label2id"] = label2id
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Copy label mappings
    for fname in ["label_mapping.json", "category_mapping.json"]:
        src = os.path.join(lora_path, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_path, fname))

    # Create lora_config.json
    lora_config = {"rank": 16, "alpha": 32, "dropout": 0.1}
    with open(os.path.join(output_path, "lora_config.json"), "w") as f:
        json.dump(lora_config, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Consistency Training for Typo Robustness"
    )
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-samples", type=int, default=25000)
    parser.add_argument("--typo-prob", type=float, default=0.20)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    main(
        model_name=args.model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_samples=args.max_samples,
        typo_prob=args.typo_prob,
        consistency_weight=args.consistency_weight,
        output_dir=args.output_dir,
    )
