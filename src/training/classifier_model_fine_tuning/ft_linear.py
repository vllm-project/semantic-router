"""
MMLU-Pro Category Classification Fine-tuning with Multiple BERT Models
Uses the Hugging Face Transformers approach with AutoModelForSequenceClassification.

Usage:
    # Train with default model (MiniLM)
    python ft_linear.py --mode train

    # Train with BERT base
    python ft_linear.py --mode train --model bert-base

    # Train with ModernBERT (recommended for best performance)
    python ft_linear.py --mode train --model modernbert-base

    # Train with custom epochs and batch size
    python ft_linear.py --mode train --model modernbert-base --epochs 10 --batch-size 32

    # Test inference with trained model
    python ft_linear.py --mode test --model bert-base

Supported models:
    - bert-base, bert-large: Standard BERT models
    - roberta-base, roberta-large: RoBERTa models
    - deberta-v3-base, deberta-v3-large: DeBERTa v3 models
    - modernbert-base, modernbert-large: ModernBERT models (recommended)
    - minilm: Lightweight MiniLM model (default for compatibility)
    - distilbert: Distilled BERT
    - electra-base, electra-large: ELECTRA models

Features:
    - Automatic classification head via AutoModelForSequenceClassification
    - Simplified training with Hugging Face Trainer
    - Built-in evaluation metrics (F1 score, accuracy)
    - Support for multiple BERT-based architectures
    - Automatic device detection (GPU/CPU)
    - Anti-overfitting measures for better generalization
"""

import json
import logging
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers import __version__ as transformers_version

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Check transformers version and compatibility
def check_transformers_compatibility():
    """Check transformers version and provide helpful messages."""
    logger.info(f"Transformers version: {transformers_version}")

    # Parse version to determine parameter names
    version_parts = transformers_version.split(".")
    major, minor = int(version_parts[0]), int(version_parts[1])

    # For versions < 4.19, use evaluation_strategy; for >= 4.19, use eval_strategy
    if major < 4 or (major == 4 and minor < 19):
        return "evaluation_strategy"
    else:
        return "eval_strategy"


# Device configuration - prioritize GPU if available
def get_device():
    """Get the best available device (GPU if available, otherwise CPU)."""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(
            f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
    else:
        device = "cpu"
        logger.warning(
            "No GPU detected. Using CPU. For better performance, ensure CUDA is installed."
        )

    logger.info(f"Using device: {device}")
    return device


# Model configurations for different BERT variants
MODEL_CONFIGS = {
    "bert-base": "bert-base-uncased",
    "bert-large": "bert-large-uncased",
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
    "deberta-v3-base": "microsoft/deberta-v3-base",
    "deberta-v3-large": "microsoft/deberta-v3-large",
    "modernbert-base": "answerdotai/ModernBERT-base",
    "modernbert-large": "answerdotai/ModernBERT-large",
    "mmbert-32k": "llm-semantic-router/mmbert-32k-yarn",  # 32K context YaRN scaled
    "minilm": "sentence-transformers/all-MiniLM-L12-v2",  # Default fallback
    "distilbert": "distilbert-base-uncased",
    "electra-base": "google/electra-base-discriminator",
    "electra-large": "google/electra-large-discriminator",
}


# Metrics computation function for Trainer
def compute_metrics(eval_pred):
    """Compute F1 score and accuracy for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    f1 = f1_score(labels, predictions, average="weighted")
    accuracy = accuracy_score(labels, predictions)
    return {"f1": f1, "accuracy": accuracy}


DEFAULT_SUPPLEMENT_DATASET = "LLM-Semantic-Router/category-classifier-supplement"


class MMLU_Dataset:
    """
    Dataset class for MMLU-Pro category classification fine-tuning.

    By default, loads MMLU-Pro (~12K samples) merged with supplementary data (~653 samples)
    that includes casual "other" category examples for better fallback detection.
    """

    def __init__(
        self,
        dataset_name="TIGER-Lab/MMLU-Pro",
        supplement_dataset: str = DEFAULT_SUPPLEMENT_DATASET,
    ):
        """
        Initialize the dataset loader.

        Args:
            dataset_name: HuggingFace dataset name for MMLU-Pro
            supplement_dataset: HuggingFace dataset ID for supplementary data (set to None to disable)
        """
        self.dataset_name = dataset_name
        self.supplement_dataset = supplement_dataset
        self.label2id = {}
        self.id2label = {}

    def _load_supplement_data(self) -> list:
        """
        Load supplementary training data from HuggingFace Hub.

        Returns:
            List of (text, label) tuples
        """
        if not self.supplement_dataset:
            return []

        try:
            print(f"📥 Loading supplement data from: {self.supplement_dataset}")
            supplement = load_dataset(self.supplement_dataset)

            # Get the train split
            data = (
                supplement["train"]
                if "train" in supplement
                else supplement[list(supplement.keys())[0]]
            )

            samples = [(item["text"], item["label"]) for item in data]
            print(f"✅ Loaded {len(samples)} samples from HuggingFace")
            return samples
        except Exception as e:
            logger.error(f"Failed to load supplement dataset: {e}")
            print(f"❌ Failed to load supplement dataset: {e}")
            return []

    def load_huggingface_dataset(self):
        """Load the MMLU-Pro dataset from HuggingFace and merge with supplement data."""
        logger.info(f"Loading dataset from HuggingFace: {self.dataset_name}")

        try:
            # Load the dataset
            dataset = load_dataset(self.dataset_name)
            logger.info(f"Dataset splits: {dataset.keys()}")

            # Extract questions and categories from the test split
            # Note: MMLU-Pro typically uses 'test' split for training data
            texts = list(dataset["test"]["question"])
            labels = list(dataset["test"]["category"])
            logger.info(f"MMLU-Pro base: {len(texts)} samples")
            print(f"📚 MMLU-Pro base: {len(texts)} samples")

            # Load and merge supplementary training data from HuggingFace Hub
            # This includes casual "other" examples and additional academic samples
            # to improve fallback detection for non-academic queries
            # Dataset: LLM-Semantic-Router/category-classifier-supplement (~653 samples)
            supplement_samples = self._load_supplement_data()
            if supplement_samples:
                supp_texts, supp_labels = zip(*supplement_samples)
                texts.extend(supp_texts)
                labels.extend(supp_labels)
                print(f"➕ Added {len(supplement_samples)} supplement samples")

            logger.info(f"Total dataset size: {len(texts)} samples")
            print(f"📊 Total dataset size: {len(texts)} samples")
            return texts, labels

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            exit(1)

    def split_dataset(
        self,
        texts,
        labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    ):
        """Split the dataset into train, validation, and test sets."""
        assert (
            abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        ), "Ratios must sum to 1.0"

        # Check class distribution
        class_counts = Counter(labels)
        logger.info(f"Class distribution: {dict(class_counts)}")

        # Remove classes with less than 3 samples for stratified splitting
        min_samples_per_class = 3
        filtered_data = []
        for text, label in zip(texts, labels):
            if class_counts[label] >= min_samples_per_class:
                filtered_data.append((text, label))

        if len(filtered_data) < len(texts):
            removed_count = len(texts) - len(filtered_data)
            rare_classes = [
                cls
                for cls, count in class_counts.items()
                if count < min_samples_per_class
            ]
            logger.warning(
                f"Removed {removed_count} samples from rare classes: {rare_classes}"
            )

        # Unpack filtered data
        filtered_texts, filtered_labels = (
            zip(*filtered_data) if filtered_data else ([], [])
        )
        filtered_texts, filtered_labels = list(filtered_texts), list(filtered_labels)

        try:
            # First split: train and temp (val + test)
            train_texts, temp_texts, train_labels, temp_labels = train_test_split(
                filtered_texts,
                filtered_labels,
                test_size=(val_ratio + test_ratio),
                random_state=random_state,
                stratify=filtered_labels,
            )

            # Second split: val and test
            val_size = val_ratio / (val_ratio + test_ratio)
            val_texts, test_texts, val_labels, test_labels = train_test_split(
                temp_texts,
                temp_labels,
                test_size=(1 - val_size),
                random_state=random_state,
                stratify=temp_labels,
            )

        except ValueError as e:
            # Fall back to non-stratified splitting if stratified fails
            logger.warning(f"Stratified split failed: {e}. Using random split instead.")

            train_texts, temp_texts, train_labels, temp_labels = train_test_split(
                filtered_texts,
                filtered_labels,
                test_size=(val_ratio + test_ratio),
                random_state=random_state,
            )

            val_size = val_ratio / (val_ratio + test_ratio)
            val_texts, test_texts, val_labels, test_labels = train_test_split(
                temp_texts,
                temp_labels,
                test_size=(1 - val_size),
                random_state=random_state,
            )

        return {
            "train": (train_texts, train_labels),
            "validation": (val_texts, val_labels),
            "test": (test_texts, test_labels),
        }

    def create_label_mappings(self, all_labels):
        """Create label to ID mappings."""
        unique_labels = sorted(list(set(all_labels)))

        self.label2id = {label: i for i, label in enumerate(unique_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        logger.info(
            f"Created mappings for {len(unique_labels)} labels: {unique_labels}"
        )

    def prepare_datasets(self):
        """Prepare train/validation/test datasets from HuggingFace MMLU-Pro dataset."""

        # Load the full dataset
        logger.info("Loading MMLU-Pro category classification dataset...")
        texts, labels = self.load_huggingface_dataset()

        logger.info(f"Loaded {len(texts)} samples")
        logger.info(
            f"Label distribution: {dict(sorted([(label, labels.count(label)) for label in set(labels)], key=lambda x: x[1], reverse=True))}"
        )

        # Split the dataset
        logger.info("Splitting dataset into train/validation/test...")
        datasets = self.split_dataset(texts, labels)

        train_texts, train_labels = datasets["train"]
        val_texts, val_labels = datasets["validation"]
        test_texts, test_labels = datasets["test"]

        # Create label mappings
        all_labels = train_labels + val_labels + test_labels
        self.create_label_mappings(all_labels)

        # Convert labels to IDs
        train_label_ids = [self.label2id[label] for label in train_labels]
        val_label_ids = [self.label2id[label] for label in val_labels]
        test_label_ids = [self.label2id[label] for label in test_labels]

        return {
            "train": (train_texts, train_label_ids),
            "validation": (val_texts, val_label_ids),
            "test": (test_texts, test_label_ids),
        }


# Function to predict category using the classification model
def predict_category(model, tokenizer, text, idx_to_label_map, device):
    """Predict category for a given text."""
    model.eval()

    # Tokenize input
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = torch.softmax(logits, dim=-1)
    confidence, predicted_idx = torch.max(probabilities, dim=-1)

    predicted_idx = predicted_idx.item()
    confidence = confidence.item()

    predicted_category = idx_to_label_map.get(predicted_idx, "Unknown Category")

    return predicted_category, confidence


# Evaluate on validation set using the classification model
def evaluate_category_classifier(
    model, tokenizer, texts_list, true_label_indices_list, idx_to_label_map, device
):
    """Evaluate the category classifier on a dataset."""
    correct = 0
    total = len(texts_list)
    predictions = []
    true_labels = []

    if total == 0:
        return 0.0, None, None, None

    for text, true_label_idx in zip(texts_list, true_label_indices_list):
        predicted_category, confidence = predict_category(
            model, tokenizer, text, idx_to_label_map, device
        )
        true_category = idx_to_label_map.get(true_label_idx)

        predictions.append(predicted_category)
        true_labels.append(true_category)

        if true_category == predicted_category:
            correct += 1

    accuracy = correct / total

    # Generate classification report
    class_report = classification_report(true_labels, predictions, output_dict=True)
    conf_matrix = confusion_matrix(true_labels, predictions)

    return accuracy, class_report, conf_matrix, (predictions, true_labels)


def main(model_name="minilm", num_epochs=3, batch_size=8):
    """Main function to demonstrate MMLU-Pro category classification fine-tuning.

    Args:
        model_name: Name of the model to use (e.g., 'modernbert-base')
        num_epochs: Number of training epochs
        batch_size: Training and evaluation batch size
    """

    # Validate model name
    if model_name not in MODEL_CONFIGS:
        logger.error(
            f"Unknown model: {model_name}. Available models: {list(MODEL_CONFIGS.keys())}"
        )
        return

    # Set up device (GPU if available)
    device = get_device()

    model_path = MODEL_CONFIGS[model_name]
    logger.info(f"Using model: {model_name} ({model_path})")
    logger.info(f"Training configuration: {num_epochs} epochs, batch size {batch_size}")

    logger.info("Loading MMLU-Pro + supplement dataset...")
    dataset_loader = MMLU_Dataset()  # Uses defaults: MMLU-Pro + supplement
    datasets = dataset_loader.prepare_datasets()

    train_texts, train_categories = datasets["train"]
    val_texts, val_categories = datasets["validation"]
    test_texts, test_categories = datasets["test"]

    unique_categories = list(dataset_loader.label2id.keys())
    category_to_idx = dataset_loader.label2id
    idx_to_category = dataset_loader.id2label

    logger.info(
        f"Found {len(unique_categories)} unique categories: {unique_categories}"
    )
    logger.info(f"Dataset sizes:")
    logger.info(f"  Train: {len(train_texts)}")
    logger.info(f"  Validation: {len(val_texts)}")
    logger.info(f"  Test: {len(test_texts)}")

    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    num_labels = len(unique_categories)

    # Suppress the expected warning about newly initialized classifier weights
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*classifier.*newly initialized.*")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            label2id=category_to_idx,
            id2label=idx_to_category,
        )

    # Move model to device
    model.to(device)

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=512
        )

    # Create datasets
    train_dataset = Dataset.from_dict({"text": train_texts, "labels": train_categories})
    val_dataset = Dataset.from_dict({"text": val_texts, "labels": val_categories})
    test_dataset = Dataset.from_dict({"text": test_texts, "labels": test_categories})

    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Check transformers version compatibility
    eval_strategy_param = check_transformers_compatibility()

    # Training arguments - anti-overfitting measures
    output_model_path = f"category_classifier_{model_name}_model"

    # Calculate adaptive parameters based on dataset size
    train_size = len(train_texts)
    effective_batch_size = min(
        batch_size, 8
    )  # Smaller batches for better regularization
    effective_epochs = min(num_epochs, 3)  # Prevent overfitting with fewer epochs

    training_args_dict = {
        "output_dir": output_model_path,
        "num_train_epochs": effective_epochs,
        "per_device_train_batch_size": effective_batch_size,
        "per_device_eval_batch_size": effective_batch_size,
        "learning_rate": 2e-5,  # Lower learning rate for stability
        "warmup_steps": min(100, train_size // (effective_batch_size * 2)),
        "weight_decay": 0.1,  # Higher weight decay for regularization
        "logging_dir": f"{output_model_path}/logs",
        "logging_steps": 50,
        eval_strategy_param: "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "save_total_limit": 2,
        "report_to": [],
        "dataloader_drop_last": False,
        "eval_steps": 50,
        "fp16": torch.cuda.is_available(),  # Use mixed precision if GPU available
        "gradient_accumulation_steps": 2,  # Effective larger batch size
    }

    training_args = TrainingArguments(**training_args_dict)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info(
        f"Starting MMLU-Pro category classification fine-tuning with {model_name}..."
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    trainer.save_model(output_model_path)
    tokenizer.save_pretrained(output_model_path)

    # Save the label mapping
    mapping_path = os.path.join(output_model_path, "category_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(
            {
                "category_to_idx": category_to_idx,
                "idx_to_category": {str(k): v for k, v in idx_to_category.items()},
            },
            f,
        )

    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    val_accuracy, val_report, val_conf_matrix, val_predictions = (
        evaluate_category_classifier(
            model, tokenizer, val_texts, val_categories, idx_to_category, device
        )
    )
    logger.info(f"Validation accuracy: {val_accuracy:.4f}")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_accuracy, test_report, test_conf_matrix, test_predictions = (
        evaluate_category_classifier(
            model, tokenizer, test_texts, test_categories, idx_to_category, device
        )
    )
    logger.info(f"Test accuracy: {test_accuracy:.4f}")

    # Save evaluation results
    results_path = os.path.join(output_model_path, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "validation_accuracy": val_accuracy,
                "test_accuracy": test_accuracy,
                "validation_report": val_report,
                "test_report": test_report,
                "validation_confusion_matrix": (
                    val_conf_matrix.tolist() if val_conf_matrix is not None else None
                ),
                "test_confusion_matrix": (
                    test_conf_matrix.tolist() if test_conf_matrix is not None else None
                ),
            },
            f,
            indent=2,
        )

    # Print final results
    print("\n" + "=" * 50)
    print("MMLU-Pro Category Classification Fine-tuning Completed!")
    print("=" * 50)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    if val_report:
        print("\nValidation Classification Report:")
        for label, metrics in val_report.items():
            if isinstance(metrics, dict):
                print(
                    f"{label}: Precision={metrics.get('precision', 0):.3f}, Recall={metrics.get('recall', 0):.3f}, F1={metrics.get('f1-score', 0):.3f}"
                )

    return model, tokenizer, idx_to_category


def demo_inference(model_name="minilm"):
    """Demonstrate inference with the trained model."""

    # Set up device (GPU if available)
    device = get_device()

    model_path = f"./category_classifier_{model_name}_model"
    if not Path(model_path).exists():
        logger.error(
            f"Trained model not found at {model_path}. Please run training first with --model {model_name}"
        )
        return

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)

    mapping_path = os.path.join(model_path, "category_mapping.json")
    with open(mapping_path, "r") as f:
        mappings = json.load(f)
        idx_to_label = {int(k): v for k, v in mappings["idx_to_category"].items()}

    print("\n" + "=" * 50)
    print("MMLU-Pro Category Classification Test")
    print("=" * 50)

    # Test cases covering various academic domains
    test_cases = [
        {
            "text": "What is the derivative of x^2?",
            "expected": "mathematics",
            "description": "Basic calculus question",
        },
        {
            "text": "Explain the concept of supply and demand in economics.",
            "expected": "economics",
            "description": "Economic principles",
        },
        {
            "text": "How does DNA replication work in eukaryotic cells?",
            "expected": "biology",
            "description": "Molecular biology",
        },
        {
            "text": "What is the difference between a civil law and common law system?",
            "expected": "law",
            "description": "Legal systems",
        },
        {
            "text": "Explain how transistors work in computer processors.",
            "expected": "engineering",
            "description": "Computer engineering",
        },
        {
            "text": "Why do stars twinkle?",
            "expected": "physics",
            "description": "Astronomy/physics",
        },
        {
            "text": "How do I create a balanced portfolio for retirement?",
            "expected": "economics",
            "description": "Financial planning",
        },
        {
            "text": "What causes mental illnesses?",
            "expected": "psychology",
            "description": "Psychology/medicine",
        },
        {
            "text": "How do computer algorithms work?",
            "expected": "computer science",
            "description": "Computer science",
        },
        {
            "text": "Explain the historical significance of the Roman Empire.",
            "expected": "history",
            "description": "Ancient history",
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        text = test_case["text"]
        expected = test_case["expected"]
        description = test_case["description"]

        predicted_category, confidence = predict_category(
            model, tokenizer, text, idx_to_label, device
        )

        print(f"\nTest Case {i}: {description}")
        print(f"Question: {text}")
        print(f"Expected: {expected}")
        print(f"Predicted: {predicted_category} (Confidence: {confidence:.4f})")
        print("-" * 60)

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MMLU-Pro Category Classification Fine-tuning"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        default="train",
        help="Mode: 'train' to fine-tune model, 'test' to run inference",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CONFIGS.keys(),
        default="minilm",
        help="Model to use for fine-tuning (e.g., bert-base, roberta-base, modernbert-base, etc.)",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training and evaluation batch size (default: 8)",
    )
    args = parser.parse_args()

    if args.mode == "train":
        main(args.model, args.epochs, args.batch_size)
    elif args.mode == "test":
        demo_inference(args.model)
