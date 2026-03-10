"""
Common Utilities for Cache Embedding Training
=============================================

Shared utilities and helper functions for cache-specific embedding training.
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration for training scripts.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    return logger


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across numpy, torch, and random.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the appropriate device for training (CPU/GPU).

    Args:
        prefer_gpu: Whether to prefer GPU if available

    Returns:
        torch.device object
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU for training")
    return device


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between two tensors.

    Args:
        a: First tensor (embeddings)
        b: Second tensor (embeddings)

    Returns:
        Cosine similarity scores
    """
    a_norm = torch.nn.functional.normalize(a, p=2, dim=-1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=-1)
    return torch.sum(a_norm * b_norm, dim=-1)


def euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute Euclidean distance between two tensors.

    Args:
        a: First tensor (embeddings)
        b: Second tensor (embeddings)

    Returns:
        Euclidean distances
    """
    return torch.nn.functional.pairwise_distance(a, b, p=2)


def save_model_artifacts(
    model: torch.nn.Module,
    tokenizer,
    output_dir: Path,
    config: Dict,
    metrics: Optional[Dict] = None,
):
    """
    Save model, tokenizer, and training artifacts.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        output_dir: Output directory
        config: Training configuration
        metrics: Optional evaluation metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save config
    import json

    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save metrics if provided
    if metrics:
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    logging.info(f"Model artifacts saved to {output_dir}")


def load_jsonl(file_path: Path) -> List[Dict]:
    """
    Load data from a JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of dictionaries
    """
    import json

    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl(data: List[Dict], file_path: Path):
    """
    Save data to a JSONL file.

    Args:
        data: List of dictionaries
        file_path: Output file path
    """
    import json

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def calculate_cache_metrics(
    similarities: List[float], labels: List[int], threshold: float = 0.85
) -> Dict[str, float]:
    """
    Calculate cache-specific metrics (precision, recall, F1).

    Args:
        similarities: Predicted similarity scores
        labels: Ground truth labels (1 = should cache, 0 = should not)
        threshold: Similarity threshold for cache hit

    Returns:
        Dictionary of metrics
    """
    predictions = [1 if s >= threshold else 0 for s in similarities]

    tp = sum(p == 1 and l == 1 for p, l in zip(predictions, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(predictions, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(predictions, labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(predictions, labels))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "threshold": threshold,
    }


def format_training_time(seconds: float) -> str:
    """
    Format training time in human-readable format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "2h 34m 12s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
