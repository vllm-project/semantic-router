#!/usr/bin/env python3
"""
Data loader for ML model selection training.

Loads benchmark data from HuggingFace or local JSONL file.
Reference: FusionFactory (arXiv:2507.10540), Avengers-Pro (arXiv:2508.12631)
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

# HuggingFace dataset info
HF_DATASET_REPO = "vllm-project/semantic-router-benchmark"
HF_DATASET_FILE = "benchmark_training_data.jsonl"

# Category list for one-hot encoding (14 categories)
CATEGORIES = [
    "math",
    "physics",
    "chemistry",
    "biology",
    "computer science",
    "history",
    "economics",
    "business",
    "law",
    "health",
    "psychology",
    "philosophy",
    "other",
    "unknown",
]


@dataclass
class RoutingRecord:
    """A single routing record from benchmark data."""

    query: str
    category: str
    model_name: str
    quality: float  # performance score (0-1)
    latency_ms: float  # response_time in milliseconds


@dataclass
class TrainingRecord:
    """A training record with features."""

    query: str
    category: str
    model_name: str
    quality: float
    latency_ms: float
    embedding: np.ndarray | None = None
    feature_vector: np.ndarray | None = None


def download_data(cache_dir: str = ".cache/ml_model_selection") -> Path:
    """
    Download benchmark data from HuggingFace.

    Args:
        cache_dir: Directory to cache downloaded files

    Returns:
        Path to the downloaded file
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    local_file = cache_path / HF_DATASET_FILE

    if local_file.exists():
        print(f"Using cached data: {local_file}")
        return local_file

    print(f"Downloading data from HuggingFace: {HF_DATASET_REPO}")
    try:
        downloaded_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=HF_DATASET_FILE,
            repo_type="dataset",
            local_dir=str(cache_path),
        )
        print(f"Downloaded to: {downloaded_path}")
        return Path(downloaded_path)
    except Exception as e:
        print(f"⚠ Failed to download from HuggingFace: {e}")
        print("  Please provide a local file using --data-file")
        raise


def load_jsonl(file_path: Path) -> list[RoutingRecord]:
    """
    Load benchmark data from JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of RoutingRecord objects
    """
    records = []

    with open(file_path, encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # Extract fields
                query = data.get("query", "")
                category = data.get("category", "other")
                model_name = data.get("model_name", "")
                performance = data.get("performance", 0.0)
                response_time = data.get("response_time", 0.0)

                if not query or not model_name:
                    continue

                # Normalize category
                if category not in CATEGORIES:
                    category = "other"

                records.append(
                    RoutingRecord(
                        query=query,
                        category=category,
                        model_name=model_name,
                        quality=float(performance),
                        latency_ms=float(response_time) * 1000,  # Convert to ms
                    )
                )

            except json.JSONDecodeError as e:
                print(f"⚠ Skipping invalid JSON at line {line_num}: {e}")
            except Exception as e:
                print(f"⚠ Error processing line {line_num}: {e}")

    print(f"Loaded {len(records)} records from {file_path}")
    return records


def get_unique_queries(records: list[RoutingRecord]) -> list[str]:
    """Get unique queries from records."""
    seen = set()
    unique = []
    for r in records:
        if r.query not in seen:
            seen.add(r.query)
            unique.append(r.query)
    return unique


def get_model_names(records: list[RoutingRecord]) -> list[str]:
    """Get unique model names from records."""
    return sorted({r.model_name for r in records})


def category_to_onehot(category: str) -> np.ndarray:
    """Convert category to one-hot vector."""
    onehot = np.zeros(len(CATEGORIES), dtype=np.float32)
    if category in CATEGORIES:
        idx = CATEGORIES.index(category)
        onehot[idx] = 1.0
    else:
        # Default to "other"
        onehot[CATEGORIES.index("other")] = 1.0
    return onehot


def create_feature_vector(embedding: np.ndarray, category: str) -> np.ndarray:
    """
    Create feature vector by concatenating embedding and category one-hot.

    Args:
        embedding: Query embedding (1024-dim for Qwen3)
        category: Category name

    Returns:
        Feature vector (1038-dim = 1024 + 14)
    """
    onehot = category_to_onehot(category)
    return np.concatenate([embedding, onehot])


def group_by_query(
    records: list[RoutingRecord],
) -> dict[str, list[RoutingRecord]]:
    """Group records by query."""
    groups: dict[str, list[RoutingRecord]] = {}
    for r in records:
        if r.query not in groups:
            groups[r.query] = []
        groups[r.query].append(r)
    return groups


def print_data_stats(records: list[RoutingRecord]) -> None:
    """Print statistics about the loaded data."""
    queries = get_unique_queries(records)
    models = get_model_names(records)
    categories = {r.category for r in records}

    print("\n" + "=" * 50)
    print("  Data Statistics")
    print("=" * 50)
    print(f"  Total records:    {len(records)}")
    print(f"  Unique queries:   {len(queries)}")
    print(f"  Models:           {', '.join(models)}")
    print(f"  Categories:       {len(categories)}")

    # Category distribution
    print("\n  Category distribution:")
    cat_counts = {}
    for r in records:
        cat_counts[r.category] = cat_counts.get(r.category, 0) + 1
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    - {cat}: {count}")

    # Model quality stats
    print("\n  Model quality (avg):")
    model_quality = {}
    model_counts = {}
    for r in records:
        if r.model_name not in model_quality:
            model_quality[r.model_name] = 0.0
            model_counts[r.model_name] = 0
        model_quality[r.model_name] += r.quality
        model_counts[r.model_name] += 1
    for model in sorted(models):
        avg = model_quality[model] / model_counts[model]
        print(f"    - {model}: {avg:.3f}")

    print("=" * 50 + "\n")
