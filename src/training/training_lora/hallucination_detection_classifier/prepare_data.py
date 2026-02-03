#!/usr/bin/env python3
"""
Standalone data preparation script for hallucination detection fine-tuning.

This script downloads and combines RAGTruth, DART, and E2E datasets into a single
training dataset. Based on the research in TRAINING_32K.md, the best configuration
is Stage 3: RAGTruth + DART + E2E = 76.56% F1.

Datasets:
- RAGTruth: ~17,790 samples (human-annotated) - Required
- DART: ~2,000 samples (LLM-generated Data2txt) - Recommended
- E2E: ~1,500 samples (LLM-generated Data2txt) - Recommended

Usage:
    # Download and prepare all datasets
    python prepare_data.py --output-dir ./data

    # Use existing local files
    python prepare_data.py \
        --ragtruth-path ./ragtruth_data.json \
        --dart-path ./dart_spans.json \
        --e2e-path ./e2e_spans.json \
        --output-dir ./data

    # Download from HuggingFace only (DART/E2E)
    python prepare_data.py \
        --ragtruth-path ./ragtruth_data.json \
        --download-augmentation \
        --output-dir ./data

Dependencies:
    pip install datasets tqdm
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Optional

try:
    from datasets import load_dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print(
        "Warning: 'datasets' library not installed. Cannot download from HuggingFace."
    )
    print("Install with: pip install datasets")

from tqdm import tqdm


def download_ragtruth(output_dir: Path) -> Optional[Path]:
    """
    Download and preprocess RAGTruth dataset.

    RAGTruth needs to be downloaded from GitHub and preprocessed.
    This function provides instructions for manual download.

    Returns:
        Path to the preprocessed RAGTruth file, or None if not available.
    """
    ragtruth_path = output_dir / "ragtruth_data.json"

    if ragtruth_path.exists():
        print(f"✓ RAGTruth found at: {ragtruth_path}")
        return ragtruth_path

    # Check if raw files exist
    raw_dir = output_dir / "ragtruth_raw"
    response_file = raw_dir / "response.jsonl"
    source_file = raw_dir / "source_info.jsonl"

    if response_file.exists() and source_file.exists():
        print("Processing RAGTruth from raw files...")
        return preprocess_ragtruth_raw(raw_dir, output_dir)

    print("\n" + "=" * 60)
    print("RAGTruth dataset not found!")
    print("=" * 60)
    print("\nRAGTruth requires manual download from GitHub:")
    print("  1. Clone: git clone https://github.com/ParticleMedia/RAGTruth.git")
    print("  2. Copy files to: " + str(raw_dir))
    print("     - response.jsonl")
    print("     - source_info.jsonl")
    print("\nOr provide a pre-processed file with --ragtruth-path")
    print("=" * 60 + "\n")

    return None


def preprocess_ragtruth_raw(raw_dir: Path, output_dir: Path) -> Path:
    """
    Preprocess RAGTruth from raw JSONL files.

    Args:
        raw_dir: Directory containing response.jsonl and source_info.jsonl
        output_dir: Output directory for processed data

    Returns:
        Path to the preprocessed JSON file
    """
    print("Loading RAGTruth raw files...")

    # Load responses
    responses = []
    with open(raw_dir / "response.jsonl", "r") as f:
        for line in f:
            responses.append(json.loads(line))

    # Load sources
    sources = []
    with open(raw_dir / "source_info.jsonl", "r") as f:
        for line in f:
            sources.append(json.loads(line))

    sources_by_id = {source["source_id"]: source for source in sources}

    # Create samples
    samples = []
    for response in tqdm(responses, desc="Processing RAGTruth"):
        source = sources_by_id[response["source_id"]]

        labels = []
        for label in response.get("labels", []):
            labels.append(
                {
                    "start": label["start"],
                    "end": label["end"],
                    "label": label["label_type"],
                }
            )

        sample = {
            "prompt": source["prompt"],
            "answer": response["response"],
            "labels": labels,
            "split": response["split"],
            "task_type": source["task_type"],
            "dataset": "ragtruth",
            "language": "en",
        }
        samples.append(sample)

    # Save
    output_path = output_dir / "ragtruth_data.json"
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"✓ Saved {len(samples)} RAGTruth samples to: {output_path}")
    return output_path


def download_dart_e2e_from_huggingface(
    output_dir: Path,
) -> tuple[Optional[Path], Optional[Path]]:
    """
    Download DART and E2E datasets from HuggingFace.

    These datasets are available at:
    - llm-semantic-router/dart-halspans
    - llm-semantic-router/e2e-halspans

    Returns:
        Tuple of (dart_path, e2e_path) or (None, None) if unavailable
    """
    if not HAS_DATASETS:
        print("Cannot download from HuggingFace: 'datasets' library not installed")
        return None, None

    dart_path = None
    e2e_path = None

    # Download DART
    try:
        print("\nDownloading DART from HuggingFace...")
        dart_ds = load_dataset("llm-semantic-router/dart-halspans", split="train")

        dart_samples = []
        for item in tqdm(dart_ds, desc="Processing DART"):
            sample = {
                "prompt": item["prompt"],
                "answer": item["answer"],
                "labels": item["labels"] if item["labels"] else [],
                "split": item.get("split", "train"),
                "task_type": item.get("task_type", "Data2txt"),
                "dataset": "dart_synthetic",
                "language": "en",
            }
            dart_samples.append(sample)

        dart_path = output_dir / "dart_spans.json"
        with open(dart_path, "w") as f:
            json.dump(dart_samples, f, indent=2)
        print(f"✓ Saved {len(dart_samples)} DART samples to: {dart_path}")

    except Exception as e:
        print(f"Warning: Could not download DART: {e}")

    # Download E2E
    try:
        print("\nDownloading E2E from HuggingFace...")
        e2e_ds = load_dataset("llm-semantic-router/e2e-halspans", split="train")

        e2e_samples = []
        for item in tqdm(e2e_ds, desc="Processing E2E"):
            sample = {
                "prompt": item["prompt"],
                "answer": item["answer"],
                "labels": item["labels"] if item["labels"] else [],
                "split": item.get("split", "train"),
                "task_type": item.get("task_type", "Data2txt"),
                "dataset": "e2e_synthetic",
                "language": "en",
            }
            e2e_samples.append(sample)

        e2e_path = output_dir / "e2e_spans.json"
        with open(e2e_path, "w") as f:
            json.dump(e2e_samples, f, indent=2)
        print(f"✓ Saved {len(e2e_samples)} E2E samples to: {e2e_path}")

    except Exception as e:
        print(f"Warning: Could not download E2E: {e}")

    return dart_path, e2e_path


def load_dataset_file(path: Path) -> list[dict]:
    """Load a dataset from a JSON file."""
    if not path.exists():
        print(f"Warning: File not found: {path}")
        return []

    with open(path, "r") as f:
        data = json.load(f)

    print(f"✓ Loaded {len(data)} samples from: {path}")
    return data


def combine_datasets(
    ragtruth_data: list[dict],
    dart_data: list[dict],
    e2e_data: list[dict],
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Combine datasets and split into train/dev/test.

    Based on TRAINING_32K.md:
    - RAGTruth: Use as-is with original splits
    - DART: All as training data
    - E2E: All as training data

    Args:
        ragtruth_data: RAGTruth samples
        dart_data: DART samples
        e2e_data: E2E samples
        seed: Random seed for shuffling

    Returns:
        Tuple of (train_samples, dev_samples, test_samples)
    """
    random.seed(seed)

    # Separate RAGTruth by split
    ragtruth_train = [s for s in ragtruth_data if s.get("split") == "train"]
    ragtruth_test = [s for s in ragtruth_data if s.get("split") == "test"]

    # Create dev split from RAGTruth train (10%)
    random.shuffle(ragtruth_train)
    dev_size = int(len(ragtruth_train) * 0.1)
    ragtruth_dev = ragtruth_train[:dev_size]
    ragtruth_train = ragtruth_train[dev_size:]

    # Combine training data
    train_samples = ragtruth_train.copy()

    # Add DART and E2E (all as training)
    for sample in dart_data:
        sample["split"] = "train"
        train_samples.append(sample)

    for sample in e2e_data:
        sample["split"] = "train"
        train_samples.append(sample)

    # Shuffle training data
    random.shuffle(train_samples)

    return train_samples, ragtruth_dev, ragtruth_test


def analyze_dataset(samples: list[dict], name: str):
    """Print statistics about a dataset."""
    if not samples:
        print(f"\n{name}: No samples")
        return

    total = len(samples)
    with_labels = sum(1 for s in samples if s.get("labels"))
    hal_rate = with_labels / total * 100 if total > 0 else 0

    # Count by task type
    task_types = {}
    for s in samples:
        task = s.get("task_type", "unknown")
        task_types[task] = task_types.get(task, 0) + 1

    # Count by dataset source
    sources = {}
    for s in samples:
        source = s.get("dataset", "unknown")
        sources[source] = sources.get(source, 0) + 1

    print(f"\n{name}:")
    print(f"  Total samples: {total}")
    print(f"  Hallucinated: {with_labels} ({hal_rate:.1f}%)")
    print(f"  Task types: {task_types}")
    print(f"  Sources: {sources}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare hallucination detection training data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory for combined data",
    )
    parser.add_argument(
        "--ragtruth-path",
        type=str,
        help="Path to preprocessed RAGTruth JSON file",
    )
    parser.add_argument(
        "--dart-path",
        type=str,
        help="Path to DART spans JSON file",
    )
    parser.add_argument(
        "--e2e-path",
        type=str,
        help="Path to E2E spans JSON file",
    )
    parser.add_argument(
        "--download-augmentation",
        action="store_true",
        help="Download DART and E2E from HuggingFace",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("HALLUCINATION DETECTION DATA PREPARATION")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Load RAGTruth
    ragtruth_data = []
    if args.ragtruth_path:
        ragtruth_data = load_dataset_file(Path(args.ragtruth_path))
    else:
        ragtruth_path = download_ragtruth(output_dir)
        if ragtruth_path:
            ragtruth_data = load_dataset_file(ragtruth_path)

    if not ragtruth_data:
        print("\nError: RAGTruth data is required!")
        print("Please provide --ragtruth-path or set up RAGTruth manually.")
        return

    # Load DART and E2E
    dart_data = []
    e2e_data = []

    if args.dart_path:
        dart_data = load_dataset_file(Path(args.dart_path))
    if args.e2e_path:
        e2e_data = load_dataset_file(Path(args.e2e_path))

    # Download from HuggingFace if requested
    if args.download_augmentation and (not dart_data or not e2e_data):
        print("\nDownloading augmentation datasets from HuggingFace...")
        dart_hf, e2e_hf = download_dart_e2e_from_huggingface(output_dir)

        if not dart_data and dart_hf:
            dart_data = load_dataset_file(dart_hf)
        if not e2e_data and e2e_hf:
            e2e_data = load_dataset_file(e2e_hf)

    # Combine datasets
    print("\n" + "-" * 60)
    print("COMBINING DATASETS")
    print("-" * 60)

    train_samples, dev_samples, test_samples = combine_datasets(
        ragtruth_data, dart_data, e2e_data, seed=args.seed
    )

    # Analyze datasets
    print("\n" + "-" * 60)
    print("DATASET STATISTICS")
    print("-" * 60)

    analyze_dataset(train_samples, "Training Set")
    analyze_dataset(dev_samples, "Development Set (RAGTruth only)")
    analyze_dataset(test_samples, "Test Set (RAGTruth only)")

    # Save combined datasets
    print("\n" + "-" * 60)
    print("SAVING DATASETS")
    print("-" * 60)

    # Save train
    train_path = output_dir / "train.json"
    with open(train_path, "w") as f:
        json.dump(train_samples, f, indent=2)
    print(f"✓ Saved {len(train_samples)} training samples to: {train_path}")

    # Save dev
    dev_path = output_dir / "dev.json"
    with open(dev_path, "w") as f:
        json.dump(dev_samples, f, indent=2)
    print(f"✓ Saved {len(dev_samples)} dev samples to: {dev_path}")

    # Save test
    test_path = output_dir / "test.json"
    with open(test_path, "w") as f:
        json.dump(test_samples, f, indent=2)
    print(f"✓ Saved {len(test_samples)} test samples to: {test_path}")

    # Save combined (train + dev for compatibility)
    combined_path = output_dir / "combined_train.json"
    with open(combined_path, "w") as f:
        json.dump(train_samples + dev_samples, f, indent=2)
    print(
        f"✓ Saved {len(train_samples) + len(dev_samples)} combined samples to: {combined_path}"
    )

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nTo train the model, run:")
    print(f"  python finetune.py \\")
    print(f"      --train-path {train_path} \\")
    print(f"      --dev-path {dev_path} \\")
    print(f"      --test-path {test_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
