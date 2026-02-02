#!/usr/bin/env python3
"""
Main training script for ML model selection.

Trains KNN, KMeans, and SVM models for query-based LLM routing.

Usage:
    python train.py --data-file benchmark_data.jsonl --output-dir models/

Reference:
- FusionFactory (arXiv:2507.10540) - Query-level fusion via LLM routers
- Avengers-Pro (arXiv:2508.12631) - Performance-efficiency optimized routing
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from data_loader import (
    CATEGORIES,
    RoutingRecord,
    category_to_onehot,
    create_feature_vector,
    download_data,
    find_best_model_per_query,
    get_model_names,
    get_unique_queries,
    group_by_query,
    load_jsonl,
    print_data_stats,
)
from embeddings import EmbeddingGenerator, generate_embeddings_for_queries
from models import KMeansModel, KNNModel, SVMModel, TrainingSample


def create_training_samples(
    records: List[RoutingRecord],
    embeddings: Dict[str, np.ndarray],
    quality_weight: float = 0.9,
) -> List[TrainingSample]:
    """
    Create training samples from records and embeddings.

    Args:
        records: List of routing records
        embeddings: Dict mapping query -> embedding
        quality_weight: Weight for quality in best model selection

    Returns:
        List of TrainingSample objects
    """
    samples = []

    for record in records:
        if record.query not in embeddings:
            continue

        # Create feature vector (embedding + category one-hot)
        embedding = embeddings[record.query]
        feature_vector = create_feature_vector(embedding, record.category)

        samples.append(
            TrainingSample(
                feature_vector=feature_vector,
                model_name=record.model_name,
                quality=record.quality,
                latency_ms=record.latency_ms,
            )
        )

    return samples


def train_models(
    samples: List[TrainingSample],
    output_dir: Path,
    knn_k: int = 5,
    kmeans_clusters: int = 8,
    svm_kernel: str = "rbf",
    svm_gamma: float = 1.0,
) -> None:
    """
    Train all models (KNN, KMeans, SVM).

    Args:
        samples: List of training samples
        output_dir: Directory to save models
        knn_k: Number of neighbors for KNN
        kmeans_clusters: Number of clusters for KMeans
        svm_kernel: SVM kernel type
        svm_gamma: SVM gamma parameter
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 50)
    print("  Training ML Models")
    print("=" * 50)
    print(f"  Samples: {len(samples)}")
    print(f"  Feature dim: {len(samples[0].feature_vector)}")
    print(f"  Models: {sorted(set(s.model_name for s in samples))}")
    print("=" * 50 + "\n")

    # Train KNN
    print("[1/3] Training KNN...")
    knn = KNNModel(k=knn_k)
    knn.train(samples)
    knn.save(str(output_dir / "knn_model.json"))

    # Train KMeans
    print("[2/3] Training KMeans...")
    kmeans = KMeansModel(n_clusters=kmeans_clusters, efficiency_weight=0.1)
    kmeans.train(samples)
    kmeans.save(str(output_dir / "kmeans_model.json"))

    # Train SVM
    print("[3/3] Training SVM...")
    svm = SVMModel(kernel=svm_kernel, gamma=svm_gamma)
    svm.train(samples)
    svm.save(str(output_dir / "svm_model.json"))

    print("\n✓ All models trained and saved to", output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Train ML models for LLM routing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with local data file
  python train.py --data-file benchmark_data.jsonl --output-dir models/

  # Download data from HuggingFace and train
  python train.py --output-dir models/

  # Use custom embedding model
  python train.py --data-file data.jsonl --embedding-model bge
        """,
    )

    parser.add_argument(
        "--data-file",
        type=str,
        help="Path to JSONL data file (downloads from HuggingFace if not specified)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained models (default: models/)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="qwen3",
        help="Embedding model: qwen3, gte, mpnet, e5, bge (default: qwen3)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".cache/ml_model_selection",
        help="Cache directory for downloads and embeddings",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=5,
        help="Number of neighbors for KNN (default: 5)",
    )
    parser.add_argument(
        "--kmeans-clusters",
        type=int,
        default=8,
        help="Number of clusters for KMeans (default: 8)",
    )
    parser.add_argument(
        "--svm-kernel",
        type=str,
        default="rbf",
        choices=["rbf", "linear"],
        help="SVM kernel type (default: rbf)",
    )
    parser.add_argument(
        "--svm-gamma",
        type=float,
        default=1.0,
        help="SVM gamma parameter for RBF kernel (default: 1.0)",
    )
    parser.add_argument(
        "--quality-weight",
        type=float,
        default=0.9,
        help="Quality weight for best model selection (default: 0.9)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for embedding model (default: cpu)",
    )

    args = parser.parse_args()

    start_time = time.time()

    print("\n" + "=" * 60)
    print("  ML Model Selection Training")
    print("=" * 60)
    print(f"  Output dir:       {args.output_dir}")
    print(f"  Embedding model:  {args.embedding_model}")
    print(f"  Quality weight:   {args.quality_weight}")
    print(f"  KNN k:            {args.knn_k}")
    print(f"  KMeans clusters:  {args.kmeans_clusters}")
    print(f"  SVM kernel:       {args.svm_kernel} (gamma={args.svm_gamma})")
    print("=" * 60 + "\n")

    # Load data
    print("[1/4] Loading data...")
    if args.data_file:
        data_path = Path(args.data_file)
    else:
        data_path = download_data(args.cache_dir)

    records = load_jsonl(data_path)
    print_data_stats(records)

    # Generate embeddings
    print("[2/4] Generating embeddings...")
    queries = get_unique_queries(records)
    print(f"  {len(queries)} unique queries")

    cache_file = Path(args.cache_dir) / f"embeddings_{args.embedding_model}.npz"
    embeddings = generate_embeddings_for_queries(
        queries,
        model_name=args.embedding_model,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        cache_file=str(cache_file),
    )
    print(f"  Embedding dim: {len(next(iter(embeddings.values())))}")

    # Create training samples
    print("[3/4] Creating training samples...")
    samples = create_training_samples(records, embeddings, args.quality_weight)
    print(f"  Created {len(samples)} training samples")

    # Train models
    print("[4/4] Training models...")
    output_dir = Path(args.output_dir)
    train_models(
        samples,
        output_dir,
        knn_k=args.knn_k,
        kmeans_clusters=args.kmeans_clusters,
        svm_kernel=args.svm_kernel,
        svm_gamma=args.svm_gamma,
    )

    elapsed = time.time() - start_time
    print(f"\n✅ Training complete in {elapsed:.1f}s")
    print(f"   Models saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
