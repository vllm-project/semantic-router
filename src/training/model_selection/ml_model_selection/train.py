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
from models import (
    KMeansModel,
    KNNModel,
    MLPModel,
    SVMModel,
    TrainingSample,
    TORCH_AVAILABLE,
)


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
    mlp_hidden_sizes: List[int] = None,
    mlp_epochs: int = 100,
    mlp_learning_rate: float = 0.001,
    mlp_dropout: float = 0.1,
    device: str = "cpu",
    skip_mlp: bool = False,
    algorithm: str = "all",
) -> None:
    """
    Train models (KNN, KMeans, SVM, MLP).

    Args:
        samples: List of training samples
        output_dir: Directory to save models
        knn_k: Number of neighbors for KNN
        kmeans_clusters: Number of clusters for KMeans
        svm_kernel: SVM kernel type
        svm_gamma: SVM gamma parameter
        mlp_hidden_sizes: Hidden layer sizes for MLP
        mlp_epochs: Training epochs for MLP
        mlp_learning_rate: Learning rate for MLP
        mlp_dropout: Dropout rate for MLP
        device: Device for MLP training (cpu, cuda, mps)
        skip_mlp: Skip MLP training (if PyTorch not available)
        algorithm: Which algorithm to train: all, knn, kmeans, svm, mlp
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if mlp_hidden_sizes is None:
        mlp_hidden_sizes = [256, 128]

    # Determine which algorithms to train
    train_knn = algorithm in ["all", "knn"]
    train_kmeans = algorithm in ["all", "kmeans"]
    train_svm = algorithm in ["all", "svm"]
    train_mlp_flag = algorithm in ["all", "mlp"]

    # Check if MLP is available
    if train_mlp_flag and not TORCH_AVAILABLE:
        print("Warning: MLP requested but PyTorch not available. Skipping MLP.")
        train_mlp_flag = False
    if train_mlp_flag and skip_mlp:
        print("Warning: MLP requested but --skip-mlp specified. Skipping MLP.")
        train_mlp_flag = False

    # Count models to train
    algorithms_to_train = []
    if train_knn:
        algorithms_to_train.append("KNN")
    if train_kmeans:
        algorithms_to_train.append("KMeans")
    if train_svm:
        algorithms_to_train.append("SVM")
    if train_mlp_flag:
        algorithms_to_train.append("MLP")

    num_models = len(algorithms_to_train)

    if num_models == 0:
        print("Error: No algorithms to train!")
        return

    print("\n" + "=" * 50)
    print("  Training ML Models")
    print("=" * 50)
    print(f"  Samples: {len(samples)}")
    print(f"  Feature dim: {len(samples[0].feature_vector)}")
    print(f"  Models: {sorted(set(s.model_name for s in samples))}")
    print(f"  Algorithms: {', '.join(algorithms_to_train)}")
    print("=" * 50 + "\n")

    step = 0

    # Train KNN
    if train_knn:
        step += 1
        print(f"[{step}/{num_models}] Training KNN...")
        knn = KNNModel(k=knn_k)
        knn.train(samples)
        knn.save(str(output_dir / "knn_model.json"))

    # Train KMeans
    if train_kmeans:
        step += 1
        print(f"[{step}/{num_models}] Training KMeans...")
        kmeans = KMeansModel(n_clusters=kmeans_clusters, efficiency_weight=0.1)
        kmeans.train(samples)
        kmeans.save(str(output_dir / "kmeans_model.json"))

    # Train SVM
    if train_svm:
        step += 1
        print(f"[{step}/{num_models}] Training SVM...")
        svm = SVMModel(kernel=svm_kernel, gamma=svm_gamma)
        svm.train(samples)
        svm.save(str(output_dir / "svm_model.json"))

    # Train MLP
    if train_mlp_flag:
        step += 1
        print(f"[{step}/{num_models}] Training MLP (device: {device})...")
        mlp = MLPModel(
            hidden_sizes=mlp_hidden_sizes,
            learning_rate=mlp_learning_rate,
            epochs=mlp_epochs,
            dropout=mlp_dropout,
            device=device,
        )
        mlp.train(samples)
        mlp.save(str(output_dir / "mlp_model.json"))

    print(f"\n✓ {num_models} model(s) trained and saved to", output_dir)


def run_training_pipeline(
    data_file: str,
    output_dir: str,
    embedding_model: str = "qwen3",
    cache_dir: str = ".cache/ml_model_selection",
    quality_weight: float = 0.9,
    batch_size: int = 32,
    device: str = "cpu",
    knn_k: int = 5,
    kmeans_clusters: int = 8,
    svm_kernel: str = "rbf",
    svm_gamma: float = 1.0,
    mlp_hidden_sizes: List[int] = None,
    mlp_epochs: int = 100,
    mlp_learning_rate: float = 0.001,
    mlp_dropout: float = 0.1,
    skip_mlp: bool = False,
    algorithm: str = "all",
    on_progress=None,
) -> List[str]:
    """
    Run the full training pipeline: load data -> embed -> create samples -> train.

    This is the shared entry point used by both the CLI (main()) and the
    HTTP service (server.py).

    Args:
        data_file: Path to JSONL benchmark data (or None to download from HF).
        output_dir: Directory to write model files.
        embedding_model: Embedding model alias (default: qwen3).
        cache_dir: Cache directory for downloads and embeddings.
        quality_weight: Weight for quality in best model selection.
        batch_size: Batch size for embedding generation.
        device: Device for MLP training (cpu, cuda, mps).
        knn_k: Number of neighbors for KNN.
        kmeans_clusters: Number of clusters for KMeans.
        svm_kernel: SVM kernel type (rbf or linear).
        svm_gamma: SVM gamma parameter.
        mlp_hidden_sizes: Hidden layer sizes for MLP.
        mlp_epochs: Training epochs for MLP.
        mlp_learning_rate: Learning rate for MLP.
        mlp_dropout: Dropout rate for MLP.
        skip_mlp: Skip MLP training.
        algorithm: Which algorithm to train (all, knn, kmeans, svm, mlp).
        on_progress: Optional callback(percent, step, message) for progress.

    Returns:
        List of output model file paths.
    """
    if mlp_hidden_sizes is None:
        mlp_hidden_sizes = [256, 128]

    def progress(pct, step, msg):
        if on_progress:
            on_progress(pct, step, msg)
        print(f"[{pct}%] {step}: {msg}")

    start_time = time.time()

    # Step 1: Load data
    progress(10, "Loading data", "Loading benchmark data")
    if data_file:
        data_path = Path(data_file)
    else:
        data_path = download_data(cache_dir)

    records = load_jsonl(data_path)
    print_data_stats(records)

    # Step 2: Generate embeddings
    progress(
        20, "Generating embeddings", "Loading embedding model and generating embeddings"
    )
    queries = get_unique_queries(records)
    print(f"  {len(queries)} unique queries")

    cache_file = Path(cache_dir) / f"embeddings_{embedding_model}.npz"
    embeddings = generate_embeddings_for_queries(
        queries,
        model_name=embedding_model,
        batch_size=batch_size,
        cache_dir=cache_dir,
        cache_file=str(cache_file),
    )
    print(f"  Embedding dim: {len(next(iter(embeddings.values())))}")

    # Step 3: Create training samples
    progress(45, "Creating samples", "Building training samples from embeddings")
    samples = create_training_samples(records, embeddings, quality_weight)
    print(f"  Created {len(samples)} training samples")

    # Step 4: Train models
    progress(50, "Training", f"Training {algorithm} model(s)")
    output_path = Path(output_dir)
    train_models(
        samples,
        output_path,
        knn_k=knn_k,
        kmeans_clusters=kmeans_clusters,
        svm_kernel=svm_kernel,
        svm_gamma=svm_gamma,
        mlp_hidden_sizes=mlp_hidden_sizes,
        mlp_epochs=mlp_epochs,
        mlp_learning_rate=mlp_learning_rate,
        mlp_dropout=mlp_dropout,
        device=device,
        skip_mlp=skip_mlp,
        algorithm=algorithm,
    )

    elapsed = time.time() - start_time
    print(f"\n✅ Training complete in {elapsed:.1f}s")
    print(f"   Models saved to: {output_path.absolute()}")

    # Collect output files
    output_files = []
    for model_name in [
        "knn_model.json",
        "kmeans_model.json",
        "svm_model.json",
        "mlp_model.json",
    ]:
        model_file = output_path / model_name
        if model_file.exists():
            output_files.append(str(model_file))

    return output_files


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
        help="Device for embedding/MLP model (default: cpu)",
    )

    # MLP-specific arguments
    parser.add_argument(
        "--mlp-hidden-sizes",
        type=str,
        default="256,128",
        help="Comma-separated hidden layer sizes for MLP (default: 256,128)",
    )
    parser.add_argument(
        "--mlp-epochs",
        type=int,
        default=100,
        help="Training epochs for MLP (default: 100)",
    )
    parser.add_argument(
        "--mlp-learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for MLP (default: 0.001)",
    )
    parser.add_argument(
        "--mlp-dropout",
        type=float,
        default=0.1,
        help="Dropout rate for MLP (default: 0.1)",
    )
    parser.add_argument(
        "--skip-mlp",
        action="store_true",
        help="Skip MLP training (useful if PyTorch not installed)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="all",
        choices=["all", "knn", "kmeans", "svm", "mlp"],
        help="Train specific algorithm: all, knn, kmeans, svm, mlp (default: all)",
    )

    args = parser.parse_args()

    # Parse MLP hidden sizes
    mlp_hidden_sizes = [int(x.strip()) for x in args.mlp_hidden_sizes.split(",")]

    print("\n" + "=" * 60)
    print("  ML Model Selection Training")
    print("=" * 60)
    print(f"  Output dir:       {args.output_dir}")
    print(f"  Embedding model:  {args.embedding_model}")
    print(f"  Quality weight:   {args.quality_weight}")
    print(f"  KNN k:            {args.knn_k}")
    print(f"  KMeans clusters:  {args.kmeans_clusters}")
    print(f"  SVM kernel:       {args.svm_kernel} (gamma={args.svm_gamma})")
    print(f"  Algorithm:        {args.algorithm}")
    if args.algorithm in ["all", "mlp"]:
        if TORCH_AVAILABLE and not args.skip_mlp:
            print(f"  MLP hidden:       {mlp_hidden_sizes}")
            print(f"  MLP epochs:       {args.mlp_epochs}")
            print(f"  MLP device:       {args.device}")
        elif args.skip_mlp:
            print("  MLP:              Skipped (--skip-mlp)")
        else:
            print("  MLP:              Skipped (PyTorch not available)")
    print("=" * 60 + "\n")

    # Run the shared pipeline
    run_training_pipeline(
        data_file=args.data_file,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        cache_dir=args.cache_dir,
        quality_weight=args.quality_weight,
        batch_size=args.batch_size,
        device=args.device,
        knn_k=args.knn_k,
        kmeans_clusters=args.kmeans_clusters,
        svm_kernel=args.svm_kernel,
        svm_gamma=args.svm_gamma,
        mlp_hidden_sizes=mlp_hidden_sizes,
        mlp_epochs=args.mlp_epochs,
        mlp_learning_rate=args.mlp_learning_rate,
        mlp_dropout=args.mlp_dropout,
        skip_mlp=args.skip_mlp,
        algorithm=args.algorithm,
    )


if __name__ == "__main__":
    main()
