#!/usr/bin/env python3
"""
Upload trained models or training data to HuggingFace Hub.

Usage:
    # Upload trained models
    python upload_model.py --model-dir models/ --repo-id abdallah1008/semantic-router-ml-models

    # Upload training data
    python upload_model.py --training-data --data-file data/benchmark_training_data.jsonl --repo-id abdallah1008/ml-selection-benchmark-data
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, upload_file, upload_folder


def upload_training_data(
    data_file: str,
    repo_id: str,
    private: bool = False,
    token: str = None,
) -> None:
    """
    Upload training data to HuggingFace datasets.

    Args:
        data_file: Path to training data file (e.g., benchmark_training_data.jsonl)
        repo_id: HuggingFace dataset repository ID
        private: Whether to create a private repository
        token: HuggingFace token (optional, uses HF_TOKEN env var if not provided)
    """
    data_path = Path(data_file)

    if not data_path.exists():
        raise ValueError(f"Training data file not found: {data_file}")

    print(f"Uploading training data from {data_file} to {repo_id}")

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=token,
        )
        print(f"✓ Dataset repository {repo_id} ready")
    except Exception as e:
        print(f"⚠ Warning creating repo: {e}")

    # Upload file
    upload_file(
        path_or_fileobj=str(data_path),
        path_in_repo=data_path.name,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message="Upload ML model selection training data",
    )

    print(f"✓ Training data uploaded to https://huggingface.co/datasets/{repo_id}")


def upload_models(
    model_dir: str,
    repo_id: str,
    private: bool = False,
    token: str = None,
) -> None:
    """
    Upload trained models to HuggingFace Hub.

    Args:
        model_dir: Directory containing trained models
        repo_id: HuggingFace repository ID (e.g., "abdallah1008/semantic-router-ml-models")
        private: Whether to create a private repository
        token: HuggingFace token (optional, uses HF_TOKEN env var if not provided)
    """
    model_path = Path(model_dir)

    if not model_path.exists():
        raise ValueError(f"Model directory not found: {model_dir}")

    # Check for expected files
    expected_files = ["knn_model.json", "kmeans_model.json", "svm_model.json"]
    missing = [f for f in expected_files if not (model_path / f).exists()]
    if missing:
        print(f"⚠ Warning: Missing model files: {missing}")

    print(f"Uploading models from {model_dir} to {repo_id}")

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
            token=token,
        )
        print(f"✓ Repository {repo_id} ready")
    except Exception as e:
        print(f"⚠ Warning creating repo: {e}")

    # Upload folder
    upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message="Upload ML model selection models",
    )

    print(f"✓ Models uploaded to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload trained models or training data to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload trained models
  python upload_model.py --model-dir models/ --repo-id abdallah1008/semantic-router-ml-models

  # Upload training data
  python upload_model.py --training-data --data-file data/benchmark_training_data.jsonl --repo-id abdallah1008/ml-selection-benchmark-data
        """,
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--training-data",
        action="store_true",
        help="Upload training data instead of trained models",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        help="Path to training data file (required with --training-data)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="HuggingFace repository ID",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace token (uses HF_TOKEN env var if not provided)",
    )

    args = parser.parse_args()

    # Use environment variable if token not provided
    token = args.token or os.environ.get("HF_TOKEN")

    if args.training_data:
        # Upload training data
        if not args.data_file:
            parser.error("--data-file is required when using --training-data")
        repo_id = args.repo_id or "abdallah1008/ml-selection-benchmark-data"
        upload_training_data(
            data_file=args.data_file,
            repo_id=repo_id,
            private=args.private,
            token=token,
        )
    else:
        # Upload trained models
        if not args.model_dir:
            parser.error("--model-dir is required when uploading models")
        repo_id = args.repo_id or "abdallah1008/semantic-router-ml-models"
        upload_models(
            model_dir=args.model_dir,
            repo_id=repo_id,
            private=args.private,
            token=token,
        )


if __name__ == "__main__":
    main()
