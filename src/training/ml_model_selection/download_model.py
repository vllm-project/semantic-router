#!/usr/bin/env python3
"""
Download trained models or training data from HuggingFace Hub.

Usage:
    # Download trained models (if available)
    python download_model.py --output-dir models/

    # Download training data (for local training)
    python download_model.py --training-data --output-dir data/

    # Download from custom repository
    python download_model.py --repo-id my-org/my-models --output-dir models/
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


# Default HuggingFace repositories
DEFAULT_MODELS_REPO = "abdallah1008/semantic-router-ml-models"
DEFAULT_DATA_REPO = "abdallah1008/ml-selection-benchmark-data"

# Expected model files
MODEL_FILES = ["knn_model.json", "kmeans_model.json", "svm_model.json"]

# Training data file
TRAINING_DATA_FILE = "benchmark_training_data.jsonl"


def download_training_data(
    output_dir: str,
    repo_id: str = DEFAULT_DATA_REPO,
    token: str = None,
) -> str:
    """
    Download training data from HuggingFace datasets.

    Args:
        output_dir: Directory to save downloaded data
        repo_id: HuggingFace dataset repository ID
        token: HuggingFace token (optional, for private repos)

    Returns:
        Path to the downloaded training data file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading training data from {repo_id}...")

    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=TRAINING_DATA_FILE,
            repo_type="dataset",
            local_dir=str(output_path),
            token=token,
        )
        print(f"✓ Training data downloaded to {local_path}")
        return local_path

    except Exception as e:
        print(f"✗ Download failed: {e}")
        print("\nTo download training data manually:")
        print(f"  1. Go to https://huggingface.co/datasets/{repo_id}")
        print(f"  2. Download {TRAINING_DATA_FILE}")
        print(f"  3. Place it in {output_dir}/")
        raise


def download_models(
    output_dir: str,
    repo_id: str = DEFAULT_MODELS_REPO,
    token: str = None,
) -> None:
    """
    Download trained models from HuggingFace Hub.

    Args:
        output_dir: Directory to save downloaded models
        repo_id: HuggingFace repository ID
        token: HuggingFace token (optional, for private repos)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading models from {repo_id}...")

    try:
        # Download all files from the repository
        # Note: .gitattributes is auto-created by HuggingFace for Git LFS - it's harmless
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(output_path),
            token=token,
        )
        print(f"✓ Models downloaded to {local_dir}")

        # Verify expected files exist
        missing = []
        for f in MODEL_FILES:
            if not (output_path / f).exists():
                missing.append(f)

        if missing:
            print(f"⚠ Warning: Missing expected files: {missing}")
        else:
            print(f"✓ All model files present: {MODEL_FILES}")

    except Exception as e:
        print(f"✗ Download failed: {e}")
        print("\nTo download models manually:")
        print(f"  1. Go to https://huggingface.co/{repo_id}")
        print(f"  2. Download the model files")
        print(f"  3. Place them in {output_dir}/")
        raise


def download_single_model(
    output_dir: str,
    model_file: str,
    repo_id: str = DEFAULT_MODELS_REPO,
    token: str = None,
) -> str:
    """
    Download a single model file from HuggingFace Hub.

    Args:
        output_dir: Directory to save downloaded model
        model_file: Name of the model file (e.g., "knn_model.json")
        repo_id: HuggingFace repository ID
        token: HuggingFace token (optional)

    Returns:
        Path to the downloaded file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_file} from {repo_id}...")

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=model_file,
        repo_type="model",
        local_dir=str(output_path),
        token=token,
    )

    print(f"✓ Downloaded to {local_path}")
    return local_path


def main():
    parser = argparse.ArgumentParser(
        description="Download trained models or training data from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download trained models (if available)
  python download_model.py --output-dir models/

  # Download training data (for local training)
  python download_model.py --training-data --output-dir data/

  # Download from custom repository
  python download_model.py --repo-id my-org/my-models --output-dir models/

  # Download a single model file
  python download_model.py --output-dir models/ --file knn_model.json
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save downloaded files (default: models/)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help=f"HuggingFace repository ID (default: {DEFAULT_MODELS_REPO} for models, {DEFAULT_DATA_REPO} for data)",
    )
    parser.add_argument(
        "--training-data",
        action="store_true",
        help="Download training data instead of trained models",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Download a single file instead of all models",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace token (uses HF_TOKEN env var if not provided)",
    )

    args = parser.parse_args()

    # Use environment variable if token not provided
    # Handle empty string case: both None and "" should result in None
    # In GitHub Actions, secrets that don't exist evaluate to "" when used
    token = args.token if args.token else os.environ.get("HF_TOKEN")
    # Ensure empty or whitespace-only tokens are treated as None
    # This prevents "Illegal header value b'Bearer '" errors
    if not token or not token.strip():
        token = None

    if args.training_data:
        # Download training data from datasets repo
        repo_id = args.repo_id or DEFAULT_DATA_REPO
        download_training_data(
            output_dir=args.output_dir,
            repo_id=repo_id,
            token=token,
        )
    elif args.file:
        repo_id = args.repo_id or DEFAULT_MODELS_REPO
        download_single_model(
            output_dir=args.output_dir,
            model_file=args.file,
            repo_id=repo_id,
            token=token,
        )
    else:
        repo_id = args.repo_id or DEFAULT_MODELS_REPO
        download_models(
            output_dir=args.output_dir,
            repo_id=repo_id,
            token=token,
        )


if __name__ == "__main__":
    main()
