"""
Verifier module for model integrity checking.

Provides verification at different levels:
- "none": Skip verification
- "size": Check files exist and have non-zero size
- "sha256": Check file integrity by computing SHA256 hashes (deep read verify)
"""

import hashlib
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def verify_model(model_path: str, verify_level: str = "size") -> bool:
    """
    Verify model integrity at the specified level.

    Args:
        model_path: Path to the model directory
        verify_level: Verification level ("none", "size", "sha256")

    Returns:
        True if verification passes, False otherwise
    """
    if verify_level == "none":
        logger.debug(f"Skipping verification for {model_path}")
        return True

    if verify_level == "size":
        return verify_size(model_path)

    if verify_level == "sha256":
        return check_file_integrity(model_path)

    logger.warning(f"Unknown verify level '{verify_level}', defaulting to size check")
    return verify_size(model_path)


def verify_size(model_path: str) -> bool:
    """
    Verify that model files exist and have non-zero size.

    Args:
        model_path: Path to the model directory

    Returns:
        True if all files exist and have content
    """
    if not os.path.isdir(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        return False

    path = Path(model_path)

    # Check for essential model files
    has_config = (path / "config.json").exists()
    has_weights = bool(
        list(path.glob("*.safetensors"))
        or list(path.glob("*.bin"))
        or list(path.glob("*.pt"))
        or
        # LoRA adapters (specific extensions only)
        list(path.glob("adapter_model.safetensors"))
        or list(path.glob("adapter_model.bin"))
    )

    # Must have config.json or weight files
    if not has_config and not has_weights:
        logger.error(
            f"No model files found in {model_path} (missing config.json and weight files)"
        )
        return False

    # Count actual model files (excluding markers and hidden files)
    model_files = [
        f
        for f in path.rglob("*")
        if f.is_file()
        and not f.name.startswith(".")
        and f.suffix not in [".lock"]
        and ".cache" not in f.parts
    ]

    if len(model_files) == 0:
        logger.error(f"No model files found in {model_path}")
        return False

    # Check all files have non-zero size
    for file_path in path.rglob("*"):
        if file_path.is_file():
            if file_path.stat().st_size == 0:
                # Skip marker files and HuggingFace temp/cache files
                if file_path.name in [".downloaded", ".gitattributes", ".gitignore"]:
                    continue
                # Skip lock files created by huggingface_hub
                if file_path.suffix == ".lock":
                    continue
                # Skip files in .cache directory (HuggingFace download cache)
                if ".cache" in file_path.parts:
                    continue
                logger.error(f"Empty file detected: {file_path}")
                return False

    logger.debug(f"Size verification passed for {model_path}")
    return True


def check_file_integrity(model_path: str) -> bool:
    """
    Verify model files by computing SHA256 checksums.

    NOTE: This performs a "deep read" verification to ensure files are readable
    and not corrupt at the filesystem level. It does NOT currently verify
    against upstream HuggingFace hashes.

    Args:
        model_path: Path to the model directory

    Returns:
        True if verification passes
    """
    if not verify_size(model_path):
        return False

    path = Path(model_path)

    # Compute SHA256 for weight files
    weight_files = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))

    for weight_file in weight_files:
        try:
            file_hash = compute_sha256(weight_file)
            logger.debug(f"SHA256 for {weight_file.name}: {file_hash}")
        except Exception as e:
            logger.error(f"Failed to compute SHA256 for {weight_file}: {e}")
            return False

    logger.debug(f"Deep read integrity check passed for {model_path}")
    return True


def compute_sha256(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file.

    Args:
        file_path: Path to the file

    Returns:
        Hex-encoded SHA256 hash
    """
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()
