"""
Cache management for downloaded models.

Handles cache hit detection and path resolution.
"""

import logging
import os
from pathlib import Path

from .config import ModelSpec

logger = logging.getLogger(__name__)


def is_cached(spec: ModelSpec, cache_dir: str) -> bool:
    """
    Check if a model is already cached and valid.

    A model is considered cached if:
    1. The directory exists
    2. The .downloaded marker file exists
    3. At least one model file (config.json or weights) exists

    Args:
        spec: Model specification
        cache_dir: Base cache directory

    Returns:
        True if model is cached and appears valid
    """
    model_path = get_model_path(spec, cache_dir)

    if not os.path.isdir(model_path):
        logger.debug(f"Model '{spec.id}' not cached: directory doesn't exist")
        return False

    # Check for marker file
    marker_path = os.path.join(model_path, ".downloaded")
    if not os.path.exists(marker_path):
        logger.debug(f"Model '{spec.id}' not cached: missing .downloaded marker")
        return False

    # Check for at least one model file
    path = Path(model_path)
    has_model_files = (
        (path / "config.json").exists()
        or list(path.glob("*.safetensors"))
        or list(path.glob("*.bin"))
    )

    if not has_model_files:
        logger.debug(f"Model '{spec.id}' not cached: no model files found")
        return False

    logger.debug(f"Model '{spec.id}' is cached at {model_path}")
    return True


def get_model_path(spec: ModelSpec, cache_dir: str) -> str:
    """
    Get the local path for a model.

    Args:
        spec: Model specification
        cache_dir: Base cache directory

    Returns:
        Absolute path to the model directory
    """
    local_dir = spec.get_local_dir()
    return os.path.abspath(os.path.join(cache_dir, local_dir))


def clear_cache(spec: ModelSpec, cache_dir: str) -> bool:
    """
    Remove a cached model.

    Args:
        spec: Model specification
        cache_dir: Base cache directory

    Returns:
        True if cache was cleared, False if model wasn't cached
    """
    import shutil

    model_path = get_model_path(spec, cache_dir)

    if not os.path.exists(model_path):
        return False

    logger.info(f"Clearing cache for model '{spec.id}' at {model_path}")
    shutil.rmtree(model_path)
    return True


def get_cache_info(spec: ModelSpec, cache_dir: str) -> dict | None:
    """
    Get information about a cached model.

    Args:
        spec: Model specification
        cache_dir: Base cache directory

    Returns:
        Dictionary with cache info, or None if not cached
    """
    model_path = get_model_path(spec, cache_dir)

    if not is_cached(spec, cache_dir):
        return None

    path = Path(model_path)

    # Read download timestamp
    marker_path = path / ".downloaded"
    download_time = None
    if marker_path.exists():
        download_time = marker_path.read_text().strip()

    # Calculate total size
    total_size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

    # Count files
    file_count = sum(1 for f in path.rglob("*") if f.is_file())

    return {
        "path": model_path,
        "downloaded_at": download_time,
        "size_bytes": total_size,
        "file_count": file_count,
    }
