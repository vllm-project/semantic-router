"""
Model Manager - Automated model download, verification, and caching.

This module provides utilities for managing ML models from HuggingFace,
including automatic download, integrity verification, and cache management.

Usage:
    from model_manager import ensure_models
    ensure_models("models.yaml")

    # Or programmatically:
    from model_manager import ModelManager
    manager = ModelManager.from_config("models.yaml")
    manager.ensure_all()
"""

from .config import ModelSpec, ModelsConfig
from .registry import load_models_config
from .downloader import download_model
from .verifier import verify_model
from .cache import is_cached, get_model_path
from .errors import (
    ModelManagerError,
    MissingModelError,
    BadChecksumError,
    DownloadError,
)

__version__ = "0.1.0"
__all__ = [
    "ensure_models",
    "ModelManager",
    "ModelSpec",
    "ModelsConfig",
    "load_models_config",
    "download_model",
    "verify_model",
    "is_cached",
    "get_model_path",
    "ModelManagerError",
    "MissingModelError",
    "BadChecksumError",
    "DownloadError",
]


def ensure_models(
    config_path: str = "models.yaml", cache_dir: str | None = None
) -> None:
    """
    Main entry point. Reads config, downloads missing models, verifies integrity.
    Called during application startup.

    Args:
        config_path: Path to models.yaml configuration file
        cache_dir: Override cache directory from config
    """
    manager = ModelManager.from_config(config_path)
    if cache_dir:
        manager.config.cache_dir = cache_dir
    manager.ensure_all()


class ModelManager:
    """
    Central manager for model download, verification, and caching.
    """

    def __init__(self, config: ModelsConfig):
        self.config = config

    @classmethod
    def from_config(cls, config_path: str) -> "ModelManager":
        """Create a ModelManager from a configuration file."""
        config = load_models_config(config_path)
        return cls(config)

    def ensure_all(self) -> dict[str, str]:
        """
        Ensure all models are downloaded and verified.

        Returns:
            Dictionary mapping model IDs to their local paths
        """
        import logging

        logger = logging.getLogger(__name__)

        results = {}
        for spec in self.config.models:
            model_path = get_model_path(spec, self.config.cache_dir)

            if is_cached(spec, self.config.cache_dir):
                logger.info(f"Model '{spec.id}' already cached at {model_path}")
                results[spec.id] = model_path
                continue

            logger.info(f"Downloading model '{spec.id}' from {spec.repo_id}...")
            local_path = download_model(spec, self.config.cache_dir)

            if self.config.verify != "none":
                logger.info(f"Verifying model '{spec.id}'...")
                if not verify_model(local_path, self.config.verify):
                    raise BadChecksumError(f"Verification failed for model '{spec.id}'")

            results[spec.id] = local_path
            logger.info(f"Model '{spec.id}' ready at {local_path}")

        return results

    def ensure_model(self, model_id: str) -> str:
        """
        Ensure a specific model is downloaded and verified.

        Args:
            model_id: ID of the model to ensure

        Returns:
            Local path to the model
        """
        spec = self.get_model_spec(model_id)
        if spec is None:
            raise MissingModelError(f"Model '{model_id}' not found in configuration")

        model_path = get_model_path(spec, self.config.cache_dir)

        if is_cached(spec, self.config.cache_dir):
            return model_path

        local_path = download_model(spec, self.config.cache_dir)

        if self.config.verify != "none":
            if not verify_model(local_path, self.config.verify):
                raise BadChecksumError(f"Verification failed for model '{model_id}'")

        return local_path

    def get_model_spec(self, model_id: str) -> ModelSpec | None:
        """Get model specification by ID."""
        for spec in self.config.models:
            if spec.id == model_id:
                return spec
        return None
