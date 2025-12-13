"""
Registry module for parsing models.yaml configuration.
"""

import os
from typing import Any

import yaml

from .config import ModelSpec, ModelsConfig
from .errors import ConfigurationError


def load_models_config(config_path: str) -> ModelsConfig:
    """
    Load and validate models configuration from a YAML file.

    Args:
        config_path: Path to the models.yaml configuration file

    Returns:
        Parsed ModelsConfig object

    Raises:
        ConfigurationError: If config file is missing or invalid
    """
    if not os.path.exists(config_path):
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {e}")

    if data is None:
        raise ConfigurationError(f"Empty configuration file: {config_path}")

    return parse_config(data)


def parse_config(data: dict[str, Any]) -> ModelsConfig:
    """
    Parse configuration dictionary into ModelsConfig object.

    Args:
        data: Raw configuration dictionary from YAML

    Returns:
        Parsed ModelsConfig object
    """
    models = []
    for model_data in data.get("models", []):
        spec = parse_model_spec(model_data)
        models.append(spec)

    return ModelsConfig(
        models=models,
        cache_dir=data.get("cache_dir", "models"),
        verify=data.get("verify", "size"),
    )


def parse_model_spec(data: dict[str, Any]) -> ModelSpec:
    """
    Parse a single model specification from configuration.

    Args:
        data: Model configuration dictionary

    Returns:
        Parsed ModelSpec object

    Raises:
        ConfigurationError: If required fields are missing
    """
    if "id" not in data:
        raise ConfigurationError("Model spec missing required 'id' field")
    if "repo_id" not in data:
        raise ConfigurationError(
            f"Model '{data['id']}' missing required 'repo_id' field"
        )

    return ModelSpec(
        id=data["id"],
        repo_id=data["repo_id"],
        revision=data.get("revision", "main"),
        local_dir=data.get("local_dir"),
        files=data.get("files"),
    )
