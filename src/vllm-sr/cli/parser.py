"""Configuration parser for vLLM Semantic Router."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from cli.compat_blocks import attach_typed_compat_blocks, extract_typed_compat_blocks
from cli.legacy_normalization import normalize_legacy_user_config
from cli.models import UserConfig
from cli.user_config_top_level import validate_user_config_top_level_keys
from cli.utils import getLogger

log = getLogger(__name__)


class ConfigParseError(Exception):
    """Configuration parsing error."""

    pass


def parse_user_config(config_path: str) -> UserConfig:
    """
    Parse and validate user configuration file.

    Args:
        config_path: Path to config.yaml

    Returns:
        UserConfig: Validated user configuration

    Raises:
        ConfigParseError: If configuration is invalid
    """
    config_file = Path(config_path)

    # Check if file exists
    if not config_file.exists():
        raise ConfigParseError(f"Configuration file not found: {config_path}")

    # Load YAML
    try:
        with open(config_file) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigParseError(f"Invalid YAML syntax: {e}") from e
    except Exception as e:
        raise ConfigParseError(f"Failed to read configuration file: {e}") from e

    if not data:
        raise ConfigParseError("Configuration file is empty")

    if not isinstance(data, dict):
        raise ConfigParseError("Configuration file must contain a top-level mapping")

    # Validate with Pydantic
    try:
        data = normalize_legacy_user_config(data)
        validate_user_config_top_level_keys(data)
        sanitized_data, compat_blocks = extract_typed_compat_blocks(data)
        config = UserConfig(**sanitized_data)
        attach_typed_compat_blocks(config, compat_blocks)
        log.info("Configuration parsed successfully")
        log.info(f"  Version: {config.version}")
        log.info(f"  Listeners: {len(config.listeners)}")
        log.info(f"  Decisions: {len(config.decisions)}")
        log.info(f"  Models: {len(config.providers.models)}")
        return config
    except ValidationError as e:
        # Format validation errors nicely
        errors = []
        for error in e.errors():
            loc = " -> ".join(str(x) for x in error["loc"])
            msg = error["msg"]
            errors.append(f"  • {loc}: {msg}")

        error_msg = "Configuration validation failed:\n" + "\n".join(errors)
        raise ConfigParseError(error_msg) from e
    except Exception as e:
        raise ConfigParseError(f"Unexpected error during validation: {e}") from e


def detect_config_format(data: dict[str, Any]) -> str:
    """
    Detect configuration format (new vs legacy).

    Args:
        data: Configuration data dictionary

    Returns:
        str: "new" or "legacy"
    """
    # New format has 'version' field starting with 'v'
    if (
        "version" in data
        and isinstance(data["version"], str)
        and data["version"].startswith("v")
    ):
        return "new"
    return "legacy"


def load_config_file(config_path: str) -> dict[str, Any]:
    """
    Load configuration file as dictionary.

    Args:
        config_path: Path to configuration file

    Returns:
        dict: Configuration data

    Raises:
        ConfigParseError: If file cannot be loaded
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise ConfigParseError(f"Configuration file not found: {config_path}")

    try:
        with open(config_file) as f:
            data = yaml.safe_load(f)
        return data or {}
    except yaml.YAMLError as e:
        raise ConfigParseError(f"Invalid YAML syntax: {e}") from e
    except Exception as e:
        raise ConfigParseError(f"Failed to read configuration file: {e}") from e


def validate_signal_uniqueness(config: UserConfig) -> list:
    """
    Validate that signal names are unique across all signal types.

    Args:
        config: User configuration

    Returns:
        list: List of validation errors (empty if valid)
    """
    errors = []
    seen = {}

    if not config.signals:
        return errors

    # Check keyword signals
    for signal in config.signals.keywords:
        if signal.name in seen:
            errors.append(
                f"Duplicate signal name '{signal.name}' in keywords "
                f"(already defined in {seen[signal.name]})"
            )
        seen[signal.name] = "keywords"

    # Check embedding signals
    for signal in config.signals.embeddings:
        if signal.name in seen:
            errors.append(
                f"Duplicate signal name '{signal.name}' in embeddings "
                f"(already defined in {seen[signal.name]})"
            )
        seen[signal.name] = "embeddings"

    # Check context signals
    if config.signals.context_rules:
        for signal in config.signals.context_rules:
            if signal.name in seen:
                errors.append(
                    f"Duplicate signal name '{signal.name}' in context rules "
                    f"(already defined in {seen[signal.name]})"
                )
            seen[signal.name] = "context_rules"

    return errors


def validate_domain_uniqueness(config: UserConfig) -> list:
    """
    Validate that domain names are unique.

    Args:
        config: User configuration

    Returns:
        list: List of validation errors (empty if valid)
    """
    errors = []

    if not config.signals or not config.signals.domains:
        return errors

    seen = set()
    for domain in config.signals.domains:
        if domain.name in seen:
            errors.append(f"Duplicate domain name '{domain.name}'")
        seen.add(domain.name)

    return errors


def validate_model_uniqueness(config: UserConfig) -> list:
    """
    Validate that model names are unique.

    Args:
        config: User configuration

    Returns:
        list: List of validation errors (empty if valid)
    """
    errors = []
    seen = set()

    for model in config.providers.models:
        if model.name in seen:
            errors.append(f"Duplicate model name '{model.name}'")
        seen.add(model.name)

    return errors
