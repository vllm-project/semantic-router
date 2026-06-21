"""Configuration parser for vLLM Semantic Router."""

import yaml
from pathlib import Path
from typing import Dict, Any
from pydantic import ValidationError

from cli.config_contract import (
    LEGACY_PROVIDER_DEFAULT_KEYS,
    LEGACY_PROVIDER_MODEL_SURFACE_KEYS,
    LEGACY_SIGNAL_KEY_TO_CANONICAL,
    iter_named_signal_entries,
)
from cli.models import UserConfig
from cli.utils import get_logger

log = get_logger(__name__)


class ConfigParseError(Exception):
    """Configuration parsing error."""

    pass


def _deprecated_config_fields(data: Dict[str, Any]) -> list[str]:
    fields: list[str] = []

    for field_name in ("signals", "decisions", *LEGACY_SIGNAL_KEY_TO_CANONICAL):
        if field_name in data:
            fields.append(field_name)

    routing = data.get("routing")
    if isinstance(routing, dict) and "models" in routing:
        fields.append("routing.models")

    providers = data.get("providers")
    if isinstance(providers, dict):
        for field_name in (
            "model_targets",
            "backends",
            "auth_profiles",
            *LEGACY_PROVIDER_DEFAULT_KEYS,
        ):
            if field_name in providers:
                fields.append(f"providers.{field_name}")

        models = providers.get("models")
        if isinstance(models, list):
            for index, model in enumerate(models):
                if not isinstance(model, dict):
                    continue
                if "access" in model:
                    fields.append(f"providers.models[{index}].access")
                for field_name in LEGACY_PROVIDER_MODEL_SURFACE_KEYS:
                    if field_name in model:
                        fields.append(f"providers.models[{index}].{field_name}")

    global_config = data.get("global")
    if isinstance(global_config, dict) and "modules" in global_config:
        fields.append("global.modules")

    return fields


def _removed_router_learning_fields(data: Dict[str, Any]) -> list[str]:
    fields: list[str] = []

    global_config = data.get("global")
    if isinstance(global_config, dict):
        router = global_config.get("router")
        if isinstance(router, dict):
            model_selection = router.get("model_selection")
            if isinstance(model_selection, dict):
                for field_name in (
                    "session_aware",
                    "model_switch_gate",
                    "lookup_tables",
                    "elo",
                ):
                    if field_name in model_selection:
                        fields.append(f"global.router.model_selection.{field_name}")
                if str(model_selection.get("method", "")).strip().lower() == "elo":
                    fields.append("global.router.model_selection.method=elo")

    routing = data.get("routing")
    if isinstance(routing, dict):
        decisions = routing.get("decisions")
        if isinstance(decisions, list):
            for index, decision in enumerate(decisions):
                if not isinstance(decision, dict):
                    continue
                algorithm = decision.get("algorithm")
                if not isinstance(algorithm, dict):
                    continue
                algorithm_type = str(algorithm.get("type", "")).strip().lower()
                if algorithm_type == "session_aware":
                    fields.append(
                        f"routing.decisions[{index}].algorithm.type=session_aware"
                    )
                if algorithm_type in {"elo", "rl_driven", "gmtrouter"}:
                    fields.append(
                        f"routing.decisions[{index}].algorithm.type={algorithm_type}"
                    )
                if "session_aware" in algorithm:
                    fields.append(f"routing.decisions[{index}].algorithm.session_aware")
                for field_name in ("elo", "rl_driven", "gmtrouter"):
                    if field_name in algorithm:
                        fields.append(
                            f"routing.decisions[{index}].algorithm.{field_name}"
                        )

    return fields


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
        with open(config_file, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigParseError(f"Invalid YAML syntax: {e}")
    except Exception as e:
        raise ConfigParseError(f"Failed to read configuration file: {e}")

    if not data:
        raise ConfigParseError("Configuration file is empty")

    deprecated_fields = _deprecated_config_fields(data)
    if deprecated_fields:
        joined_fields = ", ".join(deprecated_fields)
        raise ConfigParseError(
            "Deprecated config fields are no longer supported: "
            f"{joined_fields}. Use `vllm-sr config migrate --config {config_path}` "
            "or rewrite the file to canonical v0.3 `providers/routing/global`."
        )

    removed_router_learning_fields = _removed_router_learning_fields(data)
    if removed_router_learning_fields:
        joined_fields = ", ".join(removed_router_learning_fields)
        raise ConfigParseError(
            "Removed Router Learning config fields are no longer supported: "
            f"{joined_fields}. Enable the matching `global.router.learning.adaptations.<name>` "
            "block (`session_aware`, `elo`, `bandit`, or `personalization`) "
            "such as `global.router.learning.adaptations.session_aware`; "
            "and use `routing.decisions[].adaptations.<name>` only when a "
            "decision needs apply/observe/bypass control or sparse overrides."
        )

    # Validate with Pydantic
    try:
        config = UserConfig(**data)
        log.info(f"Configuration parsed successfully")
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
        raise ConfigParseError(error_msg)
    except Exception as e:
        raise ConfigParseError(f"Unexpected error during validation: {e}")


def detect_config_format(data: Dict[str, Any]) -> str:
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


def load_config_file(config_path: str) -> Dict[str, Any]:
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
        with open(config_file, "r") as f:
            data = yaml.safe_load(f)
        return data or {}
    except yaml.YAMLError as e:
        raise ConfigParseError(f"Invalid YAML syntax: {e}")
    except Exception as e:
        raise ConfigParseError(f"Failed to read configuration file: {e}")


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

    for family, signal_name in iter_named_signal_entries(config.signals):
        if signal_name in seen:
            errors.append(
                f"Duplicate signal name '{signal_name}' in {family} "
                f"(already defined in {seen[signal_name]})"
            )
        seen[signal_name] = family

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
