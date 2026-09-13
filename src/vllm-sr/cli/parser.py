"""Configuration parser for vLLM Semantic Router."""

import yaml
from pathlib import Path
from typing import Dict, Any
from pydantic import ValidationError

from cli.bootstrap import SETUP_MODE_KEY
from cli.config_contract import (
    LEGACY_PROVIDER_DEFAULT_KEYS,
    LEGACY_PROVIDER_MODEL_SURFACE_KEYS,
    LEGACY_SIGNAL_KEY_TO_CANONICAL,
    iter_routing_profiles,
)
from cli.config_schema.validation import validate_config_structure
from cli.config_yaml import safe_load_router_config
from cli.context_bands import references_environment
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
                    "rl_driven",
                    "gmtrouter",
                    "bandit",
                    "personalization",
                ):
                    if field_name in model_selection:
                        fields.append(f"global.router.model_selection.{field_name}")
                method = str(model_selection.get("method", "")).strip().lower()
                if method in {
                    "session_aware",
                    "lookup_tables",
                    "elo",
                    "rl_driven",
                    "gmtrouter",
                    "bandit",
                    "personalization",
                }:
                    fields.append(f"global.router.model_selection.method={method}")

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
                if algorithm_type in {
                    "elo",
                    "rl_driven",
                    "gmtrouter",
                    "bandit",
                    "personalization",
                }:
                    fields.append(
                        f"routing.decisions[{index}].algorithm.type={algorithm_type}"
                    )
                if "session_aware" in algorithm:
                    fields.append(f"routing.decisions[{index}].algorithm.session_aware")
                for field_name in (
                    "elo",
                    "rl_driven",
                    "gmtrouter",
                    "bandit",
                    "personalization",
                ):
                    if field_name in algorithm:
                        fields.append(
                            f"routing.decisions[{index}].algorithm.{field_name}"
                        )

    return fields


def _reject_invalid_config_surfaces(data: Dict[str, Any], config_path: str) -> None:
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
            f"{joined_fields}. Use `global.router.learning.adaptation` for online "
            "model-choice learning, `global.router.learning.protection` for "
            "session or conversation protection, and `routing.decisions[].adaptations` "
            "only when a decision needs apply/observe/bypass control or a local "
            "adaptation candidate_set override."
        )


def _deferred_context_limits(config: UserConfig) -> list[tuple[str, str]]:
    """Return (path, value) for every context band limit that references the
    environment, which the Router expands before parsing."""
    deferred: list[tuple[str, str]] = []
    for profile_name, profile in iter_routing_profiles(config):
        prefix = (
            "routing"
            if profile_name == "default"
            else f"recipes[{profile_name}].routing"
        )
        for rule in profile.signals.context or []:
            for field_name in ("min_tokens", "max_tokens"):
                value = getattr(rule, field_name)
                if references_environment(value):
                    deferred.append(
                        (f"{prefix}.signals.context[{rule.name}].{field_name}", value)
                    )
    return deferred


def _warn_deferred_context_limits(config: UserConfig) -> None:
    for path, value in _deferred_context_limits(config):
        log.warning(
            f"{path} references the environment: {value}. The Router resolves it "
            "when the config loads, so this band was not checked"
        )


def parse_user_config(config_path: str, *, log_summary: bool = True) -> UserConfig:
    """
    Parse and validate user configuration file.

    Args:
        config_path: Path to config.yaml
        log_summary: Emit the human-readable parse summary. Machine-readable
            callers disable this so stdout remains a valid document.

    Returns:
        UserConfig: Validated user configuration

    Raises:
        ConfigParseError: If configuration is invalid
    """
    config_file = Path(config_path)

    # Check if file exists
    if not config_file.exists():
        raise ConfigParseError(f"Configuration file not found: {config_path}")

    # Load YAML. Context band token counts keep their source spelling so the
    # checks below see the text the Router parses from the forwarded file.
    try:
        with open(config_file, "r") as f:
            data = safe_load_router_config(f)
    except yaml.YAMLError as e:
        raise ConfigParseError(f"Invalid YAML syntax: {e}")
    except Exception as e:
        raise ConfigParseError(f"Failed to read configuration file: {e}")
    if not data:
        raise ConfigParseError("Configuration file is empty")

    _reject_invalid_config_surfaces(data, config_path)

    # Setup is CLI/Dashboard lifecycle metadata, already represented by
    # UserConfig. Validate the Router projection without widening its schema;
    # retain the metadata for the operational projection below.
    router_document = {
        key: value for key, value in data.items() if key != SETUP_MODE_KEY
    }
    structural_errors = validate_config_structure(router_document)
    if structural_errors:
        rendered = "\n".join(f"  • {error}" for error in structural_errors)
        raise ConfigParseError(f"Configuration schema validation failed:\n{rendered}")

    # Validate with Pydantic
    try:
        # JSON Schema owns accepted fields. Pydantic remains an operational
        # projection for CLI code and is intentionally forward-compatible with
        # new schema fields it does not need to interpret yet.
        config = UserConfig.model_validate(data, extra="allow")
        _warn_deferred_context_limits(config)
        if log_summary:
            log.info("Configuration parsed successfully")
            log.info(f"  Version: {config.version}")
            log.info(f"  Listeners: {len(config.listeners)}")
            recipe_decisions = sum(
                len(profile.decisions)
                for name, profile in iter_routing_profiles(config)
                if name != "default"
            )
            log.info(f"  Entrypoints: {len(config.entrypoints)}")
            log.info(f"  Recipes: {len(config.recipes)}")
            log.info(
                f"  Decisions: {len(config.decisions) + recipe_decisions} total "
                f"({len(config.decisions)} default, {recipe_decisions} recipe-owned)"
            )
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
            data = safe_load_router_config(f)
        return data or {}
    except yaml.YAMLError as e:
        raise ConfigParseError(f"Invalid YAML syntax: {e}")
    except Exception as e:
        raise ConfigParseError(f"Failed to read configuration file: {e}")
