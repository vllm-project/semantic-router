"""Configuration parser for vLLM Semantic Router."""

import yaml
from pathlib import Path
from typing import Dict, Any
from pydantic import ValidationError

from cli.bootstrap import SETUP_MODE_KEY, validate_setup_metadata
from cli.config_contract import iter_routing_profiles
from cli.config_algorithm_forms import removed_algorithm_forms
from cli.config_application_forms import removed_application_forms
from cli.config_heuristic_forms import removed_heuristic_forms
from cli.config_legacy_layout_forms import (
    removed_legacy_router_forms,
    removed_previous_release_forms,
)
from cli.config_model_resource_forms import removed_model_resource_forms
from cli.config_model_runtime_forms import removed_model_runtime_forms
from cli.config_plugin_forms import removed_plugin_forms
from cli.config_renamed_forms import read_renamed_forms
from cli.config_context_forms import removed_context_forms
from cli.config_embedding_forms import removed_embedding_forms
from cli.config_multimodal_forms import removed_multimodal_forms
from cli.config_reuse_forms import removed_reuse_forms
from cli.config_recipe_scopes import recipe_scopes
from cli.config_structure_forms import removed_structure_forms
from cli.config_tools_forms import removed_tools_forms
from cli.config_schema.validation import validate_config_structure
from cli.config_yaml import safe_load_router_config
from cli.context_bands import references_environment
from cli.models import REMOVED_BACKEND_REF_FIELDS, UserConfig
from cli.utils import get_logger

log = get_logger(__name__)


class ConfigParseError(Exception):
    """Configuration parsing error."""

    pass


def _removed_backend_ref_fields(data: Dict[str, Any]) -> list[tuple[str, str]]:
    """Backend refs that still set a removed gateway opt-in, as (path, field)."""

    found: list[tuple[str, str]] = []
    providers = data.get("providers")
    models = providers.get("models") if isinstance(providers, dict) else None
    if not isinstance(models, list):
        return found
    for index, model in enumerate(models):
        if not isinstance(model, dict):
            continue
        name = model.get("name")
        label = name if isinstance(name, str) and name.strip() else str(index)
        refs = model.get("backend_refs")
        if not isinstance(refs, list):
            continue
        for ref_index, ref in enumerate(refs):
            if not isinstance(ref, dict):
                continue
            for field in REMOVED_BACKEND_REF_FIELDS:
                if field in ref:
                    found.append(
                        (f"providers.models[{label}].backend_refs[{ref_index}]", field)
                    )
    return found


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

    for prefix, recipe in recipe_scopes(data):
        decisions = recipe.get("decisions")
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
                        f"{prefix}.decisions[{index}].algorithm.type=session_aware"
                    )
                if algorithm_type in {
                    "elo",
                    "rl_driven",
                    "gmtrouter",
                    "bandit",
                    "personalization",
                }:
                    fields.append(
                        f"{prefix}.decisions[{index}].algorithm.type={algorithm_type}"
                    )
                if "session_aware" in algorithm:
                    fields.append(
                        f"{prefix}.decisions[{index}].algorithm.session_aware"
                    )
                for field_name in (
                    "elo",
                    "rl_driven",
                    "gmtrouter",
                    "bandit",
                    "personalization",
                ):
                    if field_name in algorithm:
                        fields.append(
                            f"{prefix}.decisions[{index}].algorithm.{field_name}"
                        )

    return fields


def _reject_invalid_config_surfaces(data: Dict[str, Any]) -> None:
    structure_forms = removed_previous_release_forms(data) + removed_structure_forms(
        data
    )
    if structure_forms:
        raise ConfigParseError("\n".join(structure_forms))

    removed_backend_ref_fields = _removed_backend_ref_fields(data)
    if removed_backend_ref_fields:
        path, field = removed_backend_ref_fields[0]
        raise ConfigParseError(
            f"{path}: {field} was removed; gateway response fields are accepted "
            "by default, so delete it"
        )

    removed_router_learning_fields = _removed_router_learning_fields(data)
    if removed_router_learning_fields:
        joined_fields = ", ".join(removed_router_learning_fields)
        raise ConfigParseError(
            "Removed Router Learning config fields are no longer supported: "
            f"{joined_fields}. Use `global.router.learning.adaptation` for online "
            "model-choice learning, `global.router.learning.protection` for "
            "session or conversation protection, and `recipes[].decisions[].adaptations` "
            "only when a decision needs apply/observe/bypass control or a local "
            "adaptation candidate_set override."
        )

    removed_forms = (
        removed_legacy_router_forms(data)
        + removed_plugin_forms(data)
        + removed_application_forms(data)
        + removed_heuristic_forms(data)
        + removed_algorithm_forms(data)
        + removed_tools_forms(data)
        + removed_context_forms(data)
        + removed_reuse_forms(data)
        + removed_embedding_forms(data)
        + removed_multimodal_forms(data)
        + removed_model_runtime_forms(data)
        + removed_model_resource_forms(data)
    )
    if removed_forms:
        raise ConfigParseError("\n".join(removed_forms))


def _deferred_context_limits(config: UserConfig) -> list[tuple[str, str]]:
    """Return (path, value) for every context band limit that references the
    environment, which the Router expands before parsing."""
    deferred: list[tuple[str, str]] = []
    for profile_name, profile in iter_routing_profiles(config):
        prefix = f"recipes[{profile_name}]"
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
        with open(config_file, "r", encoding="utf-8") as f:
            data = safe_load_router_config(f)
    except yaml.YAMLError as e:
        raise ConfigParseError(f"Invalid YAML syntax: {e}")
    except Exception as e:
        raise ConfigParseError(f"Failed to read configuration file: {e}")
    if not data:
        raise ConfigParseError("Configuration file is empty")

    return parse_user_config_data(data, log_summary=log_summary)


def parse_user_config_data(
    data: Dict[str, Any], *, log_summary: bool = True
) -> UserConfig:
    """Apply the parse_user_config checks to an already loaded document."""
    data = read_renamed_forms(data)
    _reject_invalid_config_surfaces(data)

    # Setup is product control-plane metadata, not part of the canonical Router
    # schema. Validate the envelope explicitly, without stripping it from the
    # document that local orchestration and Dashboard onboarding still need.
    structural_errors = validate_setup_metadata(data.get(SETUP_MODE_KEY))
    router_payload = {
        name: value for name, value in data.items() if name != SETUP_MODE_KEY
    }
    structural_errors.extend(validate_config_structure(router_payload))
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
            decisions = sum(
                len(profile.decisions) for _, profile in iter_routing_profiles(config)
            )
            log.info(f"  Entrypoints: {len(config.entrypoints)}")
            log.info(f"  Recipes: {len(config.recipes)}")
            log.info(f"  Decisions: {decisions}")
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
        with open(config_file, "r", encoding="utf-8") as f:
            data = safe_load_router_config(f)
        return data or {}
    except yaml.YAMLError as e:
        raise ConfigParseError(f"Invalid YAML syntax: {e}")
    except Exception as e:
        raise ConfigParseError(f"Failed to read configuration file: {e}")
