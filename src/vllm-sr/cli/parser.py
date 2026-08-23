"""Configuration parser for vLLM Semantic Router."""

import yaml
from pathlib import Path
from typing import Dict, Any
from pydantic import ValidationError

from cli.config_contract import iter_named_signal_entries, iter_routing_profiles
from cli.models import RecipeDistribution, RouterLearningConfig, UserConfig
from cli.utils import get_logger

log = get_logger(__name__)


class ConfigParseError(Exception):
    """Configuration parsing error."""

    pass


def _removed_config_fields(data: Dict[str, Any]) -> list[str]:
    fields: list[str] = []

    global_config = data.get("global")
    if isinstance(global_config, dict) and "modules" in global_config:
        fields.append("global.modules")
    if isinstance(global_config, dict):
        router = global_config.get("router")
        if isinstance(router, dict):
            for field_name in (
                "auto_model_name",
                "auto_model_names",
                "include_config_models_in_list",
                "strategy",
                "model_selection",
            ):
                if field_name in router:
                    fields.append(f"global.router.{field_name}")
        integrations = global_config.get("integrations")
        looper = integrations.get("looper") if isinstance(integrations, dict) else None
        if isinstance(looper, dict):
            for family_name in ("remom", "fusion", "flow"):
                family = looper.get(family_name)
                if isinstance(family, dict) and "model_names" in family:
                    fields.append(
                        f"global.integrations.looper.{family_name}.model_names"
                    )

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

    for recipe_index, recipe in enumerate(data.get("recipes") or []):
        document = recipe.get("document") if isinstance(recipe, dict) else None
        if isinstance(document, dict):
            decisions = document.get("decisions")
        else:
            decisions = None
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
                        f"recipes[{recipe_index}].document.decisions[{index}].algorithm.type=session_aware"
                    )
                if algorithm_type in {
                    "elo",
                    "rl_driven",
                    "gmtrouter",
                    "bandit",
                    "personalization",
                }:
                    fields.append(
                        f"recipes[{recipe_index}].document.decisions[{index}].algorithm.type={algorithm_type}"
                    )
                if "session_aware" in algorithm:
                    fields.append(
                        f"recipes[{recipe_index}].document.decisions[{index}].algorithm.session_aware"
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
                            f"recipes[{recipe_index}].document.decisions[{index}].algorithm.{field_name}"
                        )

    return fields


def _unknown_fields(
    value: Any,
    allowed: set[str],
    prefix: str,
) -> list[str]:
    if not isinstance(value, dict):
        return []
    return [
        f"{prefix}.{field_name}" for field_name in value if field_name not in allowed
    ]


def _unsupported_router_learning_fields(data: Dict[str, Any]) -> list[str]:
    fields: list[str] = []

    global_config = data.get("global")
    if not isinstance(global_config, dict):
        return fields
    router = global_config.get("router")
    if not isinstance(router, dict):
        return fields
    learning = router.get("learning")
    if not isinstance(learning, dict):
        return fields

    fields.extend(
        _unknown_fields(
            learning,
            {"enabled", "adaptation", "protection", "state_store"},
            "global.router.learning",
        )
    )

    adaptation = learning.get("adaptation")
    fields.extend(
        _unknown_fields(
            adaptation,
            {"enabled", "candidate_set", "strategy"},
            "global.router.learning.adaptation",
        )
    )

    protection = learning.get("protection")
    fields.extend(
        _unknown_fields(
            protection,
            {"enabled", "scope", "identity", "tuning"},
            "global.router.learning.protection",
        )
    )
    if isinstance(protection, dict):
        identity = protection.get("identity")
        fields.extend(
            _unknown_fields(
                identity,
                {"headers"},
                "global.router.learning.protection.identity",
            )
        )
        if isinstance(identity, dict):
            fields.extend(
                _unknown_fields(
                    identity.get("headers"),
                    {"session", "conversation"},
                    "global.router.learning.protection.identity.headers",
                )
            )
        fields.extend(
            _unknown_fields(
                protection.get("tuning"),
                {
                    "idle_timeout_seconds",
                    "min_turns_before_switch",
                    "switch_margin",
                    "stability_weight",
                },
                "global.router.learning.protection.tuning",
            )
        )

    return fields


def _invalid_router_learning_values(data: Dict[str, Any]) -> list[str]:
    errors: list[str] = _invalid_decision_adaptation_values(data)

    global_config = data.get("global")
    if not isinstance(global_config, dict):
        return errors
    router = global_config.get("router")
    if not isinstance(router, dict):
        return errors
    learning = router.get("learning")
    if not isinstance(learning, dict):
        return errors

    adaptation = learning.get("adaptation")
    if isinstance(adaptation, dict):
        candidate_set = adaptation.get("candidate_set")
        if candidate_set not in (None, "", "decision", "tier", "global"):
            errors.append(
                "global.router.learning.adaptation.candidate_set must be "
                "decision, tier, or global"
            )
        strategy = adaptation.get("strategy")
        if strategy not in (None, "", "routing_sampling"):
            errors.append(
                "global.router.learning.adaptation.strategy must be routing_sampling"
            )

    protection = learning.get("protection")
    if not isinstance(protection, dict):
        return errors

    scope = protection.get("scope")
    if scope not in (None, "", "conversation", "session"):
        errors.append(
            "global.router.learning.protection.scope must be conversation or session"
        )

    identity = protection.get("identity")
    if isinstance(identity, dict):
        headers = identity.get("headers")
        if isinstance(headers, dict):
            for name, value in headers.items():
                if str(name).strip() == "":
                    errors.append(
                        "global.router.learning.protection.identity.headers contains an empty key"
                    )
                if str(value).strip() == "":
                    errors.append(
                        f"global.router.learning.protection.identity.headers.{name} "
                        "cannot be empty"
                    )

    tuning = protection.get("tuning")
    if isinstance(tuning, dict):
        for name in (
            "idle_timeout_seconds",
            "min_turns_before_switch",
        ):
            _validate_non_negative_int(
                errors,
                tuning.get(name),
                f"global.router.learning.protection.tuning.{name}",
            )
        for name in (
            "switch_margin",
            "stability_weight",
        ):
            _validate_non_negative_number(
                errors,
                tuning.get(name),
                f"global.router.learning.protection.tuning.{name}",
            )

    return errors


def _invalid_decision_adaptation_values(data: Dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for recipe_index, recipe in enumerate(data.get("recipes") or []):
        document = recipe.get("document") if isinstance(recipe, dict) else None
        decisions = document.get("decisions") if isinstance(document, dict) else None
        if not isinstance(decisions, list):
            continue
        for index, decision in enumerate(decisions):
            if not isinstance(decision, dict):
                continue
            adaptations = decision.get("adaptations")
            if not isinstance(adaptations, dict):
                continue
            decision_mode = str(adaptations.get("mode") or "apply").strip() or "apply"
            if decision_mode not in {"apply", "observe", "bypass"}:
                continue
            for component_name in ("adaptation", "protection"):
                component = adaptations.get(component_name)
                if not isinstance(component, dict):
                    continue
                component_mode = str(component.get("mode") or "").strip()
                if not component_mode:
                    continue
                path = (
                    f"recipes[{recipe_index}].document.decisions[{index}]"
                    f".adaptations.{component_name}.mode"
                )
                if decision_mode == "bypass" and component_mode != "bypass":
                    errors.append(
                        f"{path} cannot be {component_mode} when adaptations.mode is bypass"
                    )
                elif decision_mode == "observe" and component_mode == "apply":
                    errors.append(
                        f"{path} cannot be apply when adaptations.mode is observe"
                    )
    return errors


def _validate_non_negative_int(
    errors: list[str],
    value: Any,
    path: str,
) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        errors.append(f"{path} must be an integer >= 0")
        return
    if value < 0:
        errors.append(f"{path} must be >= 0")


def _router_learning_schema_errors(data: Dict[str, Any]) -> list[str]:
    global_config = data.get("global")
    if not isinstance(global_config, dict):
        return []
    router = global_config.get("router")
    if not isinstance(router, dict) or "learning" not in router:
        return []
    learning = router.get("learning")
    if not isinstance(learning, dict):
        return ["global.router.learning must be an object"]
    try:
        RouterLearningConfig.model_validate(learning)
    except ValidationError as exc:
        errors: list[str] = []
        for error in exc.errors():
            loc = ".".join(str(part) for part in error["loc"])
            path = "global.router.learning"
            if loc:
                path = f"{path}.{loc}"
            errors.append(f"{path}: {error['msg']}")
        return errors
    return []


def _validate_non_negative_number(
    errors: list[str],
    value: Any,
    path: str,
) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        errors.append(f"{path} must be a number >= 0")
        return
    if value < 0:
        errors.append(f"{path} must be >= 0")


def _reject_invalid_config_surfaces(data: Dict[str, Any]) -> None:
    removed_fields = _removed_config_fields(data)
    if removed_fields:
        joined_fields = ", ".join(removed_fields)
        raise ConfigParseError(
            f"Removed config fields are not supported: {joined_fields}."
        )

    removed_router_learning_fields = _removed_router_learning_fields(data)
    if removed_router_learning_fields:
        joined_fields = ", ".join(removed_router_learning_fields)
        raise ConfigParseError(
            "Removed Router Learning config fields are no longer supported: "
            f"{joined_fields}. Use `global.router.learning.adaptation` for online "
            "model-choice learning, `global.router.learning.protection` for "
            "session or conversation protection, and "
            "`recipes[].document.decisions[].adaptations` "
            "only when a decision needs apply/observe/bypass control or a local "
            "adaptation candidate_set override."
        )

    unsupported_router_learning_fields = _unsupported_router_learning_fields(data)
    if unsupported_router_learning_fields:
        joined_fields = ", ".join(unsupported_router_learning_fields)
        raise ConfigParseError(
            "Unsupported Router Learning config fields: "
            f"{joined_fields}. The clean public API is "
            "`global.router.learning.enabled`, "
            "`global.router.learning.adaptation`, "
            "`global.router.learning.protection`, and "
            "`global.router.learning.state_store`, plus "
            "`recipes[].document.decisions[].adaptations`."
        )

    invalid_router_learning_values = _invalid_router_learning_values(data)
    if invalid_router_learning_values:
        joined_errors = "; ".join(invalid_router_learning_values)
        raise ConfigParseError(
            f"Invalid Router Learning config values: {joined_errors}."
        )

    router_learning_schema_errors = _router_learning_schema_errors(data)
    if router_learning_schema_errors:
        joined_errors = "; ".join(router_learning_schema_errors)
        raise ConfigParseError(
            f"Invalid Router Learning config values: {joined_errors}."
        )


ConfigArtifact = UserConfig | RecipeDistribution


def _read_config_document(config_path: str) -> Dict[str, Any]:
    config_file = Path(config_path)
    if not config_file.exists():
        raise ConfigParseError(f"Configuration file not found: {config_path}")

    try:
        with open(config_file, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigParseError(f"Invalid YAML syntax: {e}")
    except Exception as e:
        raise ConfigParseError(f"Failed to read configuration file: {e}")

    if not data:
        raise ConfigParseError("Configuration file is empty")
    if not isinstance(data, dict):
        raise ConfigParseError("Configuration root must be a YAML mapping")
    return data


def _validate_config_document(
    model: type[UserConfig] | type[RecipeDistribution],
    data: Dict[str, Any],
) -> ConfigArtifact:
    try:
        return model.model_validate(data)
    except ValidationError as e:
        errors = []
        for error in e.errors():
            loc = " -> ".join(str(x) for x in error["loc"])
            errors.append(f"  • {loc}: {error['msg']}")
        error_msg = "Configuration validation failed:\n" + "\n".join(errors)
        raise ConfigParseError(error_msg) from e
    except Exception as e:
        raise ConfigParseError(f"Unexpected error during validation: {e}") from e


def _log_config_summary(config: ConfigArtifact) -> None:
    log.info("Configuration parsed successfully")
    log.info(f"  Version: {config.version}")
    listeners = config.listeners if isinstance(config, UserConfig) else []
    entrypoints = config.entrypoints if isinstance(config, UserConfig) else []
    models = config.models if isinstance(config, UserConfig) else []
    log.info(f"  Listeners: {len(listeners)}")
    decisions = sum(len(recipe.document.decisions) for recipe in config.recipes)
    log.info(f"  Entrypoints: {len(entrypoints)}")
    log.info(f"  Recipes: {len(config.recipes)}")
    log.info(f"  Decisions: {decisions}")
    log.info(f"  Models: {len(models)}")


def parse_user_config(config_path: str, *, log_summary: bool = True) -> UserConfig:
    """Parse one complete v0.4 runtime manifest."""

    data = _read_config_document(config_path)

    _reject_invalid_config_surfaces(data)
    config = _validate_config_document(UserConfig, data)
    if not isinstance(config, UserConfig):  # pragma: no cover - type narrowing
        raise ConfigParseError("Expected a runtime manifest")
    if log_summary:
        _log_config_summary(config)
    return config


def parse_config_artifact(
    config_path: str, *, log_summary: bool = True
) -> ConfigArtifact:
    """Parse a runtime manifest or a portable Recipe-only distribution.

    Recipe distributions have exactly the public ``version`` and ``recipes``
    top-level fields. Any runtime-owned field makes the document a runtime
    manifest, which remains subject to the complete standalone or managed
    runtime contract.
    """

    data = _read_config_document(config_path)
    _reject_invalid_config_surfaces(data)
    model = (
        RecipeDistribution
        if "recipes" in data and set(data).issubset({"version", "recipes"})
        else UserConfig
    )
    config = _validate_config_document(model, data)
    if log_summary:
        _log_config_summary(config)
    return config


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

    for recipe_name, document in iter_routing_profiles(config):
        seen = {}
        for family, signal_name in iter_named_signal_entries(document.signals):
            if signal_name in seen:
                errors.append(
                    f"Duplicate signal name '{signal_name}' in Recipe "
                    f"'{recipe_name}' {family} (already defined in {seen[signal_name]})"
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

    for recipe_name, document in iter_routing_profiles(config):
        seen = set()
        for domain in document.signals.domains or []:
            if domain.name in seen:
                errors.append(
                    f"Duplicate domain name '{domain.name}' in Recipe '{recipe_name}'"
                )
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

    for model in config.models:
        if model.name in seen:
            errors.append(f"Duplicate model name '{model.name}'")
        seen.add(model.name)

    return errors
