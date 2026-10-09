"""Recipe, entrypoint, and global profile contract validation."""

from cli.config_contract import (
    CONDITION_TYPE_DOMAIN,
    iter_condition_leaves,
    iter_routing_profiles,
)
from cli.model_runtime_defaults import effective_model_deployments
from cli.models import Entrypoint, UserConfig
from cli.validation_error import ValidationError
from cli.validator_decision_model import public_model_name


def validate_domain_references(config: UserConfig) -> list[ValidationError]:
    """
    Validate that all domain references in decisions exist.

    Args:
        config: User configuration

    Returns:
        list: List of validation errors
    """
    errors = []
    for profile_name, routing in iter_routing_profiles(config):
        domains = routing.signals.domains or []
        decisions = routing.decisions
        effective_domains = [
            domain.model_dump(mode="json", exclude_none=True) for domain in domains
        ]
        if not effective_domains:
            generated_names = {
                condition.name
                for decision in decisions
                for condition in iter_condition_leaves(decision.rules.conditions)
                if condition.type == CONDITION_TYPE_DOMAIN and condition.name
            }
            effective_domains = [
                {
                    "name": name,
                    "description": name,
                    "mmlu_categories": ["other"],
                }
                for name in sorted(generated_names)
            ]
        domain_names = {domain["name"] for domain in effective_domains}
        for decision in decisions:
            for condition in iter_condition_leaves(decision.rules.conditions):
                if (
                    condition.type == CONDITION_TYPE_DOMAIN
                    and condition.name not in domain_names
                ):
                    errors.append(
                        ValidationError(
                            f"Decision '{decision.name}' in recipe '{profile_name}' "
                            "references unknown domain "
                            f"'{condition.name}'",
                            field=f"recipes.{profile_name}.routing.decisions.{decision.name}.rules.conditions",
                        )
                    )

    return errors


def _recipe_name_contract(
    config: UserConfig,
) -> tuple[set[str], list[ValidationError]]:
    errors: list[ValidationError] = []
    top_level_has_profile = bool(
        config.routing.budget is not None
        or config.routing.candidate_requirements is not None
        or config.routing.model_bindings
        or config.routing.signals.model_dump(exclude_defaults=True, exclude_none=True)
        or config.routing.projections.model_dump(
            exclude_defaults=True, exclude_none=True
        )
        or config.routing.decisions
        or config.routing.strategy is not None
    )
    recipe_names = {"default"}
    explicit_default_seen = False
    for recipe in config.recipes:
        if recipe.name == "@global":
            errors.append(
                ValidationError(
                    "Recipe name '@global' is reserved for shared model services",
                    field="recipes.@global",
                )
            )
        explicit_default_allowed = (
            recipe.name == "default"
            and not top_level_has_profile
            and not explicit_default_seen
        )
        if recipe.name in recipe_names and not explicit_default_allowed:
            errors.append(
                ValidationError(
                    f"Duplicate recipe name '{recipe.name}'",
                    field=f"recipes.{recipe.name}",
                    hint="Rename one recipe so every recipe has a unique name.",
                )
            )
        if recipe.name == "default":
            explicit_default_seen = True
        recipe_names.add(recipe.name)
    return recipe_names, errors


def _optional_mapping(
    parent: dict, key: str, field: str
) -> tuple[dict, list[ValidationError]]:
    raw_value = parent.get(key)
    if raw_value is None:
        return {}, []
    if isinstance(raw_value, dict):
        return raw_value, []
    return {}, [ValidationError(f"{field} must be a mapping or null", field=field)]


def _normalized_string_list(
    value, field: str
) -> tuple[list[str], list[ValidationError]]:
    if value is None:
        return [], []
    if not isinstance(value, list):
        return [], [
            ValidationError(
                f"{field} must be a list of strings or null",
                field=field,
            )
        ]
    errors = []
    if any(not isinstance(item, str) for item in value):
        errors.append(
            ValidationError(
                f"{field} must contain only strings",
                field=field,
            )
        )
    return [
        item.strip() for item in value if isinstance(item, str) and item.strip()
    ], errors


def effective_entrypoints(config: UserConfig):
    """Source mappings plus the default Chat entrypoint when not overridden."""
    entries = list(config.entrypoints)
    if not any(
        entry.recipe == "default" and entry.api != "systemone" for entry in entries
    ):
        entries.insert(0, Entrypoint(model_names=["vllm-sr/auto"], recipe="default"))
    return entries


def _reserved_routing_models(
    config: UserConfig,
    api: str = "chat",
) -> tuple[set[str], list[ValidationError]]:
    models = [
        model
        for model in config.providers.models
        if (model.api_format == "systemone") == (api == "systemone")
    ]
    names = {model.name for model in models}
    if api == "systemone":
        names.update(
            public_model_name(value)
            for value in effective_model_deployments(config).values()
        )
        return {name for name in names if name}, []
    for model in models:
        if model.provider_model_id:
            names.add(model.provider_model_id)
        names.update((model.external_model_ids or {}).values())
    for card in config.routing.model_cards:
        names.update(adapter.name for adapter in (card.loras or []))
    return {name for name in names if isinstance(name, str) and name}, []


def _validate_entrypoints(
    config: UserConfig,
    recipe_names: set[str],
    reserved_models: dict[str, set[str]],
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    claimed_models: set[tuple[str, str]] = set()
    for index, entrypoint in enumerate(effective_entrypoints(config)):
        if entrypoint.recipe not in recipe_names:
            errors.append(
                ValidationError(
                    f"Entrypoint references unknown recipe '{entrypoint.recipe}'",
                    field=f"entrypoints.{index}.recipe",
                    hint=("Change this to the name of a recipe defined under recipes."),
                )
            )
        api = entrypoint.api or "chat"
        for model_name in entrypoint.model_names:
            identity = (api, model_name)
            if identity in claimed_models:
                errors.append(
                    ValidationError(
                        f"Entrypoint model '{model_name}' is mapped more than once",
                        field=f"entrypoints.{index}.model_names",
                    )
                )
            claimed_models.add(identity)
            if model_name in reserved_models[api]:
                errors.append(
                    ValidationError(
                        f"Entrypoint model '{model_name}' conflicts with a "
                        "configured backend model",
                        field=f"entrypoints.{index}.model_names",
                        hint=(
                            "Use a distinct entrypoint model name; do not reuse "
                            "a configured backend model or provider model ID."
                        ),
                    )
                )
    return errors


def validate_recipe_contracts(config: UserConfig) -> list[ValidationError]:
    recipe_names, errors = _recipe_name_contract(config)
    errors.extend(_validate_entrypoint_contract_keys(config.global_ or {}))
    reserved_models = {}
    for api in ("chat", "systemone"):
        reserved_models[api], alias_errors = _reserved_routing_models(config, api)
        errors.extend(alias_errors)
    errors.extend(_validate_entrypoints(config, recipe_names, reserved_models))
    return errors


def _validate_entrypoint_contract_keys(global_config: dict) -> list[ValidationError]:
    router, errors = _optional_mapping(global_config, "router", "global.router")
    for name in ("auto_model_name", "auto_model_names"):
        if name in router:
            errors.append(
                ValidationError(
                    f"global.router.{name} was removed; configure entrypoints with recipe: default and model_names",
                    field=f"global.router.{name}",
                )
            )
    if "include_config_models_in_list" in router:
        errors.append(
            ValidationError(
                "Use global.router.list_backend_models",
                field="global.router.include_config_models_in_list",
            )
        )
    integrations, nested_errors = _optional_mapping(
        global_config, "integrations", "global.integrations"
    )
    errors.extend(nested_errors)
    looper, nested_errors = _optional_mapping(
        integrations, "looper", "global.integrations.looper"
    )
    errors.extend(nested_errors)
    for family in ("remom", "fusion", "flow"):
        settings, nested_errors = _optional_mapping(
            looper, family, f"global.integrations.looper.{family}"
        )
        errors.extend(nested_errors)
        if "model_names" in settings:
            errors.append(
                ValidationError(
                    "Expose the algorithm through entrypoints and recipes",
                    field=f"global.integrations.looper.{family}.model_names",
                )
            )
    return errors
