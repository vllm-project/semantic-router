"""Recipe, entrypoint, and global profile contract validation."""

from cli.config_contract import iter_routing_profiles
from cli.models import UserConfig
from cli.validation_error import ValidationError


def _iter_condition_nodes(conditions):
    """Depth-first traversal over recursive condition trees."""
    if not conditions:
        return
    for condition in conditions:
        yield condition
        if getattr(condition, "conditions", None):
            yield from _iter_condition_nodes(condition.conditions)


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
                for condition in _iter_condition_nodes(decision.rules.conditions)
                if condition.type == "domain" and condition.name
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
            for condition in _iter_condition_nodes(decision.rules.conditions):
                if condition.type == "domain" and condition.name not in domain_names:
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
        config.routing.signals.model_dump(exclude_defaults=True, exclude_none=True)
        or config.routing.projections.model_dump(
            exclude_defaults=True, exclude_none=True
        )
        or config.routing.decisions
        or config.routing.strategy is not None
    )
    recipe_names = {"default"}
    explicit_default_seen = False
    for recipe in config.recipes:
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


def _reserved_auto_aliases(
    global_config: dict,
) -> tuple[set[str], list[ValidationError]]:
    router, errors = _optional_mapping(global_config, "router", "global.router")
    if "auto_model_names" in router and isinstance(
        router.get("auto_model_names"), list
    ):
        names, name_errors = _normalized_string_list(
            router.get("auto_model_names"),
            "global.router.auto_model_names",
        )
        return set(names), errors + name_errors

    raw_names = router.get("auto_model_names")
    if raw_names is not None:
        _, name_errors = _normalized_string_list(
            raw_names,
            "global.router.auto_model_names",
        )
        errors.extend(name_errors)
    auto_model_name = router.get("auto_model_name") or "MoM"
    if not isinstance(auto_model_name, str):
        errors.append(
            ValidationError(
                "global.router.auto_model_name must be a string or null",
                field="global.router.auto_model_name",
            )
        )
        auto_model_name = "MoM"
    return {"vllm-sr/auto", "auto", auto_model_name.strip()}, errors


def _reserved_looper_aliases(
    global_config: dict,
) -> tuple[set[str], list[ValidationError]]:
    integrations, errors = _optional_mapping(
        global_config, "integrations", "global.integrations"
    )
    looper, looper_errors = _optional_mapping(
        integrations, "looper", "global.integrations.looper"
    )
    errors.extend(looper_errors)
    aliases: set[str] = set()
    for family, default_name in (
        ("remom", "vllm-sr/remom"),
        ("fusion", "vllm-sr/fusion"),
        ("flow", "vllm-sr/flow"),
    ):
        field = f"global.integrations.looper.{family}"
        family_config, family_errors = _optional_mapping(looper, family, field)
        names, name_errors = _normalized_string_list(
            family_config.get("model_names"),
            f"{field}.model_names",
        )
        errors.extend(family_errors)
        errors.extend(name_errors)
        aliases.update(names or [default_name])
    return aliases, errors


def _reserved_routing_models(
    config: UserConfig,
) -> tuple[set[str], list[ValidationError]]:
    names = {model.name for model in config.providers.models}
    for card in config.routing.model_cards:
        names.add(card.name)
        names.update(adapter.name for adapter in (card.loras or []))
    names = {name for name in names if isinstance(name, str) and name}
    global_config = config.global_ or {}
    auto_aliases, auto_errors = _reserved_auto_aliases(global_config)
    looper_aliases, looper_errors = _reserved_looper_aliases(global_config)
    return names | auto_aliases | looper_aliases, auto_errors + looper_errors


def _validate_entrypoints(
    config: UserConfig,
    recipe_names: set[str],
    reserved_models: set[str],
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    claimed_models: set[str] = set()
    for index, entrypoint in enumerate(config.entrypoints):
        if entrypoint.recipe not in recipe_names:
            errors.append(
                ValidationError(
                    f"Entrypoint references unknown recipe '{entrypoint.recipe}'",
                    field=f"entrypoints.{index}.recipe",
                )
            )
        for model_name in entrypoint.model_names:
            if model_name in claimed_models:
                errors.append(
                    ValidationError(
                        f"Entrypoint model '{model_name}' is mapped more than once",
                        field=f"entrypoints.{index}.model_names",
                    )
                )
            claimed_models.add(model_name)
            if model_name in reserved_models:
                errors.append(
                    ValidationError(
                        f"Entrypoint model '{model_name}' conflicts with a "
                        "configured model or reserved alias",
                        field=f"entrypoints.{index}.model_names",
                    )
                )
    return errors


def validate_recipe_contracts(config: UserConfig) -> list[ValidationError]:
    recipe_names, errors = _recipe_name_contract(config)
    reserved_models, alias_errors = _reserved_routing_models(config)
    errors.extend(alias_errors)
    errors.extend(_validate_entrypoints(config, recipe_names, reserved_models))
    return errors
