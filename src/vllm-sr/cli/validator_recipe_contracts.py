"""Recipe, entrypoint, and global profile contract validation."""

from dataclasses import dataclass
from urllib.parse import quote_plus

from cli.config_contract import (
    CONDITION_TYPE_DOMAIN,
    iter_condition_leaves,
    iter_routing_profiles,
)
from cli.models import UserConfig
from cli.validation_error import ValidationError


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
        config.routing.candidate_requirements is not None
        or config.routing.data_policy is not None
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


@dataclass(frozen=True)
class _LooperAliasFamily:
    key: str
    label: str
    default_name: str
    algorithm: str


# In the order request routing resolves them, so the first family that lists
# a name is the one that captures it.
_LOOPER_ALIAS_FAMILIES = (
    _LooperAliasFamily("remom", "ReMoM", "vllm-sr/remom", "remom"),
    _LooperAliasFamily("fusion", "Fusion", "vllm-sr/fusion", "fusion"),
    _LooperAliasFamily("flow", "Flow", "vllm-sr/flow", "workflows"),
)


def _looper_aliases(
    global_config: dict,
) -> tuple[list[tuple[_LooperAliasFamily, list[str]]], list[ValidationError]]:
    integrations, errors = _optional_mapping(
        global_config, "integrations", "global.integrations"
    )
    looper, looper_errors = _optional_mapping(
        integrations, "looper", "global.integrations.looper"
    )
    errors.extend(looper_errors)
    aliases = []
    for family in _LOOPER_ALIAS_FAMILIES:
        field = f"global.integrations.looper.{family.key}"
        family_config, family_errors = _optional_mapping(looper, family.key, field)
        names, name_errors = _normalized_string_list(
            family_config.get("model_names"),
            f"{field}.model_names",
        )
        errors.extend(family_errors)
        errors.extend(name_errors)
        aliases.append((family, list(dict.fromkeys(names)) or [family.default_name]))
    return aliases, errors


def _reserved_looper_aliases(
    global_config: dict,
) -> tuple[set[str], list[ValidationError]]:
    aliases, errors = _looper_aliases(global_config)
    return {name for _, names in aliases for name in names}, errors


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
                    hint=("Change this to the name of a recipe defined under recipes."),
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
                        hint=(
                            "Use a distinct entrypoint model name; do not reuse "
                            "a configured model or reserved alias such as "
                            "vllm-sr/auto."
                        ),
                    )
                )
    return errors


def validate_recipe_contracts(config: UserConfig) -> list[ValidationError]:
    recipe_names, errors = _recipe_name_contract(config)
    reserved_models, alias_errors = _reserved_routing_models(config)
    errors.extend(alias_errors)
    errors.extend(_validate_entrypoints(config, recipe_names, reserved_models))
    return errors


def _served_model_names(config: UserConfig) -> set[str]:
    card_loras = {
        card.name: [adapter.name for adapter in card.loras or []]
        for card in config.routing.model_cards
    }
    served: set[str] = set()
    for model in config.providers.models:
        served.add(model.name)
        served.update(card_loras.get(model.catalog or model.name, []))
    return served


def _routing_decision_key(profile_name: str, decision_name: str) -> str:
    if profile_name == "default":
        return decision_name
    return f"{quote_plus(profile_name)}::{quote_plus(decision_name)}"


def _model_ref_names(decision) -> list[str]:
    names = (
        (name or "").strip()
        for model_ref in decision.modelRefs or []
        for name in (model_ref.model, model_ref.lora_name)
    )
    return list(dict.fromkeys(name for name in names if name))


def _decisions_by_model_ref(config: UserConfig) -> dict[str, list[str]]:
    decisions: dict[str, list[str]] = {}
    for profile_name, routing in iter_routing_profiles(config):
        for decision in routing.decisions:
            key = _routing_decision_key(profile_name, decision.name)
            for name in _model_ref_names(decision):
                decisions.setdefault(name, []).append(key)
    return decisions


def looper_alias_collision_warnings(config: UserConfig) -> list[ValidationError]:
    """Warn about each direct Looper alias that is also a model name.

    The alias wins: a request for the name evaluates only the alias's
    decisions, so the model cannot be requested directly. The configuration
    stays valid, and the Router logs the same warning when it loads it.
    """
    aliases, _ = _looper_aliases(config.global_ or {})
    served = _served_model_names(config)
    routed_by = _decisions_by_model_ref(config)
    claimed: set[str] = set()
    warnings: list[ValidationError] = []
    for family, names in aliases:
        for alias in names:
            if alias in claimed:
                continue
            claimed.add(alias)
            decisions = routed_by.get(alias, [])
            if alias not in served and not decisions:
                continue
            uses = []
            if alias in served:
                uses.append("providers.models serves")
            if decisions:
                listed = ", ".join(f"'{name}'" for name in decisions)
                uses.append(f"decisions {listed} route to")
            warnings.append(
                ValidationError(
                    f"{family.label} alias '{alias}' is also a model that "
                    f"{' and '.join(uses)}; requests for it evaluate only "
                    f"{family.algorithm} decisions, so the model cannot be "
                    "requested directly and a request that matches none of "
                    "them fails with no_route",
                    field=f"global.integrations.looper.{family.key}.model_names",
                    hint=(
                        "Give the alias a name that no model uses, such as "
                        f"{family.default_name}."
                    ),
                )
            )
    return warnings
