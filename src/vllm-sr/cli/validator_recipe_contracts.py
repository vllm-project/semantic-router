"""Recipe, entrypoint, and global profile contract validation."""

from cli.config_contract import (
    CONDITION_TYPE_DOMAIN,
    iter_condition_leaves,
    iter_routing_profiles,
)
from cli.models import (
    PROMPT_MIN_CANDIDATES,
    Decision,
    Model,
    RoutingModel,
    UserConfig,
)
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
    # Entrypoints are the request-facing Mixture-of-Models authority. Internal
    # looper family names therefore do not reserve public aliases; an explicit
    # Entrypoint may intentionally own one. Physical Models, LoRA adapters, and
    # an enabled implicit auto Entrypoint remain real collision boundaries.
    return names | auto_aliases, auto_errors


def _validate_entrypoints(
    config: UserConfig,
    reserved_models: set[str],
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    claimed_models: set[str] = set()
    routing_by_recipe = {"default": config.routing}
    routing_by_recipe.update({recipe.name: recipe.routing for recipe in config.recipes})
    provider_models = {model.name: model for model in config.providers.models}
    model_cards = {model.name: model for model in config.routing.model_cards}
    for index, entrypoint in enumerate(config.entrypoints):
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
        routing = routing_by_recipe.get(entrypoint.recipe)
        if routing is None:
            errors.append(
                ValidationError(
                    f"Entrypoint references unknown Recipe '{entrypoint.recipe}'",
                    field=f"entrypoints.{index}.recipe",
                )
            )
            continue
        if not routing.decisions:
            errors.append(
                ValidationError(
                    f"Entrypoint references Recipe '{entrypoint.recipe}' without any decisions",
                    field=f"entrypoints.{index}.recipe",
                )
            )
            continue
        if not entrypoint.assignments:
            missing_defaults = [
                decision.name
                for decision in routing.decisions
                if not decision.modelRefs
            ]
            if missing_defaults:
                errors.append(
                    ValidationError(
                        "Entrypoint assignments are required because Recipe "
                        f"'{entrypoint.recipe}' has Decisions without default modelRefs: "
                        + ", ".join(missing_defaults),
                        field=f"entrypoints.{index}.assignments",
                    )
                )
        else:
            errors.extend(
                _validate_entrypoint_assignments(
                    entrypoint.assignments,
                    entrypoint.recipe,
                    routing.decisions,
                    provider_models,
                    model_cards,
                    f"entrypoints.{index}",
                )
            )
    return errors


def _validate_entrypoint_assignments(
    assignments,
    recipe_name: str,
    decisions: list[Decision],
    provider_models: dict[str, Model],
    model_cards: dict[str, RoutingModel],
    field: str,
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    decisions_by_name = {decision.name: decision for decision in decisions}
    if set(assignments) != set(decisions_by_name):
        errors.append(
            ValidationError(
                f"Entrypoint must assign every decision in Recipe '{recipe_name}' by name",
                field=f"{field}.assignments",
            )
        )
    for decision_name, assignment_set in assignments.items():
        decision = decisions_by_name.get(decision_name)
        assignment_field = f"{field}.assignments.{decision_name}"
        if decision is None:
            continue
        refs = assignment_set.models
        if not refs:
            errors.append(
                ValidationError(
                    "Entrypoint assignment must contain at least one Model reference",
                    field=assignment_field,
                )
            )
            continue
        errors.extend(
            _validate_assignment_model_refs(
                refs,
                assignment_field,
                provider_models,
                model_cards,
            )
        )
        errors.extend(
            _validate_binding_algorithm(decision, len(refs), assignment_field)
        )
    return errors


def _validate_assignment_model_refs(
    refs,
    field: str,
    provider_models: dict[str, Model],
    model_cards: dict[str, RoutingModel],
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    seen_targets: set[tuple] = set()
    for ref_index, ref in enumerate(refs):
        ref_field = f"{field}.models.{ref_index}"
        reasoning = ref.reasoning
        target = (
            ref.model,
            ref.lora or "",
            reasoning.enabled if reasoning else None,
            reasoning.effort if reasoning else "",
            reasoning.description if reasoning else "",
        )
        if target in seen_targets:
            errors.append(
                ValidationError(
                    "Entrypoint assignment repeats the same Model, LoRA, and reasoning target",
                    field=ref_field,
                )
            )
        seen_targets.add(target)
        model = provider_models.get(ref.model)
        card = model_cards.get(ref.model)
        if model is None or card is None:
            errors.append(
                ValidationError(
                    f"Entrypoint assignment references unknown Model '{ref.model}'",
                    field=f"{ref_field}.model",
                )
            )
            continue
        adapter_names = {adapter.name for adapter in (card.loras or [])}
        if ref.lora and ref.lora not in adapter_names:
            errors.append(
                ValidationError(
                    f"Entrypoint assignment references unknown LoRA '{ref.lora}' for Model '{ref.model}'",
                    field=f"{ref_field}.lora",
                )
            )
        if reasoning is not None and not model.reasoning_family:
            errors.append(
                ValidationError(
                    f"Model '{ref.model}' does not support reasoning controls",
                    field=f"{ref_field}.reasoning",
                )
            )
    return errors


def _validate_binding_algorithm(
    decision: Decision,
    candidate_count: int,
    field: str,
) -> list[ValidationError]:
    algorithm = decision.algorithm
    if algorithm is None:
        return []

    errors: list[ValidationError] = []
    if algorithm.fusion is not None:
        errors.extend(
            _candidate_pool_errors(
                "fusion min_successful_responses",
                algorithm.fusion.min_successful_responses or 0,
                candidate_count,
                field,
            )
        )
        grounding = algorithm.fusion.grounding
        if grounding is not None:
            errors.extend(
                _candidate_pool_errors(
                    "fusion grounding min_keep",
                    grounding.min_keep or 0,
                    candidate_count,
                    field,
                )
            )
    if algorithm.workflows is not None:
        errors.extend(
            _candidate_pool_errors(
                "workflows min_successful_responses",
                algorithm.workflows.min_successful_responses or 0,
                candidate_count,
                field,
            )
        )
    if algorithm.remom is not None:
        minimum = algorithm.remom.min_successful_responses or 0
        maximum_breadth = max(algorithm.remom.breadth_schedule, default=0)
        if maximum_breadth and minimum > maximum_breadth:
            errors.append(
                ValidationError(
                    f"remom min_successful_responses={minimum} exceeds every configured round breadth (maximum {maximum_breadth})",
                    field=field,
                )
            )
    if algorithm.type == "prompt" and candidate_count < PROMPT_MIN_CANDIDATES:
        errors.append(
            ValidationError(
                "prompt bindings require at least two models",
                field=field,
            )
        )
    return errors


def _candidate_pool_errors(
    label: str,
    minimum: int,
    candidate_count: int,
    field: str,
) -> list[ValidationError]:
    if minimum <= candidate_count:
        return []
    return [
        ValidationError(
            f"{label}={minimum} exceeds the bound candidate pool of {candidate_count}",
            field=field,
        )
    ]


def validate_recipe_contracts(config: UserConfig) -> list[ValidationError]:
    _, errors = _recipe_name_contract(config)
    for recipe in config.recipes:
        if not recipe.routing.decisions:
            errors.append(
                ValidationError(
                    f"Recipe '{recipe.name}' requires at least one decision",
                    field=f"recipes.{recipe.name}.routing.decisions",
                )
            )
    reserved_models, alias_errors = _reserved_routing_models(config)
    errors.extend(alias_errors)
    errors.extend(_validate_entrypoints(config, reserved_models))
    return errors
