"""Recipe, entrypoint, and global profile contract validation."""

from cli.config_contract import (
    CONDITION_TYPE_DOMAIN,
    iter_condition_leaves,
    iter_routing_profiles,
)
from cli.models import (
    PROMPT_MIN_CANDIDATES,
    Model,
    RecipeDecision,
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
                            field=f"recipes.{profile_name}.document.decisions.{decision.name}.rules.conditions",
                        )
                    )

    return errors


def _recipe_name_contract(
    config: UserConfig,
) -> tuple[set[str], list[ValidationError]]:
    errors: list[ValidationError] = []
    recipe_names: set[str] = set()
    for recipe in config.recipes:
        if recipe.name in recipe_names:
            errors.append(
                ValidationError(
                    f"Duplicate recipe name '{recipe.name}'",
                    field=f"recipes.{recipe.name}",
                )
            )
        recipe_names.add(recipe.name)
        if not recipe.document.decisions:
            errors.append(
                ValidationError(
                    f"Recipe '{recipe.name}' must contain at least one decision",
                    field=f"recipes.{recipe.name}.document.decisions",
                )
            )
    return recipe_names, errors


def _reserved_routing_models(
    config: UserConfig,
) -> tuple[set[str], list[ValidationError]]:
    names = {model.name for model in config.models}
    for model in config.models:
        names.update(model.card.aliases)
        names.update(model.card.loras)
    names = {name for name in names if isinstance(name, str) and name}
    return names, []


def _validate_entrypoints(
    config: UserConfig,
    reserved_models: set[str],
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    claimed_models: set[str] = set()
    routing_by_recipe = {recipe.name: recipe.document for recipe in config.recipes}
    model_cards = {model.name: model for model in config.models}
    for index, entrypoint in enumerate(config.entrypoints):
        for model_name in (entrypoint.name, *entrypoint.aliases):
            if model_name in claimed_models:
                errors.append(
                    ValidationError(
                        f"Entrypoint model '{model_name}' is mapped more than once",
                        field=f"entrypoints.{index}.aliases",
                    )
                )
            claimed_models.add(model_name)
            if model_name in reserved_models:
                errors.append(
                    ValidationError(
                        f"Entrypoint model '{model_name}' conflicts with a "
                        "configured model or reserved alias",
                        field=f"entrypoints.{index}.aliases",
                    )
                )
        routes = (
            [
                (
                    entrypoint.recipe,
                    entrypoint.assignments or {},
                    f"entrypoints.{index}",
                )
            ]
            if not entrypoint.rules
            else [
                (
                    rule.recipe,
                    rule.assignments,
                    f"entrypoints.{index}.rules.{rule_index}",
                )
                for rule_index, rule in enumerate(entrypoint.rules)
            ]
        )
        for recipe_name, assignments, field in routes:
            routing = routing_by_recipe.get(recipe_name or "")
            if routing is None:
                errors.append(
                    ValidationError(
                        f"Entrypoint references unknown Recipe '{recipe_name}'",
                        field=f"{field}.recipe",
                    )
                )
                continue
            errors.extend(
                _validate_entrypoint_assignments(
                    assignments,
                    recipe_name or "",
                    routing.decisions,
                    model_cards,
                    field,
                )
            )
    return errors


def _validate_entrypoint_assignments(
    assignments,
    recipe_name: str,
    decisions: list[RecipeDecision],
    model_cards: dict[str, Model],
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
            _validate_assignment_model_refs(refs, assignment_field, model_cards)
        )
        errors.extend(
            _validate_binding_algorithm(decision, len(refs), assignment_field)
        )
    return errors


def _validate_assignment_model_refs(
    refs,
    field: str,
    model_cards: dict[str, Model],
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
        model = model_cards.get(ref.model)
        card = model.card if model is not None else None
        if card is None:
            errors.append(
                ValidationError(
                    f"Entrypoint assignment references unknown Model '{ref.model}'",
                    field=f"{ref_field}.model",
                )
            )
            continue
        adapter_names = set(card.loras)
        if ref.lora and ref.lora not in adapter_names:
            errors.append(
                ValidationError(
                    f"Entrypoint assignment references unknown LoRA '{ref.lora}' for Model '{ref.model}'",
                    field=f"{ref_field}.lora",
                )
            )
        if reasoning is not None and not card.reasoning.type:
            errors.append(
                ValidationError(
                    f"Model '{ref.model}' does not support reasoning controls",
                    field=f"{ref_field}.reasoning",
                )
            )
        elif (
            reasoning is not None
            and reasoning.effort
            and reasoning.effort not in card.reasoning.efforts
        ):
            errors.append(
                ValidationError(
                    f"Model '{ref.model}' does not support reasoning effort '{reasoning.effort}'",
                    field=f"{ref_field}.reasoning.effort",
                )
            )
    return errors


def _validate_binding_algorithm(
    decision: RecipeDecision,
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
    reserved_models, alias_errors = _reserved_routing_models(config)
    errors.extend(alias_errors)
    errors.extend(_validate_entrypoints(config, reserved_models))
    return errors
