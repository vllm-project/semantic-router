"""Resolve human-authored Recipe, Entrypoint, and Model assignments."""

from __future__ import annotations

from typing import Any

from recipe_conformance_values import mapping, reject_fields, sequence


def recipe_profiles(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for recipe_index, raw_recipe in enumerate(sequence(config.get("recipes"))):
        recipe = mapping(raw_recipe)
        reject_fields(recipe, f"recipes[{recipe_index}]", {"id", "revision"})
        name = str(recipe.get("name") or "").strip()
        if not name:
            continue
        if name in profiles:
            raise ValueError(f"duplicate recipe name {name!r}")
        document = mapping(recipe.get("routing"))
        if not document:
            raise ValueError(f"recipe {name!r} has no routing document")
        for decision_index, raw_decision in enumerate(
            sequence(document.get("decisions"))
        ):
            reject_fields(
                mapping(raw_decision),
                f"recipes[{recipe_index}].routing.decisions[{decision_index}]",
                {"id", "revision", "decision_id"},
            )
        profiles[name] = document
    return profiles


def config_auto_entrypoints(config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        name
        for name, recipe_name in _named_entrypoints(config).items()
        if recipe_name == "default"
    )


def config_entrypoints(config: dict[str, Any]) -> dict[str, str]:
    return _named_entrypoints(config)


def config_assigned_models(
    config: dict[str, Any],
) -> dict[tuple[str, str, str], frozenset[str]]:
    """Resolve current-v0.3 Entrypoint assignments to public Model names."""

    decisions = _recipe_decisions(recipe_profiles(config))
    model_names = _public_model_names(config)
    result: dict[tuple[str, str, str], frozenset[str]] = {}
    for raw_entrypoint in sequence(config.get("entrypoints")):
        entrypoint = mapping(raw_entrypoint)
        entrypoint_names = _entrypoint_names(entrypoint)
        for recipe_name, assignments in _entrypoint_bindings(entrypoint):
            recipe_decisions = decisions.get(recipe_name)
            if recipe_decisions is None:
                raise ValueError(
                    f"entrypoint {entrypoint_names[0]!r} references unknown recipe "
                    f"{recipe_name!r}"
                )
            _collect_assignments(
                result,
                entrypoint_names,
                recipe_name,
                recipe_decisions,
                assignments,
                model_names,
            )
    return result


def _entrypoint_names(entrypoint: dict[str, Any]) -> tuple[str, ...]:
    reject_fields(
        entrypoint,
        "entrypoint",
        {"aliases", "id", "name", "recipe_id", "revision", "rules"},
    )
    names = [
        normalized
        for raw_name in sequence(entrypoint.get("model_names"))
        if (normalized := str(raw_name or "").strip())
    ]
    if not names:
        raise ValueError("every entrypoint must have at least one model_names value")
    if len(names) != len(set(names)):
        raise ValueError("entrypoint model_names must be unique")
    return tuple(names)


def _entrypoint_bindings(
    entrypoint: dict[str, Any],
) -> tuple[tuple[str, dict[str, Any]], ...]:
    entrypoint_name = _entrypoint_names(entrypoint)[0]
    recipe_name = str(entrypoint.get("recipe") or "").strip()
    assignments = entrypoint.get("assignments")
    if not recipe_name:
        raise ValueError(f"entrypoint {entrypoint_name!r} has no recipe")
    if not isinstance(assignments, dict):
        raise ValueError(
            f"entrypoint {entrypoint_name!r} assignments must be a mapping"
        )
    return ((recipe_name, assignments),)


def _named_entrypoints(config: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    recipes = recipe_profiles(config)
    for raw_entrypoint in sequence(config.get("entrypoints")):
        entrypoint = mapping(raw_entrypoint)
        names = _entrypoint_names(entrypoint)
        resolved = {recipe for recipe, _ in _entrypoint_bindings(entrypoint)}
        unknown = resolved - recipes.keys()
        if unknown:
            raise ValueError(
                f"entrypoint {names[0]!r} references unknown recipe "
                f"{sorted(unknown)[0]!r}"
            )
        if len(resolved) != 1:
            raise ValueError(
                f"entrypoint {names[0]!r} must resolve to exactly one recipe for conformance"
            )
        recipe = next(iter(resolved))
        for name in names:
            if name in result:
                raise ValueError(f"entrypoint name or alias {name!r} is duplicated")
            result[name] = recipe
    return result


def _recipe_decisions(
    profiles: dict[str, dict[str, Any]],
) -> dict[str, frozenset[str]]:
    result: dict[str, frozenset[str]] = {}
    for recipe_name, document in profiles.items():
        names: set[str] = set()
        for raw_decision in sequence(document.get("decisions")):
            name = str(mapping(raw_decision).get("name") or "").strip()
            if not name:
                raise ValueError(
                    f"recipe {recipe_name!r} has a decision without a name"
                )
            if name in names:
                raise ValueError(
                    f"recipe {recipe_name!r} has duplicate decision name {name!r}"
                )
            names.add(name)
        result[recipe_name] = frozenset(names)
    return result


def _public_model_names(config: dict[str, Any]) -> dict[str, frozenset[str]]:
    providers = mapping(config.get("providers"))
    routing = mapping(config.get("routing"))
    card_aliases: dict[str, frozenset[str]] = {}
    for index, raw_card in enumerate(sequence(routing.get("modelCards"))):
        card = mapping(raw_card)
        reject_fields(card, f"routing.modelCards[{index}]", {"id", "revision"})
        card_name = str(card.get("name") or "").strip()
        if not card_name:
            raise ValueError("every model card must have a name")
        if card_name in card_aliases:
            raise ValueError(f"duplicate model card name {card_name!r}")
        card_aliases[card_name] = frozenset(
            str(alias).strip()
            for alias in sequence(card.get("aliases"))
            if str(alias).strip()
        )

    result: dict[str, frozenset[str]] = {}
    for index, raw_model in enumerate(sequence(providers.get("models"))):
        model = mapping(raw_model)
        reject_fields(
            model,
            f"providers.models[{index}]",
            {"aliases", "card", "id", "revision"},
        )
        name = str(model.get("name") or "").strip()
        if not name:
            raise ValueError("every model must have a name")
        if name in result:
            raise ValueError(f"duplicate model name {name!r}")
        result[name] = frozenset({name, *card_aliases.get(name, frozenset())})
    return result


def _collect_assignments(
    result: dict[tuple[str, str, str], frozenset[str]],
    entrypoint_names: tuple[str, ...],
    recipe_name: str,
    recipe_decisions: frozenset[str],
    assignments: dict[str, Any],
    model_names: dict[str, frozenset[str]],
) -> None:
    for raw_decision_name, raw_assignment in assignments.items():
        decision_name = str(raw_decision_name).strip()
        if decision_name not in recipe_decisions:
            raise ValueError(
                f"entrypoint {entrypoint_names[0]!r} assignment references unknown "
                f"decision {decision_name!r} in recipe {recipe_name!r}"
            )
        assigned = _assigned_model_names(
            entrypoint_names[0], decision_name, raw_assignment, model_names
        )
        for entrypoint_name in entrypoint_names:
            key = (entrypoint_name, recipe_name, decision_name)
            result[key] = result.get(key, frozenset()) | assigned


def _assigned_model_names(
    entrypoint_name: str,
    decision_name: str,
    raw_assignment: Any,
    model_names: dict[str, frozenset[str]],
) -> frozenset[str]:
    assigned: set[str] = set()
    for index, raw_ref in enumerate(sequence(mapping(raw_assignment).get("models"))):
        model_ref = mapping(raw_ref)
        reject_fields(
            model_ref,
            f"entrypoint {entrypoint_name!r} assignment "
            f"{decision_name!r}.models[{index}]",
            {"id", "revision", "model_id"},
        )
        model_name = str(model_ref.get("model") or "").strip()
        names = model_names.get(model_name)
        if names is None:
            raise ValueError(
                f"entrypoint {entrypoint_name!r} assignment references "
                f"unknown model {model_name!r}"
            )
        assigned.update(names)
    if not assigned:
        raise ValueError(
            f"entrypoint {entrypoint_name!r} assignment for decision "
            f"{decision_name!r} has no models"
        )
    return frozenset(assigned)
