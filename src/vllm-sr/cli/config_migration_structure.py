"""Rewrite the v0.3 layout as the v0.5 one, as the Router's migration does.

The model cards become facts on providers.models, the top-level routing block
becomes the recipe named default, each recipe's routing block moves onto the
recipe, and with several recipes one chat entrypoint is marked default.
"""

from __future__ import annotations

from typing import Any

from cli.config_migration_notes import MigrationNotes
from cli.config_recipe_scopes import DEFAULT_ENTRYPOINT_MODEL
from cli.config_structure_forms import CURRENT_VERSION
from cli.models import DEFAULT_RECIPE_NAME, MODEL_FACT_FIELDS

_SEVERAL_RECIPES = 2

_CANONICAL_ORDER = (
    "version",
    "listeners",
    "providers",
    "global",
    "recipes",
    "entrypoints",
    "evaluation",
)


def restructure_layout(canonical: dict[str, Any], notes: MigrationNotes) -> None:
    """Rewrite canonical, a document in the v0.3 layout, in place."""
    routing = canonical.pop("routing", None)
    recipes = canonical.get("recipes")
    if not isinstance(recipes, list):
        recipes = []
    restructured = isinstance(routing, dict) or any(
        isinstance(recipe, dict) and "routing" in recipe for recipe in recipes
    )
    if isinstance(routing, dict):
        _move_model_cards(canonical, routing.pop("modelCards", None), notes)
        _add_default_recipe(recipes, routing)
    for recipe in recipes:
        if isinstance(recipe, dict) and "routing" in recipe:
            _flatten_recipe_routing(recipe)
    if recipes:
        canonical["recipes"] = recipes
    if restructured:
        _mark_default_entrypoint(canonical, recipes, notes)
    canonical["version"] = CURRENT_VERSION
    _order_keys(canonical)


def _move_model_cards(
    canonical: dict[str, Any], cards: Any, notes: MigrationNotes
) -> None:
    if not isinstance(cards, list) or not cards:
        return
    providers = canonical.setdefault("providers", {})
    models = providers.setdefault("models", [])
    loras = {
        lora.get("name")
        for card in cards
        if isinstance(card, dict)
        for lora in card.get("loras") or []
        if isinstance(lora, dict)
    }
    for card in cards:
        if not isinstance(card, dict) or not card.get("name"):
            continue
        name = card["name"]
        entry = _provider_model_for_card(models, name)
        if entry is None and name in loras:
            if len(card) > 1:
                notes.changed(
                    f"routing.modelCards[{name}]",
                    f"the card of LoRA adapter {name} is dropped; its loras[] "
                    "entry holds its name and description",
                )
            continue
        if entry is None:
            entry = {"name": name}
            models.append(entry)
            notes.action(
                f"providers.models[{name}]",
                f"model card {name} named no providers.models[] entry; it is a "
                "metadata-only entry, so add its backend_refs or drop it",
            )
        for field, value in card.items():
            if field == "name" or field not in MODEL_FACT_FIELDS:
                continue
            if field in entry:
                notes.changed(
                    f"providers.models[{entry.get('name')}].{field}",
                    f"model card {name} also set {field}; the entry's own value "
                    "is kept",
                )
                continue
            entry[field] = value
    notes.changed(
        "routing.modelCards",
        "each card's fields are on the providers.models[] entry of the same name",
    )


def _provider_model_for_card(models: list[Any], card: str) -> dict[str, Any] | None:
    by_catalog = None
    for item in models:
        if not isinstance(item, dict):
            continue
        catalog = item.get("catalog")
        if item.get("name") == card and catalog in (None, "", card):
            return item
        if by_catalog is None and catalog == card:
            by_catalog = item
    return by_catalog


def _add_default_recipe(recipes: list[Any], routing: dict[str, Any]) -> None:
    if any(
        isinstance(recipe, dict) and recipe.get("name") == DEFAULT_RECIPE_NAME
        for recipe in recipes
    ):
        if not routing:
            return
        raise ValueError(
            f"a recipe is already named {DEFAULT_RECIPE_NAME}; rename it first"
        )
    recipes.insert(0, {"name": DEFAULT_RECIPE_NAME, **routing})


def _flatten_recipe_routing(recipe: dict[str, Any]) -> None:
    routing = recipe.pop("routing")
    if not isinstance(routing, dict):
        return
    for field in routing:
        if field in recipe:
            raise ValueError(f"recipe {recipe.get('name')} also sets {field}; keep one")
    recipe.update(routing)


def _mark_default_entrypoint(
    canonical: dict[str, Any], recipes: list[Any], notes: MigrationNotes
) -> None:
    entrypoints = canonical.get("entrypoints")
    if not isinstance(entrypoints, list):
        entrypoints = []
    if any(isinstance(entry, dict) and entry.get("default") for entry in entrypoints):
        return
    if not any(
        isinstance(recipe, dict) and recipe.get("name") == DEFAULT_RECIPE_NAME
        for recipe in recipes
    ):
        recipes.insert(0, {"name": DEFAULT_RECIPE_NAME})
        canonical["recipes"] = recipes
        notes.changed(
            f"recipes[{DEFAULT_RECIPE_NAME}]",
            "the default routing profile was empty; it stays an empty recipe, so "
            f"{DEFAULT_ENTRYPOINT_MODEL} requests go to the default model as before",
        )
    if len(recipes) < _SEVERAL_RECIPES:
        return
    for entry in entrypoints:
        if (
            isinstance(entry, dict)
            and entry.get("recipe") == DEFAULT_RECIPE_NAME
            and entry.get("api") in (None, "", "chat")
        ):
            entry["default"] = True
            return
    entrypoints.insert(
        0,
        {
            "model_names": [DEFAULT_ENTRYPOINT_MODEL],
            "recipe": DEFAULT_RECIPE_NAME,
            "default": True,
        },
    )
    canonical["entrypoints"] = entrypoints


def _order_keys(canonical: dict[str, Any]) -> None:
    ordered = {key: canonical.pop(key) for key in _CANONICAL_ORDER if key in canonical}
    rest = dict(canonical)
    canonical.clear()
    canonical.update(ordered)
    canonical.update(rest)
