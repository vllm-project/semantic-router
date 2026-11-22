"""The recipes of a configuration document, each with its path."""

from __future__ import annotations

from typing import Any

DEFAULT_ENTRYPOINT_MODEL = "vllm-sr/auto"


def add_recipe(data: dict[str, Any], recipe: dict[str, Any]) -> None:
    """Add recipe to data. A document that had one recipe marks that recipe's
    entrypoint default first, so the requests it served stay with it."""

    recipes = data.setdefault("recipes", [])
    if len(recipes) == 1 and isinstance(recipes[0], dict):
        _keep_default_entrypoint(data, str(recipes[0].get("name", "")))
    recipes.append(recipe)


def _keep_default_entrypoint(data: dict[str, Any], recipe: str) -> None:
    entrypoints = data.setdefault("entrypoints", [])
    if any(isinstance(entry, dict) and entry.get("default") for entry in entrypoints):
        return
    for entry in entrypoints:
        if (
            isinstance(entry, dict)
            and entry.get("recipe") == recipe
            and entry.get("api") in (None, "chat")
        ):
            entry["default"] = True
            return
    entrypoints.insert(
        0,
        {"model_names": [DEFAULT_ENTRYPOINT_MODEL], "recipe": recipe, "default": True},
    )


def recipe_scopes(data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Each recipe of data with its path: recipes[<name>], or recipes[<index>]
    for a recipe without a name."""

    scopes: list[tuple[str, dict[str, Any]]] = []
    recipes = data.get("recipes")
    for index, recipe in enumerate(recipes if isinstance(recipes, list) else []):
        if not isinstance(recipe, dict):
            continue
        name = recipe.get("name")
        label = name if isinstance(name, str) and name else str(index)
        scopes.append((f"recipes[{label}]", recipe))
    return scopes
