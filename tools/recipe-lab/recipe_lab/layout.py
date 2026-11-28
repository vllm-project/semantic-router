"""Locate the recipe that a config edit or a diagnosis applies to.

The canonical v0.5 document keeps signals and decisions per recipe:
``recipes[].signals.<family>[]`` and ``recipes[].decisions[]``. The
``entrypoints`` map public model names to recipes, and one document may hold
several recipes (the multi-objective recipes ship one per public model). A
tool that reads or edits signals and decisions therefore works on one selected
recipe, never on the whole document.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contract import ContractError


@dataclass(frozen=True)
class RecipeRef:
    """One recipe of a config document: its name and its slot."""

    name: str
    index: int

    @property
    def path(self) -> tuple[int | str, ...]:
        """Location of the recipe, from the document root."""
        return ("recipes", self.index)

    def get(self, config: dict[str, Any]) -> dict[str, Any]:
        return config["recipes"][self.index]


def _recipes(config: dict[str, Any]) -> list[dict[str, Any]]:
    recipes = config.get("recipes")
    if not isinstance(recipes, list) or not recipes:
        raise ContractError(
            "base config declares no v0.5 'recipes' list; recipe-lab reads the "
            "canonical layout only (recipes[].signals and recipes[].decisions)"
        )
    for recipe in recipes:
        if not isinstance(recipe, dict) or not recipe.get("name"):
            raise ContractError("every entry of the v0.5 'recipes' list needs a name")
    return recipes


def _public_models(config: dict[str, Any], recipe: str) -> list[str]:
    served: list[str] = []
    for entry in config.get("entrypoints") or []:
        if isinstance(entry, dict) and entry.get("recipe") == recipe:
            served.extend(str(name) for name in entry.get("model_names") or [])
    return served


def _menu(config: dict[str, Any], recipes: list[dict[str, Any]]) -> str:
    return "; ".join(
        f"{recipe['name']!r} ({', '.join(_public_models(config, recipe['name'])) or 'no entrypoint'})"
        for recipe in recipes
    )


def select_recipe(
    config: dict[str, Any],
    *,
    recipe: str | None = None,
    public_model: str | None = None,
) -> RecipeRef:
    """The recipe a tool works on.

    ``recipe`` names it. ``public_model`` selects the recipe that its
    entrypoint serves. Without either, a document with exactly one recipe
    carrying signals or decisions is unambiguous; several are an error that
    lists the choices.
    """
    recipes = _recipes(config)
    if recipe is not None:
        names = [entry["name"] for entry in recipes]
        if recipe not in names:
            raise ContractError(
                f"base config has no recipe {recipe!r}; recipes: {_menu(config, recipes)}"
            )
        return RecipeRef(recipe, names.index(recipe))
    if public_model is not None:
        serving = [
            entry
            for entry in recipes
            if public_model in _public_models(config, entry["name"])
        ]
        if not serving:
            known = sorted(
                {
                    name
                    for entry in recipes
                    for name in _public_models(config, entry["name"])
                }
            )
            raise ContractError(
                f"no entrypoint serves public model {public_model!r}; "
                f"served: {', '.join(known) or 'none'}"
            )
        if len(serving) > 1:
            raise ContractError(
                f"public model {public_model!r} names several recipes "
                f"({', '.join(entry['name'] for entry in serving)}); pass --recipe"
            )
        return RecipeRef(serving[0]["name"], recipes.index(serving[0]))
    routable = [
        entry
        for entry in recipes
        if entry.get("decisions") is not None or entry.get("signals")
    ]
    if len(routable) == 1:
        return RecipeRef(routable[0]["name"], recipes.index(routable[0]))
    if len(recipes) == 1:
        # A strategy-only recipe has no thresholds; the caller reports that.
        return RecipeRef(recipes[0]["name"], 0)
    raise ContractError(
        f"base config declares {len(recipes)} recipes: {_menu(config, recipes)}; "
        "pass --recipe <name> or --public-model <name> to choose one"
    )


def recipe_signals(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    """Every signal of the recipe across its signal families, in document order."""
    signals: list[dict[str, Any]] = []
    for items in (recipe.get("signals") or {}).values():
        if isinstance(items, list):
            signals.extend(item for item in items if isinstance(item, dict))
    return signals


def find_signal(
    config: dict[str, Any], recipe: RecipeRef, signal_name: str
) -> dict[str, Any]:
    """The recipe's signal definition, whichever signal family declares it."""
    hits = [
        signal
        for signal in recipe_signals(recipe.get(config))
        if signal.get("name") == signal_name
    ]
    if not hits:
        raise ContractError(
            f"decision signal {signal_name!r} not found in recipe {recipe.name!r}"
        )
    if len(hits) > 1:
        raise ContractError(
            f"signal name {signal_name!r} is declared {len(hits)} times in recipe "
            f"{recipe.name!r}; recipe-lab cannot tell which one to use"
        )
    return hits[0]
