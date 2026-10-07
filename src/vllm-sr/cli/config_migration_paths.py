"""Locations in a canonical configuration that several migrations visit."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def dict_at(value: Any, *keys: str, default: dict[str, Any] | None = None):
    """The mapping at ``keys``, or ``default`` when any step is not a mapping."""

    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return value if isinstance(value, dict) else default


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def binding_maps(canonical: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    catalog_bindings = dict_at(canonical, "global", "model_catalog", "bindings")
    if catalog_bindings is not None:
        yield "global.model_catalog.bindings", catalog_bindings
    routing_bindings = dict_at(canonical, "routing", "model_bindings")
    if routing_bindings is not None:
        yield "routing.model_bindings", routing_bindings
    for index, recipe in enumerate(as_list(canonical.get("recipes"))):
        bindings = dict_at(recipe, "routing", "model_bindings")
        if bindings is not None:
            yield f"recipes[{index}].routing.model_bindings", bindings


def signal_maps(canonical: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    signals = dict_at(canonical, "routing", "signals")
    if signals is not None:
        yield "routing.signals", signals
    for index, recipe in enumerate(as_list(canonical.get("recipes"))):
        recipe_signals = dict_at(recipe, "routing", "signals")
        if recipe_signals is not None:
            yield f"recipes[{index}].routing.signals", recipe_signals


def decisions(canonical: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    routing = dict_at(canonical, "routing", default={})
    for index, decision in enumerate(as_list(routing.get("decisions"))):
        if isinstance(decision, dict):
            yield f"routing.decisions[{index}]", decision
    for recipe_index, recipe in enumerate(as_list(canonical.get("recipes"))):
        routing = dict_at(recipe, "routing", default={})
        for index, decision in enumerate(as_list(routing.get("decisions"))):
            if isinstance(decision, dict):
                yield f"recipes[{recipe_index}].routing.decisions[{index}]", decision
