"""Locations in a canonical configuration that several migrations visit, in
the v0.3 layout (the top-level routing block and each recipe's routing block)
that the Router's migration rewrites, and in flat v0.5 recipes."""

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


def routing_profiles(canonical: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Each routing profile of the document with its path."""

    routing = dict_at(canonical, "routing")
    if routing is not None:
        yield "routing", routing
    for index, recipe in enumerate(as_list(canonical.get("recipes"))):
        if not isinstance(recipe, dict):
            continue
        nested = dict_at(recipe, "routing")
        if nested is not None:
            yield f"recipes[{index}].routing", nested
        else:
            yield f"recipes[{index}]", recipe


def binding_maps(canonical: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    catalog_bindings = dict_at(canonical, "global", "model_catalog", "bindings")
    if catalog_bindings is not None:
        yield "global.model_catalog.bindings", catalog_bindings
    for path, profile in routing_profiles(canonical):
        bindings = dict_at(profile, "model_bindings")
        if bindings is not None:
            yield f"{path}.model_bindings", bindings


def signal_maps(canonical: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    for path, profile in routing_profiles(canonical):
        signals = dict_at(profile, "signals")
        if signals is not None:
            yield f"{path}.signals", signals


def decisions(canonical: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    for path, profile in routing_profiles(canonical):
        for index, decision in enumerate(as_list(profile.get("decisions"))):
            if isinstance(decision, dict):
                yield f"{path}.decisions[{index}]", decision
