"""The names the systemone signal, its conditions and the systemone selector
had before: decision.

The Router reads a configuration that still uses them as systemone, with a
warning that names `vllm-sr config migrate`. The CLI's checks read them the
same way, so they judge what the Router runs.
"""

from __future__ import annotations

import copy
from typing import Any

RENAMED = "decision"
CANONICAL = "systemone"


def read_renamed_forms(data: dict[str, Any]) -> dict[str, Any]:
    """A copy of data with the renamed forms written as systemone."""
    data = copy.deepcopy(data)
    for routing in _routing_scopes(data):
        signals = routing.get("signals")
        if isinstance(signals, dict):
            _rename_signals(signals)
            for rule in _items(signals.get("complexity")):
                _rename_conditions(rule.get("composer"))
        for decision in _items(routing.get("decisions")):
            _rename_conditions(decision.get("rules"))
            _rename_algorithm(decision.get("algorithm"))
        projections = routing.get("projections")
        if isinstance(projections, dict):
            for score in _items(projections.get("scores")):
                for item in _items(score.get("inputs")):
                    _rename_type(item)
    return data


def _routing_scopes(data: dict[str, Any]) -> list[dict[str, Any]]:
    scopes = [data.get("routing")]
    scopes.extend(recipe.get("routing") for recipe in _items(data.get("recipes")))
    return [scope for scope in scopes if isinstance(scope, dict)]


def _items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _rename_signals(signals: dict[str, Any]) -> None:
    if RENAMED not in signals:
        return
    existing, renamed = signals.get(CANONICAL), signals[RENAMED]
    if existing is None:
        signals[CANONICAL] = signals.pop(RENAMED)
    elif isinstance(existing, list) and isinstance(renamed, list):
        signals[CANONICAL] = existing + signals.pop(RENAMED)


def _rename_conditions(node: Any) -> None:
    if not isinstance(node, dict):
        return
    _rename_type(node)
    for child in _items(node.get("conditions")):
        _rename_conditions(child)


def _rename_algorithm(algorithm: Any) -> None:
    if not isinstance(algorithm, dict):
        return
    _rename_type(algorithm)
    if RENAMED in algorithm and CANONICAL not in algorithm:
        algorithm[CANONICAL] = algorithm.pop(RENAMED)


def _rename_type(node: dict[str, Any]) -> None:
    if node.get("type") == RENAMED:
        node["type"] = CANONICAL
