"""Algorithm forms the Router no longer reads: the types the selectors and
executors replaced, on_error inside an executor's own block or on a selector,
and the global model_selection blocks of the removed selectors.

The Router refuses each one with a line that names `vllm-sr config migrate`,
which rewrites it; the CLI refuses them with the same lines, in the Router's
order: by form, then by place.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

MIGRATE = "run `vllm-sr config migrate`"
REMOVED_ALGORITHM_TYPE = "removed_algorithm_type"
REMOVED_FIELD = "removed_field"

_TRAINED_SELECTOR = (
    "type: systemone, or static for a decision with one candidate; "
    "the trained model is not used"
)

# What each removed type becomes, in the Router's form order.
REMOVED_ALGORITHM_TYPES = {
    "confidence": "type: escalate, with its method and threshold under escalate.accept",
    "cascade": "type: escalate, with its stages and quality under escalate",
    "prompt": "type: systemone, which asks the decision model with the same instructions",
    "router_dc": (
        "type: systemone, which asks the decision model to choose by the "
        "candidates' descriptions, or static for a decision with one candidate"
    ),
    "automix": "type: multi_factor, weighted {quality: 1-t, cost: t} by cost_quality_tradeoff t",
    "hybrid": "type: multi_factor, weighted by hybrid's automix and cost weights",
    "latency_aware": "type: multi_factor, weighted on latency",
    "knn": _TRAINED_SELECTOR,
    "kmeans": _TRAINED_SELECTOR,
    "svm": _TRAINED_SELECTOR,
    "mlp": _TRAINED_SELECTOR,
}

NESTED_ON_ERROR_EXECUTORS = ("fusion", "remom", "ratings", "workflows")

# A decision that names no algorithm type uses static.
SELECTOR_TYPES = ("", "static", "random", "multi_factor", "systemone")

REMOVED_SELECTION_BLOCKS = {
    "method": (
        "global.router.model_selection.method is removed",
        "nothing: each decision's algorithm chooses its own model",
    ),
    "ml": (
        "global.router.model_selection.ml, the ML selectors' trained models, is removed",
        "the systemone selector, which needs no trained model",
    ),
    "router_dc": (
        "global.router.model_selection.router_dc is removed",
        "the systemone selector",
    ),
    "automix": (
        "global.router.model_selection.automix is removed",
        "multi_factor weights on the decisions that used automix",
    ),
    "hybrid": (
        "global.router.model_selection.hybrid is removed",
        "multi_factor weights on the decisions that used hybrid",
    ),
}


def removed_algorithm_forms(data: dict[str, Any]) -> list[str]:
    """The one-line load errors of the removed algorithm forms in data."""

    algorithms = list(_algorithms(data))
    lines = [
        _line(
            path,
            REMOVED_ALGORITHM_TYPE,
            f"algorithm type {removed} is removed",
            becomes,
        )
        for removed, becomes in REMOVED_ALGORITHM_TYPES.items()
        for path, algorithm in algorithms
        if algorithm.get("type") == removed
    ]
    lines += [
        _line(
            path,
            REMOVED_FIELD,
            f"algorithm.{algorithm['type']}.on_error is removed",
            "algorithm.on_error, which every executor reads",
        )
        for path, algorithm in algorithms
        if _nested_on_error(algorithm)
    ]
    lines += [
        _line(
            path,
            REMOVED_FIELD,
            "algorithm.on_error of a selector is removed",
            "nothing: a selector that fails falls back to the first candidate, "
            "and on_error is an executor setting",
        )
        for path, algorithm in algorithms
        if "on_error" in algorithm and _type(algorithm) in SELECTOR_TYPES
    ]
    selection = _mapping(_mapping(_mapping(data).get("global")).get("router")).get(
        "model_selection"
    )
    if isinstance(selection, dict):
        lines += [
            _line(f"global.router.model_selection.{key}", REMOVED_FIELD, what, becomes)
            for key, (what, becomes) in REMOVED_SELECTION_BLOCKS.items()
            if key in selection
        ]
    return lines


def _line(path: str, code: str, what: str, becomes: str) -> str:
    return f"{path}: {code}: {what}; {MIGRATE} ({becomes})"


def _type(algorithm: dict[str, Any]) -> str:
    value = algorithm.get("type")
    return value if isinstance(value, str) else ""


def _nested_on_error(algorithm: dict[str, Any]) -> bool:
    algorithm_type = _type(algorithm)
    if algorithm_type not in NESTED_ON_ERROR_EXECUTORS:
        return False
    on_error = _mapping(algorithm.get(algorithm_type)).get("on_error")
    return on_error not in (None, "") and not isinstance(on_error, (dict, list))


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _algorithms(data: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    for prefix, routing in _routing_scopes(data):
        decisions = routing.get("decisions")
        for index, decision in enumerate(
            decisions if isinstance(decisions, list) else []
        ):
            if not isinstance(decision, dict):
                continue
            algorithm = decision.get("algorithm")
            if not isinstance(algorithm, dict):
                continue
            name = decision.get("name")
            label = name if isinstance(name, str) and name else str(index)
            yield f"{prefix}.decisions[{label}].algorithm", algorithm


def _routing_scopes(data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    scopes: list[tuple[str, dict[str, Any]]] = []
    routing = data.get("routing")
    if isinstance(routing, dict):
        scopes.append(("routing", routing))
    recipes = data.get("recipes")
    for index, recipe in enumerate(recipes if isinstance(recipes, list) else []):
        if not isinstance(recipe, dict) or not isinstance(recipe.get("routing"), dict):
            continue
        name = recipe.get("name")
        label = name if isinstance(name, str) and name else str(index)
        scopes.append((f"recipes[{label}].routing", recipe["routing"]))
    return scopes
