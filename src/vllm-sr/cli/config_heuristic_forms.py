"""Forms of the heuristic signals other signals absorbed: the events and
structure signals (keywords rules), the actions (one systemone choice
question), the conversation feature image_content (an input_modality rule),
and the conditions and projection inputs that name them.

The Router refuses each one with a line that names `vllm-sr config migrate`,
which rewrites the ones it can; the CLI refuses them with the same lines, in
the Router's order.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

MIGRATE = "run `vllm-sr config migrate`"
REMOVED_SIGNAL_TYPE = "removed_signal_type"
INVALID_VALUE = "invalid_value"

ACTION_CHOICES = ("generate", "explain", "fix", "refactor", "test", "other")

Form = tuple[str, str, str, str]


def removed_heuristic_forms(data: dict[str, Any]) -> list[str]:
    """The one-line load errors of the removed heuristic forms in data."""

    forms = [
        *_signal_collection(
            data,
            "events",
            "the events signal is removed",
            "keywords rules with operator: OR over each rule's terms",
        ),
        *_references(data, "event", "condition type event is removed", "type: keyword"),
        *_text_bytes_rules(data),
        *_signal_collection(
            data,
            "structure",
            "the structure signal is removed",
            "keywords rules with operator: OR and a count or density predicate",
        ),
        *_references(
            data, "structure", "condition type structure is removed", "type: keyword"
        ),
        *_signal_collection(
            data,
            "actions",
            "the actions signal is removed",
            "the systemone choice question action over " + ", ".join(ACTION_CHOICES),
        ),
        *(
            (
                path,
                REMOVED_SIGNAL_TYPE,
                "condition type action is removed",
                "type: systemone, name: action and the action as its label",
            )
            for path, node in _decision_condition_nodes(data)
            if node.get("type") == "action"
        ),
        *(
            (
                path,
                REMOVED_SIGNAL_TYPE,
                "projection input type action is removed",
                "type: systemone and name: action:<action>",
            )
            for path, node in _projection_inputs(data)
            if node.get("type") == "action"
        ),
        *_image_content_rules(data),
    ]
    return [
        f"{path}: {code}: {what}; {MIGRATE} ({becomes})"
        for path, code, what, becomes in forms
    ]


def _signal_collection(
    data: dict[str, Any], collection: str, what: str, becomes: str
) -> Iterator[Form]:
    for prefix, routing in _routing_scopes(data):
        if collection in _signals(routing):
            path = f"{prefix}.signals.{collection}"
            yield (path, REMOVED_SIGNAL_TYPE, what, becomes)


def _references(
    data: dict[str, Any], signal_type: str, what: str, becomes: str
) -> Iterator[Form]:
    for path, node in [*_decision_condition_nodes(data), *_projection_inputs(data)]:
        if node.get("type") == signal_type:
            yield (path, REMOVED_SIGNAL_TYPE, what, becomes)


def _text_bytes_rules(data: dict[str, Any]) -> Iterator[Form]:
    for prefix, routing in _routing_scopes(data):
        for path, rule in _named_items(
            f"{prefix}.signals.structure", _signals(routing).get("structure")
        ):
            if str(_source(rule).get("type", "")).lower() == "text_bytes":
                yield (
                    path,
                    REMOVED_SIGNAL_TYPE,
                    "a structure rule on text_bytes is removed, and no keywords "
                    "rule measures text length",
                    "a context rule on the request's estimated tokens",
                )


def _image_content_rules(data: dict[str, Any]) -> Iterator[Form]:
    for prefix, routing in _routing_scopes(data):
        for path, rule in _named_items(
            f"{prefix}.signals.conversation", _signals(routing).get("conversation")
        ):
            if _source(rule).get("type") == "image_content":
                yield (
                    f"{path}.feature.source.type",
                    INVALID_VALUE,
                    "the conversation feature image_content is removed",
                    "an input_modality rule with modality: image, which counts the "
                    "image parts of user messages",
                )


def _decision_condition_nodes(
    data: dict[str, Any],
) -> Iterator[tuple[str, dict[str, Any]]]:
    for prefix, routing in _routing_scopes(data):
        for path, decision in _named_items(
            f"{prefix}.decisions", routing.get("decisions")
        ):
            yield from _walk(f"{path}.rules", decision.get("rules"))


def _projection_inputs(data: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    for prefix, routing in _routing_scopes(data):
        scores = _mapping_at(routing, "projections").get("scores")
        for path, score in _named_items(f"{prefix}.projections.scores", scores):
            yield from _named_items(f"{path}.inputs", score.get("inputs"))


def _walk(path: str, node: Any) -> Iterator[tuple[str, dict[str, Any]]]:
    if isinstance(node, dict):
        yield path, node
        for key, value in node.items():
            yield from _walk(f"{path}.{key}", value)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from _walk(f"{path}[{_label(item, index)}]", item)


def _named_items(path: str, items: Any) -> Iterator[tuple[str, dict[str, Any]]]:
    for index, item in enumerate(items if isinstance(items, list) else []):
        if isinstance(item, dict):
            yield f"{path}[{_label(item, index)}]", item


def _label(item: Any, index: int) -> str:
    name = item.get("name") if isinstance(item, dict) else None
    return name if isinstance(name, str) and name else str(index)


def _routing_scopes(data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    scopes: list[tuple[str, dict[str, Any]]] = []
    routing = data.get("routing")
    if isinstance(routing, dict):
        scopes.append(("routing", routing))
    for path, recipe in _named_items("recipes", data.get("recipes")):
        if isinstance(recipe.get("routing"), dict):
            scopes.append((f"{path}.routing", recipe["routing"]))
    return scopes


def _signals(routing: dict[str, Any]) -> dict[str, Any]:
    return _mapping_at(routing, "signals")


def _mapping_at(node: Any, *keys: str) -> dict[str, Any]:
    for key in keys:
        node = node.get(key) if isinstance(node, dict) else None
    return node if isinstance(node, dict) else {}


def _source(rule: dict[str, Any]) -> dict[str, Any]:
    return _mapping_at(rule, "feature", "source")
