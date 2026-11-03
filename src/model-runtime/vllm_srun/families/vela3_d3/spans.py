"""Span questions of the Vela 3.0 d3 sizes (the released ``Vela3MiniModel.span_answer`` and ``vela3_span.py``).

A span question is read here and planned as one ``SpanItem``: the package's
policy routes it (``span_policy.py``), the renderer lays its target out in
windows of d3's own prompt (``span_render.py``), and the reader runs and
answers it (``span_read.py``). The questions follow the runtime's contract:
``over`` names the target part, ``threshold`` overrides the package's,
``preset`` asks a trained label set, and answers are ``spans`` with each
span's mean word probability. The windows read the whole target, so
``overflow`` is ``window`` and a question that sets ``require_full_input``
reads its input completely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...errors import QuestionError
from ...systemone import listed_options, named_options, require_full_input
from ..decision3.family import Item
from ..vela3_shared import expand_preset, text, threshold_of
from .span_policy import Definitions, Policy, Route
from .span_render import Renderer, Window

FIELDS = frozenset(
    {
        "type",
        "instructions",
        "criteria",
        "labels",
        "threshold",
        "head",
        "over",
        "overflow",
        "require_full_input",
    }
)
# A span question reads its whole target, in windows: the one overflow it takes.
WINDOWED = "window"


def invalid(message: str, reason: str | None = None) -> QuestionError:
    return QuestionError("invalid_question", message, reason)


def is_span(question: Any) -> bool:
    """A span question, or a preset (which is one)."""
    return isinstance(question, dict) and (
        question.get("type") == "span" or "preset" in question
    )


@dataclass(frozen=True)
class SpanItem(Item):
    """A span question: its route, labels, threshold and the windows it reads."""

    route: Route = "span"
    names: tuple[str, ...] = ()
    label_set: str | None = None
    threshold: float = 0.5
    evidence: dict[str, Any] | None = None
    text: str = ""
    windows: tuple[Window, ...] = field(default=())


def parts_of(state: Any) -> list[tuple[str, str]]:
    if isinstance(state, str):
        return [("user", state)]
    if (
        isinstance(state, dict)
        and state
        and all(isinstance(v, str) for v in state.values())
    ):
        return list(state.items())
    raise invalid("span questions need a text state or an object of text parts")


def labels_of(question: dict[str, Any]) -> list[tuple[str, str]]:
    """A span question's ``(label, description)`` pairs, read as the vela3 family reads them.

    ``criteria`` is ``{label: description or null}`` or a list of names,
    ``labels`` a ``[{key, description}]`` list; an undescribed label has "".
    """
    criteria, listed = question.get("criteria"), question.get("labels")
    if criteria is not None and listed is not None:
        raise invalid("use criteria or labels, not both")
    if listed is not None:
        criteria = listed_options(listed, "labels")
    elif isinstance(criteria, list):
        names = [str(name) for name in criteria]
        if len(set(names)) != len(names):
            raise invalid("span labels must be distinct")
        criteria = dict.fromkeys(names)
    options = named_options(criteria, minimum=1)
    return [
        (name, "" if value is None else text(value)) for name, value in options.items()
    ]


def plan_question(
    question_id: str,
    question: dict[str, Any],
    state: Any,
    policy: Policy,
    renderer: Renderer,
    definitions: Definitions,
) -> SpanItem:
    """One span question's item, as the released ``span_answer`` prepares it; ``QuestionError`` if invalid."""
    if "preset" in question:
        question = expand_preset(question, policy.config.schemas)
    unknown = sorted(set(question) - FIELDS)
    if unknown:
        raise invalid(f"span questions do not take {unknown}")
    if question.get("overflow") not in (None, WINDOWED):
        raise invalid(
            f"span questions read their whole target in windows: overflow must be {WINDOWED!r}"
        )
    require_full_input(question)
    if question.get("head") not in (None, "router"):
        raise invalid("this model has no broad span head")
    parts = parts_of(state)
    roles = [role for role, _ in parts]
    target = question.get("over") or ("answer" if "answer" in roles else roles[-1])
    if not isinstance(target, str) or target not in roles:
        raise invalid(f"over names no part of the state: {target!r}")
    labels = labels_of(question)
    names = [name for name, _ in labels]
    instructions = question.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise invalid("span instructions must be a string")
    override = threshold_of(question)
    routed = policy.resolve(labels, instructions, definitions)
    windows, count = renderer.windows(parts, target, routed.question, routed.labels)
    threshold = (
        override
        if override is not None
        else policy.threshold(routed.route, routed.label_set, routed.labels, count)
    )
    return SpanItem(
        question_id=question_id,
        task_type="span",
        ids=list(windows[0].ids),
        keys=names,
        descriptions=[d or None for _, d in routed.labels],
        cost=sum(len(w.ids) for w in windows),
        route=routed.route,
        names=tuple(names),
        label_set=routed.label_set,
        threshold=threshold,
        evidence=routed.evidence,
        text=dict(parts)[target],
        windows=tuple(windows),
    )
