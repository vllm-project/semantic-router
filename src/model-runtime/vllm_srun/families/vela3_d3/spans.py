"""Span questions of the Vela 3.0 d3 sizes (the released ``Vela3MiniModel.span_answer`` and ``vela3_span.py``).

A span question is read here and planned as one ``SpanItem``: the package's
policy routes it (``span_policy.py``), the renderer lays its target out in
windows of d3's own prompt (``span_render.py``), and the reader runs and
answers it (``span_read.py``). The questions follow the runtime's contract:
``over`` names the target part (by default the answer part, else the user
part, else the only one; a state of several parts with neither needs
``over``), ``threshold`` overrides the package's (one above it keeps the
package's spans that reach it, whole), ``preset`` asks a trained label set,
and answers are ``spans`` with each
span's mean word probability. The windows read the whole target, so
``overflow`` is ``window`` and a question that sets ``require_full_input``
reads its input completely.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ...errors import SCAN_BUDGET_EXCEEDED, QuestionError
from ...systemone import (
    REQUEST_ROLES,
    STATE_KEY_ROLES,
    listed_options,
    named_options,
    require_full_input,
)
from ..vela3_shared import SPAN_HEADS, expand_preset, invalid, text, threshold_of
from .package import SPAN_TARGET_TOKENS
from .span_policy import Policy
from .span_read import SpanItem, SpanReader
from .span_read import answer as span_answer
from .span_render import Renderer

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


def is_span(question: Any) -> bool:
    """A span question, or a preset (which is one)."""
    return isinstance(question, dict) and (
        question.get("type") == "span" or "preset" in question
    )


def parts_of(state: Any) -> list[tuple[str, str]]:
    """A span state's parts by name: text is the user part, an object of text parts its keys in order, and,
    with ``roles: null`` (no part has a role), an object the user's content as Decision 3 writes it.
    """
    if isinstance(state, str):
        return [("user", state)]
    if (
        isinstance(state, dict)
        and state
        and all(isinstance(v, str) for v in state.values())
    ):
        if REQUEST_ROLES.get() is None:
            return [("user", json.dumps(state, ensure_ascii=False))]
        return list(state.items())
    raise invalid("span questions need a text state or an object of text parts")


def default_target(names: list[str]) -> str:
    """The part a span question reads without ``over``, as the contract says: the answer part, else the user
    part (the first part ``roles`` gives that role, or without roles the first key whose implicit role it is),
    else the only part. A state of several parts with neither names no part: the question is invalid rather
    than answered from one of them, which would leave the others unread."""
    roles = REQUEST_ROLES.get()

    def role_of(name: str) -> Any:
        if isinstance(roles, dict):
            return roles.get(name)
        return STATE_KEY_ROLES.get(name.lower(), "context")

    for role in ("answer", "user"):
        found = [name for name in names if role_of(name) == role]
        if found:
            return found[0]
    if len(names) == 1:
        return names[0]
    raise invalid(
        "a span question reads one part of the state, and this state has no answer or user part:"
        f" name the part with over (one of {names})"
    )


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


def over_budget(tokens: int) -> QuestionError | None:
    """Why a span target of ``tokens`` tokens is refused (over ``SPAN_TARGET_TOKENS``), or None."""
    if tokens <= SPAN_TARGET_TOKENS:
        return None
    return QuestionError(
        SCAN_BUDGET_EXCEEDED,
        f"the span target has {tokens} tokens, over this model's span budget of "
        f"{SPAN_TARGET_TOKENS} tokens; nothing was read",
        "scan_budget",
    )


@dataclass(frozen=True)
class Spans:
    """A model's span questions: planned with the package's policy and the base's renderer, read by its heads."""

    policy: Policy
    renderer: Renderer
    reader: SpanReader

    def plan(
        self,
        question_id: str,
        question: dict[str, Any],
        state: Any,
        length_error: Callable[[int], QuestionError | None],
    ) -> SpanItem:
        """One span question's item, as the released ``span_answer`` prepares it; ``QuestionError`` if invalid.

        The target may hold at most ``SPAN_TARGET_TOKENS`` tokens
        (``scan_budget_exceeded``, checked before it is laid out). Every window, whose other parts are not
        windowed, must pass the base's ``length_error``.
        """
        if "preset" in question:
            question = expand_preset(question, self.policy.config.schemas)
        unknown = sorted(set(question) - FIELDS)
        if unknown:
            raise invalid(f"span questions do not take {unknown}")
        if question.get("overflow") not in (None, WINDOWED):
            raise invalid(
                f"span questions read their whole target in windows: overflow must be {WINDOWED!r}"
            )
        require_full_input(question)
        if question.get("head") not in (None, *SPAN_HEADS):
            raise invalid("this model has no broad span head")
        parts = parts_of(state)
        roles = [role for role, _ in parts]
        target = question.get("over") or default_target(roles)
        if not isinstance(target, str) or target not in roles:
            raise invalid(f"over names no part of the state: {target!r}")
        labels = labels_of(question)
        names = [name for name, _ in labels]
        instructions = question.get("instructions")
        if instructions is not None and not isinstance(instructions, str):
            raise invalid("span instructions must be a string")
        override = threshold_of(question)
        routed = self.policy.resolve(labels, instructions)
        windows, count = self.renderer.windows(
            parts,
            target,
            routed.question,
            routed.labels,
            over_budget,
        )
        refused = length_error(max(len(window.ids) for window in windows))
        if refused is not None:
            raise refused
        own = self.policy.threshold(
            routed.route, routed.label_set, routed.labels, count
        )
        threshold = own if override is None else override
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
            decode_threshold=own if threshold > own else None,
            evidence=routed.evidence,
            text=dict(parts)[target],
            windows=tuple(windows),
        )

    def answer(self, item: SpanItem) -> dict[str, Any]:
        return span_answer(item, *self.reader.logits(item))
