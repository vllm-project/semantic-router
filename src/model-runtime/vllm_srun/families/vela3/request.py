"""A ``/v1/decisions`` request read the way the Vela 3.0 encoder package reads System One.

Choice and Noul questions are decisions on the backbone (``ub_d3_common``):
the whole state is one user part, a string as it is and anything else as
compact JSON; Noul is the two-option choice ``no`` / ``yes`` (descriptions
empty, no abstain option), so a Noul that describes ``false`` or ``true`` is
invalid. A Choice or Set question whose instructions are a
router text is a router question: it reads the state's typed parts
(``vela2.request.read_state``), defaults to the part its schema was trained on
(else the last part), shows the trained abstain option, and fills an option
the request leaves undescribed (or every option, when it gives none) from the
trained schema. Span questions (and the ``pii`` / ``halu`` presets) read the
typed parts, default to the answer (else the last part) and route by label
set and wording (``routing.Router.span_route``). Score questions and Set
questions that are not router questions are invalid on this model, as are
``head: broad`` spans. ``over``, ``threshold``, ``overflow`` and
``require_full_input`` follow the System One contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ...errors import INVALID_QUESTION, QuestionError, question_error
from ...systemone import (
    content,
    listed_options,
    named_options,
    read_question,
    require_full_input,
)
from ..vela2.request import Question, State, read_state, resolve_over
from .routing import ROUTER_TYPES, Router, RouterQuestion, SpanRoute

DECISION, ROUTER, SPAN = "decision", "router", "span"
QUESTION_TYPES = ("choice", "noul", "set", "span")
FAMILY_FIELDS = frozenset({"over", "preset", "overflow", "require_full_input"})
SPAN_FIELDS = frozenset(
    {"type", "instructions", "criteria", "labels", "threshold", "head", "preset"}
)
SET_FIELDS = frozenset({"type", "instructions", "criteria", "labels", "threshold"})
OVERFLOW = ("window", "truncate")
PRESETS = ("pii", "halu")
NOUL_OPTIONS = (("no", ""), ("yes", ""))


def text(value: Any) -> str:
    """Model text: a string as it is, anything else as compact JSON (``ub_d3_common.text``)."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _invalid(message: str, reason: str | None = None) -> QuestionError:
    return QuestionError(INVALID_QUESTION, message, reason=reason)


@dataclass(frozen=True)
class Asked:
    """One valid question: the path that answers it and its rendering.

    ``question`` is the engine form shared with the Vela 2.0 layout (for a
    decision, ``over`` is ``user``: the whole state as one part).
    """

    question: Question
    path: str
    router: RouterQuestion | None = None
    route: SpanRoute | None = None


@dataclass
class Read:
    """A request after validation: its typed state (None when it has none), valid questions and errors."""

    raw_state: Any
    state: State | None
    question_ids: list[str]
    asked: list[Asked] = field(default_factory=list)
    errors: dict[str, dict[str, Any]] = field(default_factory=dict)


class Reader:
    """Validates questions against one package's router texts and span routes."""

    def __init__(self, router: Router):
        self.router = router

    def read(self, state: Any, questions: dict[str, Any]) -> Read:
        try:
            typed: State | None = read_state(state)
        except ValueError:
            typed = None
        request = Read(state, typed, list(questions))
        for question_id, question in questions.items():
            try:
                request.asked.append(self.question(question_id, question, typed))
            except QuestionError as exc:
                kind = question.get("type") if isinstance(question, dict) else None
                request.errors[question_id] = question_error(kind, exc)
        taken = set(questions)
        for asked in list(request.asked):
            question = asked.question
            if question.kind == "set" and any(
                f"{question.id}.{name}" in taken for name in question.names
            ):
                request.asked.remove(asked)
                request.errors[question.id] = question_error(
                    "set",
                    _invalid(
                        f"a label answer {question.id}.<label> is another question's ID"
                    ),
                )
        return request

    def question(self, question_id: str, question: Any, state: State | None) -> Asked:
        """One question and the path that answers it; ``QuestionError`` when it is invalid."""
        if not isinstance(question, dict):
            raise _invalid("a question must be an object")
        if question.get("preset") is not None:
            return self._span(question_id, self._preset(question), state)
        kind = question.get("type")
        if kind == "span":
            return self._span(question_id, question, state)
        if kind in ROUTER_TYPES:
            schema = self.router.router_question(question.get("instructions"))
            if schema is not None:
                if schema.type != kind:
                    raise _invalid(
                        f"this router question was trained as a {schema.type} question"
                    )
                return self._router(question_id, question, schema, state)
        if kind in ("choice", "noul"):
            return self._decision(question_id, question)
        if kind == "set":
            raise _invalid(
                "set questions take one of the trained router texts on this model",
                "type_unsupported",
            )
        raise _invalid(
            f"type must be one of {list(QUESTION_TYPES)}", "type_unsupported"
        )

    # -- decisions -----------------------------------------------------------

    def _decision(self, question_id: str, question: dict[str, Any]) -> Asked:
        parsed = read_question(question, extra_fields=FAMILY_FIELDS)
        if question.get("over") is not None:
            raise _invalid("decision questions read the whole state; remove over")
        if question.get("preset") is not None:
            raise _invalid("this model defines only the pii and halu span presets")
        overflow = _overflow(question)
        instructions = question.get("instructions")
        if parsed.kind == "noul":
            if any(value is not None for value in parsed.criteria.values()):
                raise _invalid(
                    "this model reads only a noul question's instructions; remove the false and true descriptions"
                )
            options: tuple[tuple[str, str], ...] = NOUL_OPTIONS
        else:
            options = tuple(
                (name, "" if value is None else text(value))
                for name, value in parsed.criteria.items()
            )
        return Asked(
            Question(
                id=question_id,
                kind=parsed.kind,
                type="choice",
                text="" if instructions is None else text(instructions),
                over="user",
                options=options,
                criteria=parsed.criteria,
                key=question_id,
                truncate=overflow == "truncate",
                require_full_input=require_full_input(question),
            ),
            DECISION,
        )

    # -- router questions ----------------------------------------------------

    def _router(
        self,
        question_id: str,
        question: dict[str, Any],
        schema: RouterQuestion,
        state: State | None,
    ) -> Asked:
        if state is None:
            raise _invalid("router questions need a text state or typed parts")
        kind = question["type"]
        given = question
        if question.get("criteria") is None and not question.get(
            "labels" if kind == "set" else "choices"
        ):
            # The trained options; an undescribed one stays null (rendered as its name alone).
            given = {
                **question,
                "criteria": {
                    name: (text or None) for name, text in schema.options.items()
                },
            }
        if kind == "set":
            unknown = set(given) - SET_FIELDS - FAMILY_FIELDS
            if unknown:
                raise _invalid(f"set questions do not take {sorted(unknown)}")
            criteria = given.get("criteria")
            if given.get("labels") is not None:
                if criteria is not None:
                    raise _invalid("use criteria or labels, not both")
                criteria = listed_options(given["labels"], "labels")
            criteria = named_options(criteria, minimum=1)
        else:
            criteria = read_question(given, extra_fields=FAMILY_FIELDS).criteria
        options = tuple(
            (
                str(name),
                text(value) if value else schema.options.get(str(name), ""),
            )
            for name, value in criteria.items()
        )
        if question.get("over") is not None:
            over, _ = resolve_over(question["over"], state)
        else:
            over = schema.over if schema.over in state.roles else state.roles[-1]
        overflow = _overflow(question)
        return Asked(
            Question(
                id=question_id,
                kind=kind,
                type=kind,
                text=text(question["instructions"]),
                over=over,
                options=options,
                criteria=criteria,
                key=schema.id,
                abstain=schema.abstain,
                threshold=_threshold(question) if kind == "set" else None,
                truncate=overflow == "truncate",
                require_full_input=require_full_input(question),
            ),
            ROUTER,
            router=schema,
        )

    # -- spans -----------------------------------------------------------------

    def _preset(self, question: dict[str, Any]) -> dict[str, Any]:
        name = question["preset"]
        if name not in PRESETS:
            raise _invalid(f"preset must be one of {list(PRESETS)}", "preset_unknown")
        kept = {"over", "threshold", "overflow", "require_full_input"}
        if set(question) - kept - {"preset", "type"}:
            raise _invalid(
                "a preset question takes only preset, type, over, threshold and overflow"
            )
        if question.get("type") not in (None, "span"):
            raise _invalid(f"preset {name} is a span question")
        schema = self.router.span["schemas"][name]
        return {
            "type": "span",
            "instructions": schema["question"],
            "criteria": dict(schema["labels"]),
            "preset": name,
            **{key: question[key] for key in kept if key in question},
        }

    def _span(
        self, question_id: str, question: dict[str, Any], state: State | None
    ) -> Asked:
        if state is None:
            raise _invalid("span questions need a text state or typed parts")
        unknown = set(question) - SPAN_FIELDS - FAMILY_FIELDS
        if unknown:
            raise _invalid(f"span questions do not take {sorted(unknown)}")
        head = question.get("head")
        if head not in (None, "router"):
            raise _invalid("this model has no broad span head")
        criteria = question.get("criteria")
        if question.get("labels") is not None:
            if criteria is not None:
                raise _invalid("use criteria or labels, not both")
            criteria = listed_options(question["labels"], "labels")
        if isinstance(criteria, list):
            criteria = {str(name): None for name in criteria}
        criteria = named_options(criteria, minimum=1)
        instructions = question.get("instructions")
        if instructions is not None:
            instructions = text(content(instructions, "instructions"))
        labels = [
            (name, "" if value is None else text(value))
            for name, value in criteria.items()
        ]
        route = self.router.span_route(instructions, labels)
        if question.get("over") is not None:
            over, span_range = resolve_over(question["over"], state)
            if isinstance(over, tuple):
                raise _invalid("a span question reads one part")
        else:
            roles = state.roles
            over, span_range = ("answer" if "answer" in roles else roles[-1]), None
        overflow = _overflow(question)
        return Asked(
            Question(
                id=question_id,
                kind="span",
                type="span",
                text=route.text,
                over=over,
                options=route.labels,
                criteria=criteria,
                key=question.get("preset") or question_id,
                threshold=_threshold(question),
                span_range=span_range,
                truncate=overflow == "truncate",
                require_full_input=require_full_input(question),
            ),
            SPAN,
            route=route,
        )


def _overflow(question: dict[str, Any]) -> str:
    overflow = question.get("overflow", "window")
    if overflow not in OVERFLOW:
        raise _invalid(f"overflow must be one of {list(OVERFLOW)}")
    return str(overflow)


def _threshold(question: dict[str, Any]) -> float | None:
    threshold = question.get("threshold")
    if threshold is None:
        return None
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0.0 <= threshold <= 1.0
    ):
        raise _invalid("threshold must be a number in [0, 1]")
    return float(threshold)
