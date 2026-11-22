"""A ``/v1/decisions`` request read the way a Vela 3.1 pico package reads System One (render31).

Every question is rendered exactly as given (``families.render31``): options
and labels with a description read ``name: description``, without one
``name``, and nothing is filled in from a trained schema. A router question
asked with no criteria takes its ``router_questions.json`` options as written;
a span question with no instructions takes its trained set's question.

- Choice and Noul questions that are not router questions are decisions: the
  question as given over the whole state (``render31.decision``); a Noul is
  ``{"no": false text, "yes": true text}``, so it may describe either side.
- A Choice or Set question whose instructions are a router text is a router
  question in its trained type (another type is invalid), with the trained
  abstain option; it reads the request's ``over``, else its trained part,
  else the last part.
- Span questions and the ``pii`` / ``halu`` presets read one part: the
  request's ``over``, else the answer, else the last part.

The state's parts are System One's typed parts (``vela2.request.typed_state``):
a string is the user part as it is, an object's keys take their roles. A
request goes into one sequence when the package's ``system_one`` would try its
one pass (``one_pass``): some question is a span, a Set or a router text, and
every question is a span, a router text, a Choice or a Noul. Score questions,
Set questions that are not router questions and ``head: broad`` spans are
invalid on this model. ``over``, ``threshold``, ``overflow`` and
``require_full_input`` follow the System One contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...errors import QuestionError, question_error
from ...systemone import (
    content,
    listed_options,
    named_options,
    read_question,
    require_full_input,
)
from .. import render31
from ..vela2.request import Question, State, resolve_over, typed_state
from ..vela3_shared import SPAN_HEADS, expand_preset, invalid, text, threshold_of
from .routing import RouterQuestion, Routes, SpanRoute

DECISION, ROUTER, SPAN = "decision", "router", "span"
QUESTION_TYPES = ("choice", "noul", "set", "span")
ROUTER_TYPES = ("choice", "set")
FAMILY_FIELDS = frozenset({"over", "preset", "overflow", "require_full_input"})
SPAN_FIELDS = frozenset(
    {"type", "instructions", "criteria", "labels", "threshold", "head", "preset"}
)
SET_FIELDS = frozenset({"type", "instructions", "criteria", "labels", "threshold"})
OVERFLOW = ("window", "truncate")


@dataclass(frozen=True)
class Asked:
    """One valid question: the path that answers it and its rendering in the engine's form."""

    question: Question
    path: str
    router: RouterQuestion | None = None
    route: SpanRoute | None = None


@dataclass
class Read:
    """A request after validation: its typed state (None when it has none), valid questions, errors and its pass."""

    raw_state: Any
    state: State | None
    question_ids: list[str]
    asked: list[Asked] = field(default_factory=list)
    errors: dict[str, dict[str, Any]] = field(default_factory=dict)
    one_pass: bool = False


class Reader:
    """Validates and renders questions against one package's router texts and span rules."""

    def __init__(self, routes: Routes):
        self.routes = routes

    def read(self, state: Any, questions: dict[str, Any]) -> Read:
        try:
            typed: State | None = typed_state(state)
        except ValueError:
            typed = None
        request = Read(state, typed, list(questions), one_pass=self.one_pass(questions))
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
                    invalid(
                        f"a label answer {question.id}.<label> is another question's ID"
                    ),
                )
        return request

    def one_pass(self, questions: dict[str, Any]) -> bool:
        """Whether the package's ``system_one`` tries its one pass on these questions as given.

        It tries when some question is a span, a Set or a router text, and
        ``render_call`` gives up when a question is anything but a span, a
        router text, a Choice or a Noul. A preset is a span question.
        """

        def kind(question: Any) -> str | None:
            if not isinstance(question, dict):
                return None
            if question.get("preset") is not None or question.get("type") == "span":
                return SPAN
            if self.routes.router_question(question.get("instructions")) is not None:
                return ROUTER
            value = question.get("type")
            return str(value) if isinstance(value, str) else None

        kinds = [kind(question) for question in questions.values()]
        tried = any(k in (SPAN, ROUTER, "set") for k in kinds)
        return tried and all(k in (SPAN, ROUTER, "choice", "noul") for k in kinds)

    def question(self, question_id: str, question: Any, state: State | None) -> Asked:
        """One question and the path that answers it; ``QuestionError`` when it is invalid."""
        if not isinstance(question, dict):
            raise invalid("a question must be an object")
        if question.get("preset") is not None:
            return self._span(question_id, self._preset(question), state)
        kind = question.get("type")
        if kind == "span":
            return self._span(question_id, question, state)
        if kind in ROUTER_TYPES:
            schema = self.routes.router_question(question.get("instructions"))
            if schema is not None:
                if schema.type != kind:
                    raise invalid(
                        f"this router question was trained as a {schema.type} question"
                    )
                return self._router(question_id, question, schema, state)
        if kind in ("choice", "noul"):
            return self._decision(question_id, question)
        if kind == "set":
            raise invalid(
                "set questions take one of the trained router texts on this model",
                "type_unsupported",
            )
        raise invalid(f"type must be one of {list(QUESTION_TYPES)}", "type_unsupported")

    # -- decisions -----------------------------------------------------------

    def _decision(self, question_id: str, question: dict[str, Any]) -> Asked:
        parsed = read_question(question, extra_fields=FAMILY_FIELDS)
        if question.get("over") is not None:
            raise invalid("decision questions read the whole state; remove over")
        overflow = _overflow(question)
        rendered, options = render31.decision(
            {
                "type": parsed.kind,
                "instructions": question.get("instructions"),
                "criteria": parsed.criteria,
            }
        )
        return Asked(
            Question(
                id=question_id,
                kind=parsed.kind,
                type="choice",
                text=rendered,
                over="user",
                options=tuple(options.items()),
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
            raise invalid("router questions need a text state or typed parts")
        kind = question["type"]
        given = question.get("criteria") is not None or bool(
            question.get("labels" if kind == "set" else "choices")
        )
        criteria: Any = None
        if kind == "set":
            unknown = set(question) - SET_FIELDS - FAMILY_FIELDS
            if unknown:
                raise invalid(f"set questions do not take {sorted(unknown)}")
            if given:
                criteria = question.get("criteria")
                if question.get("labels") is not None:
                    if criteria is not None:
                        raise invalid("use criteria or labels, not both")
                    criteria = listed_options(question["labels"], "labels")
                criteria = named_options(criteria, minimum=1)
        elif given:
            criteria = read_question(question, extra_fields=FAMILY_FIELDS).criteria
        else:
            # Validate the rest of the question against the table's option names.
            read_question(
                {
                    **{k: v for k, v in question.items() if k != "choices"},
                    "criteria": dict.fromkeys(schema.entry["options"]),
                },
                extra_fields=FAMILY_FIELDS,
            )
        # As given, or the table's options as written when the request gives none.
        options = render31.options(
            {"type": schema.type, "criteria": criteria}, builtin=schema.entry
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
                text=render31.question_text(question, schema.entry),
                over=over,
                options=tuple(options.items()),
                criteria=criteria,
                key=schema.id,
                abstain=schema.abstain and kind == "choice",
                threshold=threshold_of(question) if kind == "set" else None,
                truncate=overflow == "truncate",
                require_full_input=require_full_input(question),
            ),
            ROUTER,
            router=schema,
        )

    # -- spans -----------------------------------------------------------------

    def _preset(self, question: dict[str, Any]) -> dict[str, Any]:
        expanded = expand_preset(question, self.routes.span["schemas"])
        return {**expanded, "preset": question["preset"]}

    def _span(
        self, question_id: str, question: dict[str, Any], state: State | None
    ) -> Asked:
        if state is None:
            raise invalid("span questions need a text state or typed parts")
        unknown = set(question) - SPAN_FIELDS - FAMILY_FIELDS
        if unknown:
            raise invalid(f"span questions do not take {sorted(unknown)}")
        if question.get("head") not in (None, *SPAN_HEADS):
            raise invalid("this model has no broad span head")
        criteria = question.get("criteria")
        if question.get("labels") is not None:
            if criteria is not None:
                raise invalid("use criteria or labels, not both")
            criteria = listed_options(question["labels"], "labels")
        if isinstance(criteria, list):
            criteria = {str(name): None for name in criteria}
        criteria = named_options(criteria, minimum=1)
        instructions = question.get("instructions")
        if instructions is not None:
            instructions = text(content(instructions, "instructions"))
        route = self.routes.span_route(
            {"instructions": instructions, "criteria": criteria}
        )
        if question.get("over") is not None:
            over, span_range = resolve_over(question["over"], state)
            if isinstance(over, tuple):
                raise invalid("a span question reads one part")
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
                threshold=threshold_of(question),
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
        raise invalid(f"overflow must be one of {list(OVERFLOW)}")
    return str(overflow)
