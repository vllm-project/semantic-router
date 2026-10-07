"""A ``/v1/decisions`` request read the way the Vela 2.0 packages read System One.

The state becomes typed parts: a string is the user part; a JSON object maps
``request`` / ``user`` / ``prompt`` to the user part, ``answer`` /
``response`` to the answer part and every other key to the context part (a
part fed by several keys, or by an unknown key, is ``key:\\n<value>`` blocks
joined by a blank line); an array is canonical JSON in the user part.
Questions follow the shared System One rules (``systemone.read_question``)
and become the engine's typed questions: Noul is a two-option choice (``no``
/ ``yes``), Choice and Noul show the abstain option, Score levels are the
options in order. Vela 2.0 adds Set and Span questions (``{label:
description}`` with an optional ``threshold``; a Span may name its
``head``), the package's presets, ``over``, which names a state key, a part
or a list of them, and ``overflow``: ``truncate`` reads a part longer than
one input only as far as its first tokens. A question that does not validate
gets ``invalid_question`` and never affects the others.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ...errors import INVALID_QUESTION, QuestionError, question_error
from ...systemone import QUESTION_TYPES as SYSTEM_ONE_TYPES
from ...systemone import (
    canonical,
    content,
    json_payload,
    listed_options,
    named_options,
    read_question,
)
from .calibration import SPAN_HEADS, Calibration

ROLES = ("user", "context", "answer")
STATE_KEY_ROLES = {
    "request": "user",
    "user": "user",
    "prompt": "user",
    "source": "context",
    "context": "context",
    "document": "context",
    "tools": "context",
    "previous_answer": "context",
    "answer": "answer",
    "response": "answer",
}
LABELLED_TYPES = ("set", "span")
QUESTION_TYPES = (*SYSTEM_ONE_TYPES, *LABELLED_TYPES)
FAMILY_FIELDS = frozenset({"over", "preset", "overflow"})
# How a question reads a part longer than one input: whole, in windows up to
# the request's scan budget (``window``, the default), or its first tokens.
OVERFLOW = ("window", "truncate")
LABELLED_FIELDS = {
    "set": frozenset({"type", "instructions", "criteria", "labels", "threshold"}),
    "span": frozenset(
        {"type", "instructions", "criteria", "labels", "threshold", "head"}
    ),
}
NOUL_DEFAULT_NO = "No. The statement or question is not satisfied."
NOUL_DEFAULT_YES = "Yes. The statement or question is satisfied."
NOUL_OPTIONS = (("no", "false", NOUL_DEFAULT_NO), ("yes", "true", NOUL_DEFAULT_YES))
PRESETS = ("pii", "halu", "relevance")


def content_text(value: Any) -> str:
    """JSON content as model text: strings as they are, anything else as canonical JSON."""
    return value if isinstance(value, str) else canonical(value)


@dataclass(frozen=True)
class Part:
    role: str
    text: str


@dataclass(frozen=True)
class State:
    """Typed parts in user / context / answer order, and each state key's code-point range in its part."""

    parts: tuple[Part, ...]
    fields: dict[str, tuple[str, int, int]]

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(part.role for part in self.parts)

    def text(self, role: str) -> str:
        return next(part.text for part in self.parts if part.role == role)


@dataclass(frozen=True)
class Question:
    """One validated question in the engine's form.

    ``type`` is the engine type (Noul is a ``choice``); ``kind`` the request
    type. ``options`` are ``(name, description)`` pairs in order (Span:
    labels; Score names may repeat, so answers read levels by position).
    ``key`` names the calibration entry (the question ID, or the preset).
    ``span_range`` clips a span to one state key of a shared part.
    """

    id: str
    kind: str
    type: str
    text: str
    over: str | tuple[str, ...]
    options: tuple[tuple[str, str], ...]
    criteria: Any
    key: str
    abstain: bool = False
    threshold: float | None = None
    head: str | None = None
    span_range: tuple[int, int] | None = None
    truncate: bool = False

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.options]

    @property
    def roles(self) -> tuple[str, ...]:
        return self.over if isinstance(self.over, tuple) else (self.over,)


@dataclass
class Plan:
    """A request after validation: its state, valid questions in order and per-question errors."""

    state: State
    question_ids: list[str]
    questions: list[Question] = field(default_factory=list)
    errors: dict[str, dict[str, Any]] = field(default_factory=dict)


def read_state(state: Any) -> State:
    """The typed parts of a System One state; ValueError when the state is empty or malformed."""
    if isinstance(state, str) and state.strip().startswith("{"):
        try:
            decoded = json.loads(state.strip())
        except ValueError:
            decoded = None
        if isinstance(decoded, dict):
            state = decoded
    if isinstance(state, str):
        if not state.strip():
            raise ValueError("state must not be empty or whitespace")
        return State((Part("user", state),), {"request": ("user", 0, len(state))})
    if isinstance(state, list):
        if not json_payload(state):
            raise ValueError("state must hold JSON values")
        text = canonical(state)
        return State((Part("user", text),), {"request": ("user", 0, len(text))})
    if not isinstance(state, dict) or not state or not json_payload(state):
        raise ValueError("state must be a non-empty string, JSON object or JSON array")
    by_role: dict[str, list[tuple[str, str, bool]]] = defaultdict(list)
    for key, value in state.items():
        known = str(key).lower() in STATE_KEY_ROLES
        role = STATE_KEY_ROLES.get(str(key).lower(), "context")
        by_role[role].append((str(key), content_text(value), known))
    parts, fields = [], {}
    for role in ROLES:
        items = by_role.get(role)
        if not items:
            continue
        if len(items) == 1 and items[0][2]:
            key, text, _ = items[0]
            fields[key] = (role, 0, len(text))
        else:
            text = ""
            for key, value, _ in items:
                if text:
                    text += "\n\n"
                head = f"{key}:\n"
                fields[key] = (
                    role,
                    len(text) + len(head),
                    len(text) + len(head) + len(value),
                )
                text += head + value
        if not text.strip():
            raise ValueError(f"the {role} part of the state is empty")
        parts.append(Part(role, text))
    return State(tuple(parts), fields)


def _invalid(message: str) -> QuestionError:
    return QuestionError(INVALID_QUESTION, message)


def resolve_over(
    over: Any, state: State
) -> tuple[str | tuple[str, ...], tuple[int, int] | None]:
    """``over`` as the part (or parts, in sequence order) it reads, and a single key's range."""
    items = over if isinstance(over, list) else [over]
    roles, span_range = [], None
    for item in items:
        if not isinstance(item, str):
            raise _invalid("over must be a state key, a part name or a list of them")
        if item in state.fields:
            role, start, end = state.fields[item]
            if not isinstance(over, list):
                span_range = (start, end)
        elif item in state.roles:
            role = item
        elif item in STATE_KEY_ROLES and STATE_KEY_ROLES[item] in state.roles:
            role = STATE_KEY_ROLES[item]
        else:
            raise _invalid(f"{item!r} is not a field of the state")
        if role not in roles:
            roles.append(role)
    if not roles:
        raise _invalid("over must not be empty")
    ordered = tuple(role for role in state.roles if role in roles)
    return (ordered[0] if len(ordered) == 1 else ordered), span_range


def _default_over(kind: str, state: State) -> str | tuple[str, ...]:
    roles = state.roles
    if kind == "span":
        return (
            "answer" if "answer" in roles else ("user" if "user" in roles else roles[0])
        )
    return roles[0] if len(roles) == 1 else roles


def _labelled(question: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """A Set or Span question's instructions and ``{label: description}`` criteria (1 to 255 labels).

    ``labels`` is the ordered ``[{key, description}]`` form the Router config
    uses, an alternative to the criteria object.
    """
    kind = question["type"]
    unknown = set(question) - LABELLED_FIELDS[kind] - FAMILY_FIELDS
    if unknown:
        raise _invalid(f"{kind} questions do not take {sorted(unknown)}")
    criteria = question.get("criteria")
    if question.get("labels") is not None:
        if criteria is not None:
            raise _invalid("use criteria or labels, not both")
        criteria = listed_options(question["labels"], "labels")
    return (
        content(question.get("instructions"), "instructions"),
        named_options(criteria, minimum=1),
    )


def _text(value: Any, default: str = "") -> str:
    """A description as model text; null takes ``default``."""
    return default if value is None else content_text(value)


def _described(criteria: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Named options in order, a null description as empty text."""
    return tuple((name, _text(value)) for name, value in criteria.items())


class QuestionReader:
    """Validates questions against one package's presets and span heads."""

    def __init__(self, calibration: Calibration, broad_head: bool):
        self.calibration = calibration
        self.broad_head = broad_head
        self.presets = tuple(name for name in PRESETS if calibration.schema(name))

    def read(self, state: Any, questions: dict[str, Any]) -> Plan:
        """The request's state and questions; an invalid question becomes its own error."""
        plan = Plan(read_state(state), list(questions))
        for question_id, question in questions.items():
            try:
                plan.questions.append(self.question(question_id, question, plan.state))
            except QuestionError as exc:
                kind = question.get("type") if isinstance(question, dict) else None
                plan.errors[question_id] = question_error(kind, exc)
        taken = set(questions)
        for question in list(plan.questions):
            if question.kind == "set" and any(
                f"{question.id}.{name}" in taken for name in question.names
            ):
                plan.questions.remove(question)
                plan.errors[question.id] = question_error(
                    "set",
                    QuestionError(
                        INVALID_QUESTION,
                        f"a label answer {question.id}.<label> is another question's ID",
                    ),
                )
        return plan

    def question(self, question_id: str, question: Any, state: State) -> Question:
        """One question in the engine's form; ``QuestionError`` when it is invalid.

        System One types read through ``systemone.read_question``; Set and
        Span, presets and ``over`` are the family's own.
        """
        named = None
        if isinstance(question, dict) and question.get("preset") is not None:
            question, named = self._expand_preset(question)
        if isinstance(question, dict) and question.get("type") in LABELLED_TYPES:
            kind = question["type"]
            instructions, criteria = _labelled(question)
            options = _described(criteria)
        else:
            parsed = read_question(question, extra_fields=FAMILY_FIELDS)
            kind, instructions, criteria = (
                parsed.kind,
                parsed.instructions,
                parsed.criteria,
            )
            if kind == "noul":
                options = tuple(
                    (name, _text(criteria.get(key), default))
                    for name, key, default in NOUL_OPTIONS
                )
            elif kind == "score":
                levels = [content_text(level) for level in criteria]
                descriptions = levels if named else [""] * len(levels)
                options = tuple(zip(named or levels, descriptions, strict=True))
            else:
                options = _described(criteria)
        if question.get("over") is not None:
            over, span_range = resolve_over(question["over"], state)
        else:
            over, span_range = _default_over(kind, state), None
        if kind != "span":
            span_range = None
        elif isinstance(over, tuple):
            raise _invalid("a span question reads one part")
        threshold = question.get("threshold")
        if threshold is not None and (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not 0.0 <= threshold <= 1.0
        ):
            raise _invalid("threshold must be a number in [0, 1]")
        overflow = question.get("overflow", "window")
        if overflow not in OVERFLOW:
            raise _invalid(f"overflow must be one of {list(OVERFLOW)}")
        head = question.get("head")
        if head is not None and head not in SPAN_HEADS:
            raise _invalid(f"head must be one of {list(SPAN_HEADS)}")
        if head == "broad" and not self.broad_head:
            raise _invalid("this model has no broad span head")
        return Question(
            id=question_id,
            kind=kind,
            type="choice" if kind == "noul" else kind,
            text=content_text(instructions),
            over=over,
            options=options,
            criteria=criteria,
            key=question.get("preset") or question_id,
            abstain=kind in ("choice", "noul"),
            threshold=None if threshold is None else float(threshold),
            head=head,
            span_range=span_range,
            truncate=overflow == "truncate",
        )

    def _expand_preset(
        self, question: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str] | None]:
        """A preset question with the package's trained schema, and its level names (Score presets).

        ``over``, ``threshold`` and ``overflow`` stay the caller's. The relevance preset
        renders its levels as named options with descriptions, as the
        packages' ``score_relevance`` does.
        """
        name = question["preset"]
        if name not in self.presets:
            raise _invalid(f"preset must be one of {list(self.presets)}")
        if set(question) - {"preset", "type", "over", "threshold", "overflow"}:
            raise _invalid(
                "a preset question takes only preset, type, over, threshold and overflow"
            )
        schema = self.calibration.schema(name) or {}
        named = None
        if name == "relevance":
            expanded = {
                "type": "score",
                "instructions": schema["text"],
                "criteria": list(schema["levels"].values()),
            }
            named = list(schema["levels"])
        else:
            expanded = {
                "type": "span",
                "instructions": schema["text"],
                "criteria": dict(schema["labels"]),
            }
        if question.get("type") not in (None, expanded["type"]):
            raise _invalid(f"preset {name} is a {expanded['type']} question")
        kept: dict[str, Any] = {
            key: question[key]
            for key in ("over", "threshold", "overflow")
            if key in question
        }
        return {**expanded, "preset": name, **kept}, named
