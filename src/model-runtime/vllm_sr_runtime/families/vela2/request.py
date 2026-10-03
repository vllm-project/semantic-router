"""A ``/v1/decisions`` request read the way the Vela 2.0 packages read System One.

The state becomes typed parts: a string is the user part; a JSON object maps
``request`` / ``user`` / ``prompt`` to the user part, ``answer`` /
``response`` to the answer part and every other key to the context part (a
part fed by several keys, or by an unknown key, is ``key:\\n<value>`` blocks
joined by a blank line); an array is canonical JSON in the user part.
Questions become the engine's typed questions: Noul is a two-option choice
(``no`` / ``yes``), Choice and Noul show the abstain option, Score levels are
the options in order, Set and Span take ``{label: description}``. ``over``
names a state key, a part or a list of them. A question that does not
validate gets ``invalid_question`` and never affects the others.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ...errors import INVALID_QUESTION, QuestionError
from ...systemone import canonical, json_payload
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
QUESTION_TYPES = ("noul", "choice", "score", "set", "span")
NOUL_DEFAULT_NO = "No. The statement or question is not satisfied."
NOUL_DEFAULT_YES = "Yes. The statement or question is satisfied."
MAX_OPTIONS = 255
MIN_LEVELS, MAX_LEVELS = 2, 10
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
    labels). ``key`` names the calibration entry (the question ID, or the
    preset). ``span_range`` clips a span to one state key of a shared part.
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


def _nonblank(value: Any) -> bool:
    return not (isinstance(value, str) and not value.strip())


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


def _labelled(kind: str, criteria: Any) -> tuple[tuple[str, str], ...]:
    low = 2 if kind == "choice" else 1
    if not isinstance(criteria, dict) or not low <= len(criteria) <= MAX_OPTIONS:
        raise _invalid(
            f"{kind} criteria must be an object with {low} to {MAX_OPTIONS} entries"
        )
    options = []
    for key, value in criteria.items():
        if not isinstance(key, str) or not key.strip() or not _nonblank(value):
            raise _invalid(
                "option names and descriptions must not be empty or whitespace"
            )
        if value is not None and not json_payload(value):
            raise _invalid("descriptions must be text or JSON values")
        options.append((key, "" if value is None else content_text(value)))
    return tuple(options)


def _choices(choices: Any) -> dict[str, Any]:
    if not isinstance(choices, list):
        raise _invalid("choices must be a list of {key, description}")
    criteria: dict[str, Any] = {}
    for choice in choices:
        if (
            not isinstance(choice, dict)
            or "key" not in choice
            or set(choice) - {"key", "description"}
        ):
            raise _invalid("each choice is {key, description}")
        if not isinstance(choice["key"], str) or choice["key"] in criteria:
            raise _invalid("choice keys must be unique strings")
        criteria[choice["key"]] = choice.get("description")
    return criteria


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
                plan.errors[question_id] = {"type": kind, "error": exc.code}
        taken = set(questions)
        for question in list(plan.questions):
            if question.kind == "set" and any(
                f"{question.id}.{name}" in taken for name in question.names
            ):
                plan.questions.remove(question)
                plan.errors[question.id] = {"type": "set", "error": INVALID_QUESTION}
        return plan

    def question(self, question_id: str, question: Any, state: State) -> Question:
        """One question in the engine's form; ``QuestionError`` when it is invalid."""
        if not question_id.strip():
            raise _invalid("question IDs must not be empty or whitespace")
        if not isinstance(question, dict):
            raise _invalid("a question must be an object")
        named = None
        if question.get("preset") is not None:
            question, named = self._expand_preset(question)
        kind = question.get("type")
        if kind not in QUESTION_TYPES:
            raise _invalid(f"type must be one of {list(QUESTION_TYPES)}")
        allowed = {"type", "instructions", "criteria", "over", "preset"}
        allowed |= {"choices"} if kind in ("choice", "noul") else set()
        allowed |= {"levels"} if kind == "score" else set()
        allowed |= {"threshold"} if kind in ("set", "span") else set()
        allowed |= {"head"} if kind == "span" else set()
        extra = sorted(set(question) - allowed)
        if extra:
            raise _invalid(f"{kind} questions do not take {extra}")
        instructions = question.get("instructions")
        if not json_payload(instructions, require_nonempty_text=True) or not _nonblank(
            instructions
        ):
            raise _invalid("instructions must be non-empty text or JSON")
        if "over" in question and question["over"] is not None:
            over, span_range = resolve_over(question["over"], state)
        else:
            over, span_range = _default_over(kind, state), None
        if kind != "span":
            span_range = None
        elif isinstance(over, tuple):
            raise _invalid("a span question reads one part")
        fields: dict[str, Any] = {
            "id": question_id,
            "kind": kind,
            "type": "choice" if kind == "noul" else kind,
            "text": content_text(instructions),
            "over": over,
            "key": question.get("preset") or question_id,
            "span_range": span_range,
        }
        criteria = question.get("criteria")
        if kind in ("choice", "noul") and "choices" in question:
            if criteria is not None:
                raise _invalid("use criteria or choices, not both")
            criteria = _choices(question["choices"])
        if kind == "score" and "levels" in question:
            if criteria is not None:
                raise _invalid("use criteria or levels, not both")
            criteria = question["levels"]
        if kind == "noul":
            criteria = self._noul(criteria)
            fields["options"] = (
                (
                    "no",
                    (
                        NOUL_DEFAULT_NO
                        if criteria.get("false") is None
                        else content_text(criteria["false"])
                    ),
                ),
                (
                    "yes",
                    (
                        NOUL_DEFAULT_YES
                        if criteria.get("true") is None
                        else content_text(criteria["true"])
                    ),
                ),
            )
            fields["abstain"] = True
        elif kind == "score":
            fields["options"] = self._levels(criteria, named)
        else:
            fields["options"] = _labelled(kind, criteria)
            fields["abstain"] = kind == "choice"
        fields["criteria"] = criteria
        if question.get("threshold") is not None:
            threshold = question["threshold"]
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not 0.0 <= threshold <= 1.0
            ):
                raise _invalid("threshold must be a number in [0, 1]")
            fields["threshold"] = float(threshold)
        if question.get("head") is not None:
            if question["head"] not in SPAN_HEADS:
                raise _invalid(f"head must be one of {list(SPAN_HEADS)}")
            if question["head"] == "broad" and not self.broad_head:
                raise _invalid("this model has no broad span head")
            fields["head"] = question["head"]
        return Question(**fields)

    def _expand_preset(
        self, question: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str] | None]:
        """A preset question with the package's trained schema, and its level names (Score presets).

        ``over`` and ``threshold`` stay the caller's. The relevance preset
        renders its levels as named options with descriptions, as the
        packages' ``score_relevance`` does.
        """
        name = question["preset"]
        if name not in self.presets:
            raise _invalid(f"preset must be one of {list(self.presets)}")
        if set(question) - {"preset", "type", "over", "threshold"}:
            raise _invalid(
                "a preset question takes only preset, type, over and threshold"
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
        kept = {key: question[key] for key in ("over", "threshold") if key in question}
        return {**expanded, "preset": name, **kept}, named

    @staticmethod
    def _noul(criteria: Any) -> dict[str, Any]:
        if criteria is not None and (
            not isinstance(criteria, dict) or set(criteria) - {"true", "false"}
        ):
            raise _invalid("noul criteria is an object with optional true / false")
        criteria = criteria or {}
        for key in ("true", "false"):
            value = criteria.get(key)
            if not _nonblank(value) or (value is not None and not json_payload(value)):
                raise _invalid("Noul criteria must be non-empty text or JSON")
        return criteria

    @staticmethod
    def _levels(criteria: Any, named: list[str] | None) -> tuple[tuple[str, str], ...]:
        if (
            not isinstance(criteria, list)
            or not MIN_LEVELS <= len(criteria) <= MAX_LEVELS
        ):
            raise _invalid(
                f"score criteria must be a list of {MIN_LEVELS} to {MAX_LEVELS} levels"
            )
        names = []
        for value in criteria:
            if value is None or not _nonblank(value) or not json_payload(value):
                raise _invalid("score levels must be non-empty text or JSON")
            names.append(content_text(value))
        if len(set(names)) != len(names):
            raise _invalid("score levels must be distinct")
        if named is not None:
            return tuple(zip(named, names, strict=True))
        return tuple((name, "") for name in names)
