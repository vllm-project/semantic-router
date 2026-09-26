"""Owned, deterministic input rendering for Decision model families.

The model repositories are immutable data sources.  They are never imported or
executed by the runtime.  This module preserves the released input semantics in
ordinary Python values so backend implementations can tokenize them without
depending on repository code.

Question identifiers are bookkeeping only and never enter model text.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .contracts import (
    ChoiceQuestion,
    JsonContent,
    NoulQuestion,
    Question,
    ScoreQuestion,
)

if TYPE_CHECKING:
    from .physical_batching import DecisionRow

ChoiceNullDescriptionPolicy = Literal["render_key", "preserve_json_null"]
NoulExplicitNullPolicy = Literal[
    "preserve_json_null",
    "use_default",
    "reject",
]

QWEN_DEFAULT_NO = "The answer to the question is no."
QWEN_DEFAULT_YES = "The answer to the question is yes."
VELA_DEFAULT_NO = "No. The statement or question is not satisfied."
VELA_DEFAULT_YES = "Yes. The statement or question is satisfied."
QWEN_PROMPT_VERSION = "structured-segmented-candidate-endpoints-global-query-v2"


@dataclass(frozen=True, slots=True)
class CandidateInput:
    """One ordered candidate before family-specific tokenization."""

    key: str
    description: JsonContent | None


@dataclass(frozen=True, slots=True)
class ModelInput:
    """One typed Decision row with request identity kept out of rendered text."""

    question_id: str
    type: Literal["noul", "choice", "score"]
    state: JsonContent
    instructions: JsonContent | None
    candidates: tuple[CandidateInput, ...]
    choice_null_description: ChoiceNullDescriptionPolicy


@dataclass(frozen=True, slots=True)
class QwenSegments:
    """Exact separately-tokenized Qwen prompt segments."""

    prefix: str
    options: tuple[str, ...]
    suffix: str

    @property
    def rendered(self) -> str:
        return self.prefix + "".join(self.options) + self.suffix


@dataclass(frozen=True, slots=True)
class VelaTextInput:
    """Exact text spans consumed by the Vela marker collator."""

    question: str
    candidates: tuple[str, ...]
    state: str


def build_model_input(
    *,
    question_id: str,
    state: JsonContent,
    question: Question,
    choice_null_description: ChoiceNullDescriptionPolicy,
    noul_default_false: str,
    noul_default_true: str,
    noul_explicit_null: NoulExplicitNullPolicy,
) -> ModelInput:
    """Convert a validated wire question into the released candidate ordering."""

    if choice_null_description not in {"render_key", "preserve_json_null"}:
        raise ValueError("unsupported Choice null-description policy")
    if noul_explicit_null not in {
        "preserve_json_null",
        "use_default",
        "reject",
    }:
        raise ValueError("unsupported Noul explicit-null policy")
    if any(
        not isinstance(value, str) or not value.strip()
        for value in (noul_default_false, noul_default_true)
    ):
        raise ValueError("Noul defaults must be nonempty strings")

    if isinstance(question, NoulQuestion):
        false_description = _noul_description(
            question,
            "false",
            default=noul_default_false,
            explicit_null=noul_explicit_null,
        )
        true_description = _noul_description(
            question,
            "true",
            default=noul_default_true,
            explicit_null=noul_explicit_null,
        )
        candidates = (
            CandidateInput("false", false_description),
            CandidateInput("true", true_description),
        )
    elif isinstance(question, ChoiceQuestion):
        candidates = tuple(
            CandidateInput(key, _snapshot(description))
            for key, description in question.criteria.items()
        )
    elif isinstance(question, ScoreQuestion):
        candidates = tuple(
            CandidateInput(str(index), _snapshot(description))
            for index, description in enumerate(question.criteria)
        )
    else:  # pragma: no cover - closed Pydantic union defense
        raise TypeError("unsupported Decision question")

    return ModelInput(
        question_id=question_id,
        type=question.type,
        state=_snapshot(state),
        instructions=_snapshot(question.instructions),
        candidates=candidates,
        choice_null_description=choice_null_description,
    )


def build_model_inputs(
    rows: tuple[DecisionRow, ...],
    *,
    choice_null_description: ChoiceNullDescriptionPolicy,
    noul_default_false: str,
    noul_default_true: str,
    noul_explicit_null: NoulExplicitNullPolicy,
) -> tuple[ModelInput, ...]:
    """Snapshot shared request values once while preserving exact row rendering.

    A batch endpoint expands one state across many questions and one question
    across many states. The expanded rows retain their source-object identities,
    so request-local identity maps avoid repeated deep copies and JSON encoding.
    Only immutable state text and owned question snapshots are reused; nothing
    survives this preparation call or is shared between requests.
    """

    states: dict[int, tuple[JsonContent, str]] = {}
    questions: dict[int, tuple[Question, ModelInput]] = {}
    inputs = []
    for row in rows:
        state_id = id(row.state)
        state_entry = states.get(state_id)
        if state_entry is None or state_entry[0] is not row.state:
            state_entry = (row.state, content_text(row.state))
            states[state_id] = state_entry
        state_text = state_entry[1]

        question_id = id(row.question)
        question_entry = questions.get(question_id)
        if question_entry is None or question_entry[0] is not row.question:
            template = build_model_input(
                question_id=row.question_id,
                state=state_text,
                question=row.question,
                choice_null_description=choice_null_description,
                noul_default_false=noul_default_false,
                noul_default_true=noul_default_true,
                noul_explicit_null=noul_explicit_null,
            )
            question_entry = (row.question, template)
            questions[question_id] = question_entry
        template = question_entry[1]
        inputs.append(
            ModelInput(
                question_id=row.question_id,
                type=template.type,
                state=state_text,
                instructions=template.instructions,
                candidates=template.candidates,
                choice_null_description=template.choice_null_description,
            )
        )
    return tuple(inputs)


def _noul_description(
    question: NoulQuestion,
    field: Literal["false", "true"],
    *,
    default: str,
    explicit_null: NoulExplicitNullPolicy,
) -> JsonContent | None:
    criteria = question.criteria
    if criteria is None or field not in criteria.model_fields_set:
        return default
    value = getattr(criteria, field)
    if value is None and explicit_null == "use_default":
        return default
    if value is None and explicit_null == "reject":
        raise ValueError(f"Vela Noul criterion {field!r} must not be null")
    return _snapshot(value)


def _snapshot(value):
    """Detach validated JSON content from caller-owned mutable containers."""

    return deepcopy(value)


def qwen_segments(row: ModelInput) -> QwenSegments:
    """Render the exact pointer-v2 segments used by Qwen3.5 Decision models."""

    prefix = (
        f"Context:\n{content_text(row.state)}\n\n"
        f"Task type: {row.type}\n"
        f"Question:\n{content_text(row.instructions)}\n"
        "Options:"
    )
    options = tuple(
        "\n<option>\n"
        + canonical_json(
            {
                "key": candidate.key,
                "description": (
                    candidate.key
                    if candidate.description is None
                    and row.type == "choice"
                    and row.choice_null_description == "render_key"
                    else candidate.description
                ),
            }
        )
        + "\n</option>"
        for candidate in row.candidates
    )
    suffix = (
        "\n\nSelect the single option best supported by the context and "
        "instructions.\nDecision:"
    )
    return QwenSegments(prefix=prefix, options=options, suffix=suffix)


def vela_text_input(row: ModelInput) -> VelaTextInput:
    """Render spans used by the released Vela candidate-marker tokenizer."""

    candidates = []
    for index, candidate in enumerate(row.candidates):
        description = candidate.description
        if description is None:
            if row.type != "choice" or row.choice_null_description != "render_key":
                raise ValueError("Vela candidate descriptions must not be null")
            rendered = candidate.key
        else:
            rendered = content_text(description)
            if row.type == "choice":
                rendered = f"{candidate.key}: {rendered}"
        if row.type == "score":
            rendered = f"level {index}: {rendered}"
        candidates.append(rendered)
    return VelaTextInput(
        question=f"{row.type} question: {content_text(row.instructions)}",
        candidates=tuple(candidates),
        state=content_text(row.state),
    )


def content_text(value: JsonContent | None) -> str:
    """Preserve strings and canonically serialize other JSON content."""

    return value if isinstance(value, str) else canonical_json(value)


def canonical_json(value: object) -> str:
    """Serialize model-visible JSON with the released stable representation."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
