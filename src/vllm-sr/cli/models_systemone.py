"""Systemone signals: typed questions answered by a model_runtime or systemone deployment."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models_predicates import NumericPredicate

MAX_SYSTEMONE_TIMEOUT_MS = 60_000
MIN_SYSTEMONE_CHOICES = 2
MAX_SYSTEMONE_CHOICES = 255
MIN_SYSTEMONE_LEVELS = 2
MAX_SYSTEMONE_LEVELS = 10
MIN_SYSTEMONE_LABELS = 1
MAX_SYSTEMONE_LABELS = 255
MAX_SYSTEMONE_PREVIOUS_TURNS = 32
NOUL_CHOICE_KEYS = frozenset({"false", "true"})
LABELLED_QUESTION_TYPES = frozenset({"set", "span"})
SYSTEMONE_QUESTION_TYPES = ("choice", "noul", "score", "set", "span")
# Question types whose conditions may name one of the question's options.
OPTION_QUESTION_TYPES = frozenset({"choice", "set", "span"})


def _trimmed(value: str) -> bool:
    return bool(value.strip()) and value == value.strip()


class SystemOneOption(BaseModel):
    """One option of a choice question, a noul description, or a set or span label."""

    model_config = ConfigDict(extra="forbid")

    key: str
    description: str | None = None


class SystemOneQuestion(BaseModel):
    """A System One question: choice, noul or score; set and span for models that declare them."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["choice", "noul", "score", "set", "span"]
    instructions: str
    choices: list[SystemOneOption] = Field(default_factory=list)
    levels: list[str] = Field(default_factory=list)
    labels: list[SystemOneOption] = Field(default_factory=list)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    head: Literal["router", "broad"] | None = None

    @model_validator(mode="after")
    def validate_shape(self):
        if not self.instructions.strip():
            raise ValueError("instructions are required")
        if self.head is not None and self.type != "span":
            raise ValueError("head applies only to span questions")
        if self.type in LABELLED_QUESTION_TYPES:
            return self._validate_labels()
        if self.labels:
            raise ValueError("labels apply only to set and span questions")
        if self.threshold is not None:
            raise ValueError("threshold applies only to set and span questions")
        if self.type != "score" and self.levels:
            raise ValueError("levels apply only to score questions")
        if self.type == "score":
            if self.choices:
                raise ValueError("a score question takes ordered levels, not choices")
            if not MIN_SYSTEMONE_LEVELS <= len(self.levels) <= MAX_SYSTEMONE_LEVELS:
                raise ValueError(
                    f"a score question needs {MIN_SYSTEMONE_LEVELS}..{MAX_SYSTEMONE_LEVELS} levels"
                )
            if any(not level.strip() for level in self.levels):
                raise ValueError("every level needs a description")
            return self
        if self.type == "choice" and not (
            MIN_SYSTEMONE_CHOICES <= len(self.choices) <= MAX_SYSTEMONE_CHOICES
        ):
            raise ValueError(
                f"a choice question needs {MIN_SYSTEMONE_CHOICES}..{MAX_SYSTEMONE_CHOICES} choices"
            )
        keys = [choice.key for choice in self.choices]
        if any(not _trimmed(key) for key in keys):
            raise ValueError("choice keys must be nonempty and trimmed")
        if self.type == "noul" and not set(keys) <= NOUL_CHOICE_KEYS:
            raise ValueError("a noul question accepts only the false and true choices")
        if len(set(keys)) != len(keys):
            raise ValueError("choice keys must be unique")
        return self

    def _validate_labels(self):
        if self.choices or self.levels:
            raise ValueError(
                f"a {self.type} question takes labels, not choices or levels"
            )
        if not MIN_SYSTEMONE_LABELS <= len(self.labels) <= MAX_SYSTEMONE_LABELS:
            raise ValueError(
                f"a {self.type} question needs "
                f"{MIN_SYSTEMONE_LABELS}..{MAX_SYSTEMONE_LABELS} labels"
            )
        keys = [label.key for label in self.labels]
        if any(not _trimmed(key) for key in keys):
            raise ValueError("label keys must be nonempty and trimmed")
        if len(set(keys)) != len(keys):
            raise ValueError("label keys must be unique")
        return self


class SystemOneInput(BaseModel):
    """What a rule reads besides the current user turn."""

    model_config = ConfigDict(extra="forbid")

    history: bool = False
    previous_turns: int = Field(default=0, ge=0, le=MAX_SYSTEMONE_PREVIOUS_TURNS)
    # The model's answer and its grounding context: response stage only.
    answer: bool = False
    context: bool = False
    tool_results: bool = False
    # Media parts; the answering deployment must declare the input modality.
    image: bool = False
    audio: bool = False
    video: bool = False
    # False leaves the current user turn out: the rule reads only the earlier
    # user turns or the tool results it names, and is not asked before they
    # hold text.
    request: bool | None = None


def turnless_input_error(reads, stage: str | None) -> str | None:
    """What is wrong with an input that leaves the current user turn out."""
    if reads.request is not False:
        return None
    if (stage or "request") != "request":
        return "input.request: false applies to a request-stage rule"
    if not reads.history and not reads.tool_results:
        return (
            "input.request: false leaves the current user turn out, so the rule "
            "reads input.history or input.tool_results"
        )
    media = (getattr(reads, key, False) for key in ("image", "audio", "video"))
    if getattr(reads, "previous_turns", 0) > 0 or any(media):
        return (
            "input.request: false reads only the earlier user turns and the tool "
            "results; drop input.previous_turns and the media inputs"
        )
    return None


class SystemOneRule(BaseModel):
    """Routes on a System One model's answer to one question."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    question: SystemOneQuestion
    # The options or labels a choice, set or span rule matches; a set or span
    # rule without them counts every label.
    labels: list[str] = Field(default_factory=list)
    ignore_labels: list[str] = Field(default_factory=list)
    predicate: NumericPredicate | None = None
    # Without one, the question asks the decision model
    # (global.model_catalog.system.decision_model).
    deployment: str | None = None
    stage: Literal["request", "response"] | None = None
    input: SystemOneInput | None = None
    coverage: Literal["truncate", "full"] | None = None
    on_unscanned: Literal["match", "no_match", "unknown"] | None = None
    timeout_ms: int | None = Field(default=None, ge=0, le=MAX_SYSTEMONE_TIMEOUT_MS)

    @model_validator(mode="after")
    def validate_rule(self):
        if not _trimmed(self.name) or ":" in self.name:
            raise ValueError(
                "systemone signal name must be nonempty, trimmed and without ':'"
            )
        if self.deployment is not None and not self.deployment.strip():
            raise ValueError(
                "deployment must name a model_runtime or systemone deployment; "
                "omit it to ask the decision model"
            )
        if self.question.type == "score" and self.predicate is None:
            raise ValueError(
                "a score question requires a predicate on its expected level"
            )
        self._validate_labels()
        self._validate_input()
        return self

    def _validate_labels(self):
        kind = self.question.type
        if self.labels and kind not in OPTION_QUESTION_TYPES:
            raise ValueError(
                "labels name the options a choice, set or span rule matches; "
                f"a {kind} rule takes none"
            )
        if self.ignore_labels and kind not in LABELLED_QUESTION_TYPES:
            raise ValueError("ignore_labels apply only to set and span rules")
        options = self.option_keys()
        for field_name, keys in (
            ("labels", self.labels),
            ("ignore_labels", self.ignore_labels),
        ):
            for key in keys:
                if key not in options:
                    raise ValueError(
                        f"{field_name}: {key!r} is not an option of the question"
                    )
            if len(set(keys)) != len(keys):
                raise ValueError(f"{field_name} lists a label twice")
        if overlap := sorted(set(self.labels) & set(self.ignore_labels)):
            raise ValueError(f"{overlap[0]!r} is both in labels and in ignore_labels")
        if self.ignore_labels and not self.counted_labels():
            raise ValueError("ignore_labels leaves no label to count")

    def _validate_input(self):
        reads = self.input or SystemOneInput()
        if (self.stage or "request") == "request" and (reads.answer or reads.context):
            raise ValueError(
                "input.answer and input.context read the model's answer and its "
                "grounding context, which a request-stage rule does not have; "
                "set stage: response, or read the request's tool results with "
                "input.tool_results"
            )
        if error := turnless_input_error(reads, self.stage):
            raise ValueError(error)
        if self.on_unscanned is not None and self.coverage != "full":
            raise ValueError("on_unscanned applies only to a rule with coverage: full")

    def option_keys(self) -> list[str]:
        if self.question.type in LABELLED_QUESTION_TYPES:
            return [label.key for label in self.question.labels]
        if self.question.type == "noul":
            return ["false", "true"]
        if self.question.type == "score":
            return [str(index) for index in range(len(self.question.levels))]
        return [choice.key for choice in self.question.choices]

    def counted_labels(self) -> list[str]:
        """The options a rule-level match reads, in question order."""
        if self.question.type not in OPTION_QUESTION_TYPES:
            return []
        every = self.question.type != "choice" and not self.labels
        return [
            key
            for key in self.option_keys()
            if (every or key in self.labels) and key not in self.ignore_labels
        ]

    def matches_itself(self) -> bool:
        """Every rule but a choice rule without labels matches itself."""
        return self.question.type != "choice" or bool(self.labels)


class SystemOneSelectionConfig(BaseModel):
    """algorithm.systemone: a Choice over a decision's modelRefs.

    Without a deployment, the Router's decision model chooses.
    """

    model_config = ConfigDict(extra="forbid")

    deployment: str | None = None
    instructions: str
    candidates: dict[str, str] = Field(default_factory=dict)
    timeout_ms: int | None = Field(default=None, ge=0, le=MAX_SYSTEMONE_TIMEOUT_MS)

    @model_validator(mode="after")
    def validate_required_text(self):
        if self.deployment is not None and not _trimmed(self.deployment):
            raise ValueError(
                "deployment must name a model_runtime or systemone deployment; "
                "omit it to ask the decision model"
            )
        if not self.instructions.strip():
            raise ValueError("instructions are required")
        return self
