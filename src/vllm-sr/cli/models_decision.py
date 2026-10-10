"""Decision signals: typed questions answered by a model_runtime deployment."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models_predicates import NumericPredicate

MAX_DECISION_TIMEOUT_MS = 60_000
MIN_DECISION_CHOICES = 2
MAX_DECISION_CHOICES = 255
MIN_DECISION_LEVELS = 2
MAX_DECISION_LEVELS = 10
MIN_DECISION_LABELS = 1
MAX_DECISION_LABELS = 255
NOUL_CHOICE_KEYS = frozenset({"false", "true"})
LABELLED_QUESTION_TYPES = frozenset({"set", "span"})
# Question types whose conditions name one of the question's options.
OPTION_QUESTION_TYPES = frozenset({"choice", "set", "span"})


def _trimmed(value: str) -> bool:
    return bool(value.strip()) and value == value.strip()


class DecisionChoice(BaseModel):
    """One option of a choice question, a noul description, or a set or span label."""

    model_config = ConfigDict(extra="forbid")

    key: str
    description: str | None = None


class DecisionQuestion(BaseModel):
    """A System One question: choice, noul or score; set and span for models that declare them."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["choice", "noul", "score", "set", "span"]
    instructions: str
    choices: list[DecisionChoice] = Field(default_factory=list)
    levels: list[str] = Field(default_factory=list)
    labels: list[DecisionChoice] = Field(default_factory=list)
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
            if not MIN_DECISION_LEVELS <= len(self.levels) <= MAX_DECISION_LEVELS:
                raise ValueError(
                    f"a score question needs {MIN_DECISION_LEVELS}..{MAX_DECISION_LEVELS} levels"
                )
            if any(not level.strip() for level in self.levels):
                raise ValueError("every level needs a description")
            return self
        if self.type == "choice" and not (
            MIN_DECISION_CHOICES <= len(self.choices) <= MAX_DECISION_CHOICES
        ):
            raise ValueError(
                f"a choice question needs {MIN_DECISION_CHOICES}..{MAX_DECISION_CHOICES} choices"
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
        if not MIN_DECISION_LABELS <= len(self.labels) <= MAX_DECISION_LABELS:
            raise ValueError(
                f"a {self.type} question needs "
                f"{MIN_DECISION_LABELS}..{MAX_DECISION_LABELS} labels"
            )
        keys = [label.key for label in self.labels]
        if any(not _trimmed(key) for key in keys):
            raise ValueError("label keys must be nonempty and trimmed")
        if len(set(keys)) != len(keys):
            raise ValueError("label keys must be unique")
        return self


class DecisionSignalRule(BaseModel):
    """Routes on a decision model's answer to one question about the request."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    # Without one, the question asks the decision model
    # (global.model_catalog.system.decision_model).
    deployment: str | None = None
    question: DecisionQuestion
    predicate: NumericPredicate | None = None
    timeout_ms: int | None = Field(default=None, ge=0, le=MAX_DECISION_TIMEOUT_MS)

    @model_validator(mode="after")
    def validate_rule(self):
        if not _trimmed(self.name) or ":" in self.name:
            raise ValueError(
                "decision signal name must be nonempty, trimmed and without ':'"
            )
        if self.deployment is not None and not self.deployment.strip():
            raise ValueError(
                "deployment must name a model_runtime deployment; omit it to ask "
                "the decision model"
            )
        if self.question.type == "score" and self.predicate is None:
            raise ValueError(
                "a score question requires a predicate on its expected level"
            )
        return self

    def option_keys(self) -> list[str]:
        if self.question.type in LABELLED_QUESTION_TYPES:
            return [label.key for label in self.question.labels]
        return [choice.key for choice in self.question.choices]


class DecisionSelectionConfig(BaseModel):
    """algorithm.decision: a Choice over a decision's modelRefs.

    Without a deployment, the Router's decision model chooses.
    """

    model_config = ConfigDict(extra="forbid")

    deployment: str | None = None
    instructions: str
    candidates: dict[str, str] = Field(default_factory=dict)
    timeout_ms: int | None = Field(default=None, ge=0, le=MAX_DECISION_TIMEOUT_MS)

    @model_validator(mode="after")
    def validate_required_text(self):
        if self.deployment is not None and not _trimmed(self.deployment):
            raise ValueError(
                "deployment must name a model_runtime deployment; omit it to ask "
                "the decision model"
            )
        if not self.instructions.strip():
            raise ValueError("instructions are required")
        return self
