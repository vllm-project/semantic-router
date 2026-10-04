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
NOUL_CHOICE_KEYS = frozenset({"false", "true"})


def _trimmed(value: str) -> bool:
    return bool(value.strip()) and value == value.strip()


class DecisionChoice(BaseModel):
    """One option of a choice question; noul questions may describe false and true."""

    model_config = ConfigDict(extra="forbid")

    key: str
    description: str | None = None


class DecisionQuestion(BaseModel):
    """A System One question: choice, noul or score."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["choice", "noul", "score"]
    instructions: str
    choices: list[DecisionChoice] = Field(default_factory=list)
    levels: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_shape(self):
        if not self.instructions.strip():
            raise ValueError("instructions are required")
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


class DecisionSignalRule(BaseModel):
    """Routes on a decision model's answer to one question about the request."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    deployment: str
    question: DecisionQuestion
    predicate: NumericPredicate | None = None
    timeout_ms: int | None = Field(default=None, ge=0, le=MAX_DECISION_TIMEOUT_MS)

    @model_validator(mode="after")
    def validate_rule(self):
        if not _trimmed(self.name) or ":" in self.name:
            raise ValueError(
                "decision signal name must be nonempty, trimmed and without ':'"
            )
        if not self.deployment.strip():
            raise ValueError("deployment is required")
        if self.question.type == "score" and self.predicate is None:
            raise ValueError(
                "a score question requires a predicate on its expected level"
            )
        return self

    def option_keys(self) -> list[str]:
        return [choice.key for choice in self.question.choices]


class DecisionSelectionConfig(BaseModel):
    """algorithm.decision: a Choice over a decision's modelRefs."""

    model_config = ConfigDict(extra="forbid")

    deployment: str
    instructions: str
    candidates: dict[str, str] = Field(default_factory=dict)
    timeout_ms: int | None = Field(default=None, ge=0, le=MAX_DECISION_TIMEOUT_MS)

    @model_validator(mode="after")
    def validate_required_text(self):
        if not self.deployment.strip():
            raise ValueError("deployment is required")
        if not self.instructions.strip():
            raise ValueError("instructions are required")
        return self
