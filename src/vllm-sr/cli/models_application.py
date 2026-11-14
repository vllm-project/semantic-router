"""Application signals: rules that ask the Router's presets about a request or
a model's answer.

The Router compiles each rule into one systemone rule (the preset's question,
stage, input and coverage) with the rule's own labels, predicate, deployment
and timeout. Conditions name the application type, and a condition's label
names one of the rule's options.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models_predicates import NumericPredicate
from .models_systemone import MAX_SYSTEMONE_PREVIOUS_TURNS, turnless_input_error

Stage = Literal["request", "response"]
# What a jailbreak or pii rule reads as when its deployment gives no answer.
OnUnknown = Literal["match", "no_match"]
# What a jailbreak or pii rule reads as when only some windows of a long input
# were read.
OnUnscanned = Literal["best_effort", "match", "no_match", "unknown"]
# The share of a long input's windows a guard rule that read only some of them
# must have read to decide on them (best_effort); below it the rule is unknown.
MinCoverage = Annotated[float, Field(ge=0, le=1)]


def min_coverage_error(min_coverage: float | None, on_unscanned: str | None) -> str:
    """Why a rule's min_coverage does not apply to its on_unscanned, or ""."""
    if min_coverage and on_unscanned not in (None, "best_effort"):
        return "min_coverage applies to on_unscanned: best_effort, which decides on the windows read"
    return ""


class ApplicationRule(BaseModel):
    """The fields every application rule accepts."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    # Without one, the rule asks its type's binding (the recipe's, then the
    # global one), else the decision model.
    deployment: str | None = None
    timeout_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_name(self):
        if not self.name.strip() or self.name != self.name.strip() or ":" in self.name:
            raise ValueError("every rule needs a name, trimmed and without ':'")
        if self.deployment is not None and not self.deployment.strip():
            raise ValueError(
                "deployment must name a deployment; omit it to ask the default"
            )
        return self


class ModelScore(BaseModel):
    """A candidate model's weight for requests in a domain."""

    model: str
    score: float
    use_reasoning: bool | None = None


class Domain(ApplicationRule):
    """The request's subject area is one of labels (by default the rule's name)."""

    labels: list[str] = Field(default_factory=list)
    predicate: NumericPredicate | None = None
    model_scores: list[ModelScore] = Field(default_factory=list)


class FactCheck(ApplicationRule):
    """Answering the request needs fact checking (FACT_CHECK_NEEDED) or not."""

    labels: list[str] = Field(default_factory=list)
    predicate: NumericPredicate | None = None


class UserFeedback(ApplicationRule):
    """The feedback a user turn gives about the previous answer."""

    labels: list[str] = Field(default_factory=list)
    predicate: NumericPredicate | None = None


class JailbreakInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    history: bool = False


class JailbreakRule(ApplicationRule):
    """A prompt injection or jailbreak attempt in the request or the answer."""

    predicate: NumericPredicate | None = None
    stage: Stage | None = None
    input: JailbreakInput | None = None
    on_unscanned: OnUnscanned | None = None
    min_coverage: MinCoverage | None = None
    on_unknown: OnUnknown | None = None

    @model_validator(mode="after")
    def validate_min_coverage(self):
        if error := min_coverage_error(self.min_coverage, self.on_unscanned):
            raise ValueError(error)
        return self


class SafetyRule(ApplicationRule):
    """Harmful content, or with labels, content in one of those harm categories."""

    labels: list[str] = Field(default_factory=list)
    predicate: NumericPredicate | None = None
    stage: Stage | None = None
    min_coverage: MinCoverage | None = None


class PIIInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    history: bool = False
    tool_results: bool = False
    # False leaves the current user turn out, as a systemone rule's input does.
    request: bool | None = None


class PIIRule(ApplicationRule):
    """Personal data in the request: a span of a label ignore_labels leaves counted."""

    predicate: NumericPredicate | None = None
    ignore_labels: list[str] = Field(default_factory=list)
    input: PIIInput | None = None
    on_unscanned: OnUnscanned | None = None
    min_coverage: MinCoverage | None = None
    on_unknown: OnUnknown | None = None

    @model_validator(mode="after")
    def validate_input(self):
        if self.input and (error := turnless_input_error(self.input, None)):
            raise ValueError(error)
        if error := min_coverage_error(self.min_coverage, self.on_unscanned):
            raise ValueError(error)
        return self


class HallucinationRule(ApplicationRule):
    """Claims in the answer that its grounding context does not support."""

    predicate: NumericPredicate | None = None


class RefusalRule(ApplicationRule):
    """An answer in which the model refused the request."""

    predicate: NumericPredicate | None = None


class TurnRelationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    previous_turns: int | None = Field(
        default=None, ge=0, le=MAX_SYSTEMONE_PREVIOUS_TURNS
    )


class TurnRelationRule(ApplicationRule):
    """How the current user turn relates to the turns before it."""

    labels: list[str] = Field(default_factory=list)
    input: TurnRelationInput | None = None


class ComplexityRule(ApplicationRule):
    """How difficult the request is: easy, medium or hard."""

    labels: list[str] = Field(default_factory=list)


class ModalityRule(ApplicationRule):
    """The output the request asks for: text, image, text_image and the others."""

    labels: list[str] = Field(default_factory=list)
    predicate: NumericPredicate | None = None
