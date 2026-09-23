"""Strict request and response contracts for the SystemOne-compatible API."""

from __future__ import annotations

import math
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    JsonValue,
    NonNegativeInt,
    field_validator,
    model_validator,
)

from .confidence import normalized_top_confidence

JsonContent: TypeAlias = str | dict[str, JsonValue] | list[JsonValue]
Probability: TypeAlias = Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]

PROBABILITY_SUM_TOLERANCE = 2e-5
# These are wire-shape limits only. Raw request bytes, expanded input bytes,
# model token admission, and physical microbatch sizing belong to the production
# Gateway/backend adapters and are deliberately not approximated in this layer.
MAX_BATCH_DECISIONS = 1024
MAX_BATCH_QUESTIONS = 1024
MAX_BATCH_STATES = 1024


class ContractModel(BaseModel):
    """Shared strictness for every public request and response object."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        strict=True,
    )


class NoulCriteria(ContractModel):
    """Optional descriptions of the true and false outcomes."""

    true: JsonContent | None = None
    false: JsonContent | None = None


class NoulQuestion(ContractModel):
    """A yes/no question with required, non-null instructions."""

    type: Literal["noul"]
    instructions: JsonContent
    criteria: NoulCriteria | None = None


class ChoiceQuestion(ContractModel):
    """A selection among two to 255 named options."""

    type: Literal["choice"]
    instructions: JsonContent
    criteria: Annotated[
        dict[str, JsonContent | None], Field(min_length=2, max_length=255)
    ]

    @field_validator("criteria")
    @classmethod
    def option_names_must_be_nonblank(
        cls, value: dict[str, JsonContent | None]
    ) -> dict[str, JsonContent | None]:
        if any(not name.strip() for name in value):
            raise ValueError("Choice option names must not be empty or whitespace")
        return value


class ScoreQuestion(ContractModel):
    """A rating against two to ten ordered rubric levels."""

    type: Literal["score"]
    instructions: JsonContent
    criteria: Annotated[list[JsonContent], Field(min_length=2, max_length=10)]


Question: TypeAlias = Annotated[
    NoulQuestion | ChoiceQuestion | ScoreQuestion,
    Field(discriminator="type"),
]


class SystemOneRequest(ContractModel):
    """One state and one or more named questions for an explicit model."""

    state: JsonContent
    model: Annotated[str, Field(min_length=1)]
    questions: Annotated[dict[str, Question], Field(min_length=1)]

    @field_validator("model")
    @classmethod
    def model_must_be_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be empty or whitespace")
        return value

    @field_validator("questions")
    @classmethod
    def question_ids_must_be_nonblank(
        cls, value: dict[str, Question]
    ) -> dict[str, Question]:
        if any(not question_id.strip() for question_id in value):
            raise ValueError("question IDs must not be empty or whitespace")
        return value


class BatchState(ContractModel):
    """One caller-identified state in a shared-question batch."""

    id: Annotated[str, Field(min_length=1, max_length=128)]
    state: JsonContent

    @field_validator("id")
    @classmethod
    def id_must_be_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("state ID must not be empty or whitespace")
        return value


class SystemOneBatchRequest(ContractModel):
    """Many identified states evaluated against one shared question map."""

    model: Annotated[str, Field(min_length=1)]
    states: Annotated[
        list[BatchState], Field(min_length=1, max_length=MAX_BATCH_STATES)
    ]
    questions: Annotated[
        dict[str, Question], Field(min_length=1, max_length=MAX_BATCH_QUESTIONS)
    ]

    @field_validator("model")
    @classmethod
    def model_must_be_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be empty or whitespace")
        return value

    @field_validator("states")
    @classmethod
    def state_ids_must_be_unique(cls, value: list[BatchState]) -> list[BatchState]:
        identifiers = [state.id for state in value]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("state IDs must be unique")
        return value

    @field_validator("questions")
    @classmethod
    def question_ids_must_be_nonblank(
        cls, value: dict[str, Question]
    ) -> dict[str, Question]:
        if any(not question_id.strip() for question_id in value):
            raise ValueError("question IDs must not be empty or whitespace")
        return value

    @model_validator(mode="after")
    def decision_count_must_fit_batch(self) -> SystemOneBatchRequest:
        if len(self.states) * len(self.questions) > MAX_BATCH_DECISIONS:
            raise ValueError(
                f"states multiplied by questions must not exceed "
                f"{MAX_BATCH_DECISIONS} decisions"
            )
        return self


class NoulAnswer(ContractModel):
    """Probability that a Noul statement is true."""

    type: Literal["noul"]
    noul: Probability


class ChoiceAnswer(ContractModel):
    """Selected option, its distribution, and Decision confidence."""

    type: Literal["choice"]
    choice: str
    confidence: Probability
    probabilities: dict[str, Probability]


class ScoreAnswer(ContractModel):
    """Expected rubric score, distribution, legend, and Decision confidence."""

    type: Literal["score"]
    score: FiniteFloat
    confidence: Probability
    legend: dict[str, JsonContent]
    probabilities: dict[str, Probability]


Answer: TypeAlias = Annotated[
    NoulAnswer | ChoiceAnswer | ScoreAnswer,
    Field(discriminator="type"),
]


class Usage(ContractModel):
    """Request-level token counts."""

    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt


class SystemOneResponse(ContractModel):
    """Official response-body surface: model, answers, and usage only."""

    model: Annotated[str, Field(min_length=1)]
    answers: Annotated[dict[str, Answer], Field(min_length=1)]
    usage: Usage

    @field_validator("model")
    @classmethod
    def model_must_be_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be empty or whitespace")
        return value


class SystemOneBatchResult(ContractModel):
    """Official answer and usage shapes for one identified batch state."""

    id: Annotated[str, Field(min_length=1, max_length=128)]
    answers: Annotated[dict[str, Answer], Field(min_length=1)]
    usage: Usage

    @field_validator("id")
    @classmethod
    def id_must_be_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("state ID must not be empty or whitespace")
        return value


class SystemOneBatchResponse(ContractModel):
    """Atomic shared-question batch response without diagnostic fields."""

    model: Annotated[str, Field(min_length=1)]
    results: Annotated[list[SystemOneBatchResult], Field(min_length=1)]
    usage: Usage

    @field_validator("model")
    @classmethod
    def model_must_be_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be empty or whitespace")
        return value


class ResponseContractError(ValueError):
    """A backend response violated a request-relative invariant."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message


def _require_distribution(
    probabilities: dict[str, float], expected_keys: list[str], path: str
) -> None:
    if set(probabilities) != set(expected_keys):
        raise ResponseContractError(
            path, "probability keys must match the request criteria"
        )
    if not math.isclose(
        math.fsum(probabilities.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=PROBABILITY_SUM_TOLERANCE,
    ):
        raise ResponseContractError(path, "probabilities must sum to one")


def _require_confidence(
    confidence: float,
    probabilities: dict[str, float],
    path: str,
) -> None:
    expected = normalized_top_confidence(tuple(probabilities.values()))
    if not math.isclose(
        confidence,
        expected,
        rel_tol=0.0,
        abs_tol=PROBABILITY_SUM_TOLERANCE,
    ):
        raise ResponseContractError(
            path, "confidence must match Decision normalized-top confidence"
        )


def validate_response_for_request(
    request: SystemOneRequest,
    response: SystemOneResponse,
) -> SystemOneResponse:
    """Validate answer identity and probability math against the request.

    Pydantic validates each object in isolation. This function enforces the
    invariants that depend on the originating request and must run before an
    HTTP response is returned.
    """

    if response.model != request.model:
        raise ResponseContractError("model", "response model must match the request")

    if set(response.answers) != set(request.questions):
        raise ResponseContractError(
            "answers", "answer keys must match the request questions"
        )

    for question_id, question in request.questions.items():
        answer = response.answers[question_id]
        path = f"answers.{question_id}"
        if answer.type != question.type:
            raise ResponseContractError(path, "answer type must match question type")

        if isinstance(question, NoulQuestion):
            continue

        if isinstance(question, ChoiceQuestion):
            if not isinstance(answer, ChoiceAnswer):
                raise ResponseContractError(path, "expected a Choice answer")
            keys = list(question.criteria)
            _require_distribution(answer.probabilities, keys, f"{path}.probabilities")
            _require_confidence(
                answer.confidence,
                answer.probabilities,
                f"{path}.confidence",
            )
            if answer.choice not in answer.probabilities:
                raise ResponseContractError(
                    f"{path}.choice", "choice must name a requested option"
                )
            top = max(answer.probabilities.values())
            if not math.isclose(
                answer.probabilities[answer.choice],
                top,
                rel_tol=0.0,
                abs_tol=PROBABILITY_SUM_TOLERANCE,
            ):
                raise ResponseContractError(
                    f"{path}.choice", "choice must be a highest-probability option"
                )
            continue

        if not isinstance(answer, ScoreAnswer):
            raise ResponseContractError(path, "expected a Score answer")
        keys = [str(index) for index in range(len(question.criteria))]
        _require_distribution(answer.probabilities, keys, f"{path}.probabilities")
        _require_confidence(
            answer.confidence,
            answer.probabilities,
            f"{path}.confidence",
        )
        expected_legend = {
            str(index): criterion for index, criterion in enumerate(question.criteria)
        }
        if answer.legend != expected_legend:
            raise ResponseContractError(
                f"{path}.legend", "legend must reproduce the ordered criteria"
            )
        expected_score = math.fsum(
            index * answer.probabilities[str(index)] for index in range(len(keys))
        )
        if not math.isclose(
            answer.score,
            expected_score,
            rel_tol=0.0,
            abs_tol=PROBABILITY_SUM_TOLERANCE,
        ):
            raise ResponseContractError(
                f"{path}.score", "score must be the probability-weighted level"
            )

    return response


def validate_batch_response_for_request(
    request: SystemOneBatchRequest,
    response: SystemOneBatchResponse,
) -> SystemOneBatchResponse:
    """Validate atomic identity, answer math, and usage for a batch response."""

    if response.model != request.model:
        raise ResponseContractError("model", "response model must match the request")

    expected_ids = [state.id for state in request.states]
    actual_ids = [result.id for result in response.results]
    if actual_ids != expected_ids:
        raise ResponseContractError(
            "results", "result state identity and order must match the request"
        )

    for index, (state, result) in enumerate(
        zip(request.states, response.results, strict=True)
    ):
        single_request = SystemOneRequest(
            state=state.state,
            model=request.model,
            questions=request.questions,
        )
        single_response = SystemOneResponse(
            model=response.model,
            answers=result.answers,
            usage=result.usage,
        )
        try:
            validate_response_for_request(single_request, single_response)
        except ResponseContractError as error:
            raise ResponseContractError(
                f"results.{index}.{error.path}", error.message
            ) from error

    expected_input_tokens = sum(
        result.usage.input_tokens for result in response.results
    )
    expected_output_tokens = sum(
        result.usage.output_tokens for result in response.results
    )
    if response.usage.input_tokens != expected_input_tokens:
        raise ResponseContractError(
            "usage.input_tokens", "must equal the sum of result input tokens"
        )
    if response.usage.output_tokens != expected_output_tokens:
        raise ResponseContractError(
            "usage.output_tokens", "must equal the sum of result output tokens"
        )

    return response
