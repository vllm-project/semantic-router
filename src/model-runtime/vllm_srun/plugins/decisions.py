"""The decisions surface (``/v1/decisions``): System One questions over a state, one answer per question.

A decision family's loaded model subclasses ``DecisionModel``, the decisions
mixin over ``LoadedModel``: ``plan`` validates and renders a request's
questions and ``answer`` turns one rendered question's readout into its
answer. The mixin serves them as the ``decisions`` surface, compares golden
answers question by question, and reports each question's type and outcome
for the runtime's metrics. A family that assembles a request's answers itself
(``vela2``) overrides ``finish_surface`` instead of ``answer``.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Generic

from ..errors import INVALID_QUESTION, RuntimeServiceError
from .base import (
    Expired,
    ItemT,
    LoadedModel,
    Results,
    ResultT,
    SurfacePlan,
    SurfaceRequest,
    UnsupportedSurfaceError,
)

SURFACE = "decisions"
SUM_TOLERANCE = 1e-6


@dataclass(frozen=True)
class RenderedItem:
    """One question rendered to model inputs."""

    question_id: str
    task_type: str
    ids: list[int]
    gather: list[int]
    query: int
    keys: list[str]
    descriptions: list[Any]


@dataclass
class RequestPlan(Generic[ItemT]):
    """A request after validation and rendering, before execution."""

    question_ids: list[str]
    items: Sequence[ItemT]
    errors: dict[str, dict[str, Any]]
    input_tokens: int


def refuse_unanswerable(plan: RequestPlan[Any]) -> None:
    """Fail a request whose questions are all invalid; nothing in it can be answered.

    The ``ValueError`` becomes 400 invalid_request with every question's
    reason. An invalid question among valid ones still fails alone, in its
    answer.
    """
    if plan.items or len(plan.errors) < len(plan.question_ids):
        return
    if any(
        plan.errors.get(question_id, {}).get("error") != INVALID_QUESTION
        for question_id in plan.question_ids
    ):
        return
    reasons = "; ".join(
        f"{question_id}: {plan.errors[question_id].get('message', INVALID_QUESTION)}"
        for question_id in plan.question_ids
    )
    raise ValueError(f"no question is valid ({reasons})")


def split_states(body: Any) -> tuple[list[dict[str, Any]], list[str]] | None:
    """A request with further ``states`` as one request per state, its own first, and the states' names.

    Every request has the original's model and options; None for a request
    without ``states``. ``ValueError`` when ``states`` is malformed.
    """
    if not isinstance(body, dict) or "states" not in body:
        return None
    states = body["states"]
    if not isinstance(states, dict):
        raise ValueError("states must be an object of named states")
    shared = {key: body[key] for key in ("model", "options") if key in body}
    bodies = [{key: value for key, value in body.items() if key != "states"}]
    names = []
    for name, entry in states.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("every state in states needs a non-blank name")
        if not isinstance(entry, dict) or set(entry) != {"state", "questions"}:
            raise ValueError(f"states[{name!r}] must hold exactly state and questions")
        bodies.append(
            {**shared, "state": entry["state"], "questions": entry["questions"]}
        )
        names.append(name)
    return bodies, names


def join_states(
    outcomes: list[tuple[int, dict[str, Any]]], names: list[str]
) -> tuple[int, dict[str, Any]]:
    """The response of a request with further ``states`` from the responses of its states, its own first.

    The first failure answers the request. A request none of whose questions
    is valid is refused, as ``refuse_unanswerable`` refuses one about one state.
    """
    for status, body in outcomes:
        if status != 200:
            return status, body
    answers = [
        (question_id, answer)
        for _, body in outcomes
        for question_id, answer in body.get("answers", {}).items()
    ]
    if answers and all(
        answer.get("error") == INVALID_QUESTION for _, answer in answers
    ):
        reasons = "; ".join(
            f"{question_id}: {answer.get('message', INVALID_QUESTION)}"
            for question_id, answer in answers
        )
        error = RuntimeServiceError(
            "invalid_request", f"no question is valid ({reasons})"
        )
        return error.status, error.body()
    response = dict(outcomes[0][1])
    response["states"] = {
        name: body for name, (_, body) in zip(names, outcomes[1:], strict=True)
    }
    return 200, response


def well_formed(answer: dict[str, Any]) -> bool:
    """A finite, normalized answer of the declared type."""
    if "error" in answer:
        return False
    kind = answer.get("type")
    if kind == "noul":
        value = answer.get("noul")
        return isinstance(value, float) and 0.0 <= value <= 1.0
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        return False
    values: list[float] = list(probabilities.values())
    if any(not isinstance(v, float) or not math.isfinite(v) or v < 0 for v in values):
        return False
    return abs(sum(values) - 1.0) < SUM_TOLERANCE


def compare_answers(
    answers: dict[str, Any], expected: dict[str, Any], tolerance: float
) -> tuple[int, int] | None:
    """(checked, matched) of golden answers against reference answers; None when one is malformed.

    Each reference question that was answered is checked once: its type, its
    probabilities' keys and every value within ``tolerance``.
    """
    if not all(well_formed(answer) for answer in answers.values()):
        return None
    checked = matched = 0
    for question_id, reference in expected.items():
        answer = answers.get(question_id)
        if answer is None:
            continue
        checked += 1
        if answer.get("type") != reference.get("type"):
            continue
        if answer.get("type") == "noul":
            matched += abs(answer["noul"] - reference["noul"]) <= tolerance
            continue
        left, right = answer.get("probabilities", {}), reference.get(
            "probabilities", {}
        )
        if set(left) == set(right) and all(
            abs(left[k] - right[k]) <= tolerance for k in left
        ):
            matched += 1
    return checked, matched


class DecisionModel(LoadedModel[ItemT, ResultT]):
    """A loaded model that serves ``/v1/decisions`` through ``plan`` and ``answer``.

    ``scan_tokens`` is the most tokens of one state part the model reads in
    windows (its card's ``max_scan_tokens``); None for a model that reads one
    bounded input and rejects a longer one. A request's ``options.max_tokens``
    overrides it, and only a model that has one takes that option.
    """

    scan_tokens: int | None = None

    @abstractmethod
    def plan(
        self, state: Any, questions: dict[str, Any], scan: int | None = None
    ) -> RequestPlan[ItemT]:
        """Validate and render every question; failures become per-question errors.

        ``scan`` is the request's scan budget, given only to a model with ``scan_tokens``.
        """

    def answer(self, item: RenderedItem, logits: list[float] | None) -> dict[str, Any]:
        """The API answer for one rendered question (``finish_surface`` assembles them)."""
        raise NotImplementedError(f"{type(self).__name__} assembles its own answers")

    def plan_surface(self, surface: str, request: SurfaceRequest) -> SurfacePlan[ItemT]:
        """Validate a decisions request and render its questions with ``plan``."""
        if surface != SURFACE or SURFACE not in self.info.surfaces:
            raise UnsupportedSurfaceError(surface, self.info.id)
        body = request.body
        if "state" not in body:
            raise ValueError("state is required")
        questions = body.get("questions")
        if (
            not isinstance(questions, dict)
            or not questions
            or any(not isinstance(key, str) or not key.strip() for key in questions)
        ):
            raise ValueError("questions must be a nonempty mapping of question IDs")
        scan = request.options.get("max_tokens")
        if scan is not None:
            if isinstance(scan, bool) or not isinstance(scan, int) or scan < 1:
                raise ValueError("max_tokens must be a positive integer")
            if self.scan_tokens is None:
                raise ValueError(
                    "max_tokens is the scan budget of a model that reads parts in"
                    " windows; this model reads one bounded input and rejects a longer one"
                )
        plan = self.plan(body["state"], questions, scan)
        if not request.part:
            refuse_unanswerable(plan)
        items: list[Any] = list(plan.items)
        return SurfacePlan(SURFACE, items, plan.input_tokens, plan)

    def finish_surface(
        self, plan: SurfacePlan[ItemT], results: Results[ResultT]
    ) -> dict[str, Any]:
        """Every question's answer, in request order: its error, or ``answer`` of its readout."""
        from ..errors import DEADLINE_EXCEEDED

        request_plan: RequestPlan[RenderedItem] = plan.state
        answered: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(request_plan.items):
            if isinstance(results, Expired):
                answered[item.question_id] = {
                    "type": item.task_type,
                    "error": DEADLINE_EXCEEDED,
                }
            else:
                logits: Any = results[index]
                answered[item.question_id] = self.answer(item, logits)
        return {
            "answers": {
                question_id: request_plan.errors.get(question_id)
                or answered[question_id]
                for question_id in request_plan.question_ids
            }
        }

    def golden_values(self, surface: str, response: dict[str, Any]) -> dict[str, Any]:
        """A golden response's answers by question ID."""
        if surface != SURFACE:
            return super().golden_values(surface, response)
        answers: dict[str, Any] = response["answers"]
        return answers

    def golden_compare(
        self,
        surface: str,
        values: dict[str, Any],
        reference: dict[str, Any],
        tolerance: float,
    ) -> tuple[int, int] | None:
        """Answers compare question by question (``compare_answers``)."""
        if surface != SURFACE:
            return super().golden_compare(surface, values, reference, tolerance)
        return compare_answers(values, reference, tolerance)

    def outcomes(self, surface: str, body: dict[str, Any]) -> Iterable[tuple[str, str]]:
        """Each question's type and outcome: ``answered`` or its error code."""
        if surface != SURFACE:
            return super().outcomes(surface, body)
        return [
            (str(answer.get("type")), answer.get("error", "answered"))
            for answer in body.get("answers", {}).values()
        ]
