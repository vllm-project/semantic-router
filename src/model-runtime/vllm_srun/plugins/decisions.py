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

import contextvars
import math
from abc import abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, ClassVar, Generic

from .. import assets as assetlib
from ..errors import (
    INVALID_QUESTION,
    QUESTION_FAILED,
    MediaError,
    NoValidQuestionError,
    QuestionError,
    RuntimeServiceError,
    question_error,
    with_reason,
)
from ..systemone import BOX_TYPES, read_box
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
STATE_FIELDS = frozenset({"state", "questions", "images", "videos"})
# Fields a decisions request shares with every entry of its ``states``.
SHARED_FIELDS = ("model", "options", "assets")
# Whether the request being planned asked for long context (``options.long_context``): a family that caps
# questions at a validated length below its maximum reads it in ``plan`` / ``plan_images``.
LONG_CONTEXT: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "long_context", default=False
)


@dataclass(frozen=True)
class ResolvedDecision:
    """A request with assets or Box / track questions, for a model that reads images where the request puts them
    (``plan_resolved``).

    ``state`` and each question's ``instructions`` and option descriptions hold
    ``assets.Segments`` where they reference an asset; ``images`` and
    ``videos`` are the request's own lists, read in front of the state.
    ``targets`` gives each valid box / track question its ``assets.Target``.
    """

    state: Any
    questions: dict[str, Any]
    images: list[Any]
    videos: list[Any]
    uses: list[assetlib.Use]
    targets: dict[str, assetlib.Target]
    long_context: bool


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
    complete_inputs: frozenset[str] = field(default_factory=frozenset)


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
    raise NoValidQuestionError(
        f"no question is valid ({reasons})",
        [(question_id, plan.errors[question_id]) for question_id in plan.question_ids],
    )


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
    shared = {key: body[key] for key in SHARED_FIELDS if key in body}
    bodies = [{key: value for key, value in body.items() if key != "states"}]
    names = []
    for name, entry in states.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("every state in states needs a non-blank name")
        if (
            not isinstance(entry, dict)
            or not {"state", "questions"} <= set(entry) <= STATE_FIELDS
        ):
            raise ValueError(
                f"states[{name!r}] must hold state and questions, and may hold images and videos"
            )
        bodies.append({**shared, **entry})
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
        if status != HTTPStatus.OK:
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
            "invalid_request",
            f"no question is valid ({reasons})",
            answers[0][1].get("reason"),
            question=answers[0][0],
        )
        return error.status, error.body()
    response = dict(outcomes[0][1])
    response["states"] = {
        name: body for name, (_, body) in zip(names, outcomes[1:], strict=True)
    }
    return HTTPStatus.OK, response


def as_systemone(status: int, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """A decisions outcome as ``POST /v1/systemone`` answers it: a failed question fails the whole request.

    System One answers carry no per-question error, and the official SDKs
    validate each answer by its type, so the first failed question (of the
    request's own state, then of each ``states`` entry) answers the request
    with its code, fixed reason and ID; the message lists every failure. A
    request none of whose questions is valid is that question's error too.
    """
    if status == HTTPStatus.OK:
        failures = [
            (question_id, answer, None)
            for question_id, answer in body.get("answers", {}).items()
            if "error" in answer
        ]
        for name, entry in (body.get("states") or {}).items():
            failures += [
                (question_id, answer, name)
                for question_id, answer in entry.get("answers", {}).items()
                if "error" in answer
            ]
        if not failures:
            return status, body
        listed = "; ".join(
            ("" if name is None else f"states.{name}.")
            + f"questions.{question_id}: {answer['error']} ({answer.get('reason')})"
            + (f": {answer['message']}" if answer.get("message") else "")
            for question_id, answer, name in failures
        )
        question_id, answer, _ = failures[0]
        error = RuntimeServiceError(
            QUESTION_FAILED,
            f"{len(failures)} question(s) cannot be answered: {listed}",
            answer.get("reason"),
            question=question_id,
            question_error=answer["error"],
        )
        return error.status, error.body()
    refused = body.get("error") if isinstance(body, dict) else None
    if (
        status == HTTPStatus.BAD_REQUEST
        and isinstance(refused, dict)
        and refused.get("question") is not None
    ):
        error = RuntimeServiceError(
            QUESTION_FAILED,
            str(refused.get("message", "")),
            refused.get("reason"),
            question=refused["question"],
            question_error=INVALID_QUESTION,
        )
        return error.status, error.body()
    return status, body


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
    ``image_inputs`` says the model reads a request's ``images``
    (``plan_images``); a text model refuses a non-empty list. ``video_inputs``
    says it also reads ``videos`` (passed to ``plan_images``).
    """

    scan_tokens: int | None = None
    image_inputs: ClassVar[bool] = False
    video_inputs: bool = False

    @abstractmethod
    def plan(
        self, state: Any, questions: dict[str, Any], scan: int | None = None
    ) -> RequestPlan[ItemT]:
        """Validate and render every question; failures become per-question errors.

        ``scan`` is the request's scan budget, given only to a model with ``scan_tokens``.
        """

    def plan_images(
        self,
        state: Any,
        questions: dict[str, Any],
        images: list[Any],
        videos: list[Any] | None = None,
    ) -> RequestPlan[ItemT]:
        """``plan`` for a request with images (and videos, models with ``video_inputs``) that every question sees.

        A malformed image or video fails the request (``ValueError``).
        """
        raise NotImplementedError(f"{type(self).__name__} reads no images")

    def answer(self, item: RenderedItem, logits: list[float] | None) -> dict[str, Any]:
        """The API answer for one rendered question (``finish_surface`` assembles them)."""
        raise NotImplementedError(f"{type(self).__name__} assembles its own answers")

    def plan_resolved(self, request: ResolvedDecision) -> RequestPlan[ItemT]:
        """``plan`` for a request with assets or Box / track questions on a model that reads images where the
        request puts them.

        Called only when ``reads_resolved``: the model's ``info.image_placements``
        go beyond the state or its ``question_types`` list box or track.
        """
        raise NotImplementedError(
            f"{type(self).__name__} reads images only in front of the state"
        )

    def reads_resolved(self) -> bool:
        return bool(set(self.info.image_placements) - {"state"}) or any(
            kind in self.info.question_types for kind in BOX_TYPES
        )

    def plan_surface(self, surface: str, request: SurfaceRequest) -> SurfacePlan[ItemT]:
        """Validate a decisions request and render its questions with ``plan``.

        Box and track questions a model does not declare, and questions with
        their own images on a model that reads images only in front of the
        state, fail alone with their fixed reasons. A request with assets on
        such a model reads each state use as one more image in front of the
        state, the use written as its label (``assets.hoist``). On a model that
        ``reads_resolved``, a request with assets or with a Box or track
        question it answers goes to ``plan_resolved``, whether or not it has
        assets; any other request to ``plan`` / ``plan_images``.
        """
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
        images = body.get("images")
        if images is not None and not isinstance(images, list):
            raise MediaError("image_format", "images must be a list of image data URLs")
        if images and not self.image_inputs:
            raise MediaError(
                "image_unsupported",
                f"model {self.info.id} reads text only; images are not supported",
            )
        videos = body.get("videos")
        if videos is not None and not isinstance(videos, list):
            raise MediaError("video_format", "videos must be a list of video data URLs")
        if videos and not self.video_inputs:
            raise MediaError(
                "video_unsupported", f"model {self.info.id} reads no videos"
            )
        long_context = bool(request.options.get("long_context", False))
        failures: dict[str, dict[str, Any]] = {}
        for question_id, question in questions.items():
            kind = question.get("type") if isinstance(question, dict) else None
            if kind in BOX_TYPES and kind not in self.info.question_types:
                failures[question_id] = question_error(
                    kind,
                    QuestionError(
                        INVALID_QUESTION,
                        f"model {self.info.id} does not answer {kind} questions",
                        f"{kind}_unsupported",
                    ),
                )
        assets = body.get("assets")
        state = body["state"]
        extra: list[assetlib.Asset] = []
        located = any(
            question_id not in failures
            and isinstance(question, dict)
            and question.get("type") in BOX_TYPES
            for question_id, question in questions.items()
        )
        token = LONG_CONTEXT.set(long_context)
        try:
            if assets and not self.image_inputs:
                raise MediaError(
                    "image_unsupported",
                    f"model {self.info.id} reads text only; assets are not supported",
                )
            if self.reads_resolved() and (assets or located):
                resolved = assetlib.resolve(state, questions, assets or {})
                plan = self.plan_resolved(
                    self._resolved(
                        resolved, images or [], videos or [], long_context, failures
                    )
                )
                return self._surface(request, plan, failures)
            if assets:
                resolved = assetlib.resolve(state, questions, assets)
                for question_id in questions:
                    if question_id not in failures and resolved.question_uses(
                        question_id
                    ):
                        failures[question_id] = question_error(
                            (
                                questions[question_id].get("type")
                                if isinstance(questions[question_id], dict)
                                else None
                            ),
                            QuestionError(
                                INVALID_QUESTION,
                                f"model {self.info.id} reads images only in the state, not in a "
                                "question's instructions or options",
                                "question_images_unsupported",
                            ),
                        )
                state, extra = assetlib.hoist(resolved.state, len(images or []))
            remaining = {
                question_id: question
                for question_id, question in questions.items()
                if question_id not in failures
            }
            if not remaining:
                plan = RequestPlan(list(questions), [], {}, 0)
            elif images or extra or videos:
                plan = self._plan_images_located(
                    state, remaining, images or [], extra, videos or None
                )
            else:
                plan = self.plan(state, remaining, scan)
        finally:
            LONG_CONTEXT.reset(token)
        return self._surface(request, plan, failures)

    def _plan_images_located(
        self,
        state: Any,
        questions: dict[str, Any],
        images: list[Any],
        extra: list[assetlib.Asset],
        videos: list[Any] | None,
    ) -> RequestPlan[ItemT]:
        """``plan_images`` with hoisted assets after the request's images; a rejected asset is named as one."""
        try:
            return self.plan_images(
                state,
                questions,
                [*images, *(asset.data_url() for asset in extra)],
                videos,
            )
        except MediaError as exc:
            prefix = str(exc).split(":", 1)[0]
            for number, asset in enumerate(extra, start=len(images)):
                if prefix == f"images[{number}]":
                    message = str(exc).split(":", 1)[1].strip()
                    raise MediaError(
                        exc.reason,
                        f"assets[{asset.reference[:64]!r}]: {message}",
                        exc.limit,
                    ) from exc
            raise

    def _resolved(
        self,
        resolved: assetlib.Resolved,
        images: list[Any],
        videos: list[Any],
        long_context: bool,
        failures: dict[str, dict[str, Any]],
    ) -> ResolvedDecision:
        state_images = len(images) + len(resolved.state_uses())
        targets: dict[str, assetlib.Target] = {}
        for question_id, question in resolved.questions.items():
            if question_id in failures or not isinstance(question, dict):
                continue
            if question.get("type") not in BOX_TYPES:
                continue
            try:
                read_box(assetlib.plain(question))
            except QuestionError as exc:
                failures[question_id] = question_error(question.get("type"), exc)
                continue
            target = assetlib.box_target(
                question, resolved.question_uses(question_id), state_images, len(videos)
            )
            if isinstance(target, str):
                failures[question_id] = question_error(
                    question.get("type"),
                    QuestionError(
                        INVALID_QUESTION,
                        f"{question['type']} question: {target}",
                        target,
                    ),
                )
            else:
                targets[question_id] = target
        return ResolvedDecision(
            resolved.state,
            {qid: q for qid, q in resolved.questions.items() if qid not in failures},
            images,
            videos,
            resolved.uses,
            targets,
            long_context,
        )

    def _surface(
        self,
        request: SurfaceRequest,
        plan: RequestPlan[ItemT],
        failures: dict[str, dict[str, Any]],
    ) -> SurfacePlan[ItemT]:
        if failures:
            order = list(request.body["questions"])
            plan = RequestPlan(
                question_ids=order,
                items=plan.items,
                errors={
                    question_id: failures.get(question_id) or plan.errors[question_id]
                    for question_id in order
                    if question_id in failures or question_id in plan.errors
                },
                input_tokens=plan.input_tokens,
                complete_inputs=plan.complete_inputs,
            )
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
                answer = self.answer(item, logits)
                if (
                    item.question_id in request_plan.complete_inputs
                    and "error" not in answer
                ):
                    answer["input_coverage"] = "complete"
                answered[item.question_id] = answer
        return {
            "answers": {
                question_id: (
                    with_reason(request_plan.errors[question_id])
                    if question_id in request_plan.errors
                    else with_reason(answered[question_id])
                )
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
