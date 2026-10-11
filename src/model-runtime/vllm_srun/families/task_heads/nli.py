"""Choice and Noul decisions from a sequence classifier's named NLI labels.

The native encoder and classifier stay shared with /v1/classify. Choice
reads one pair per option and normalizes entailment logits across options;
Noul reads one hypothesis and sums every non-entailment label into false.
No classification probability is mistaken for a candidate logit.
"""

from __future__ import annotations

import math
import string
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from ...errors import (
    DEADLINE_EXCEEDED,
    INVALID_MODEL_OUTPUT,
    INVALID_QUESTION,
    QuestionError,
)
from ...heads.sequence import SequenceHead
from ...heads.task import HeadOptions, Item, Rows
from ...plugins.base import DEADLINE, LoadedModel, SurfacePlan, SurfaceRequest
from ...plugins.decisions import RequestPlan, compare_answers, refuse_unanswerable
from ...systemone import canonical, read_question, valid_state
from ...text.windows import InputTooLongError

LOGITS_HEAD = "_nli_logits"
MAX_HYPOTHESES = 2048


def entailment_index(labels: Sequence[str]) -> int | None:
    """Recognize binary or ternary NLI labels, independent of their index order."""
    names = [label.casefold() for label in labels]
    if len(set(names)) != len(names) or set(names) not in (
        {"entailment", "not_entailment"},
        {"entailment", "contradiction", "neutral"},
    ):
        return None
    return names.index("entailment")


def text(value: Any) -> str:
    return value if isinstance(value, str) else canonical(value)


def hypotheses(raw: Any) -> tuple[str, list[str], list[str]]:
    """Apply shared question validation, then the NLI family's rendering rules."""
    question = read_question(raw)
    instruction = question.instructions
    if not isinstance(instruction, str):
        raise QuestionError(INVALID_QUESTION, "NLI instructions must be text")
    if question.kind == "noul":
        if question.criteria.get("false") is not None:
            raise QuestionError(INVALID_QUESTION, "NLI false is non-entailment")
        hypothesis = question.criteria.get("true")
        return "noul", [], [instruction if hypothesis is None else text(hypothesis)]
    if question.kind != "choice":
        raise QuestionError(INVALID_QUESTION, "NLI supports Choice and Noul only")
    try:
        fields = list(string.Formatter().parse(instruction))
    except ValueError as exc:
        raise QuestionError(INVALID_QUESTION, "invalid hypothesis template") from exc
    placeholders = [
        (field, spec, conversion)
        for _, field, spec, conversion in fields
        if field is not None
    ]
    if placeholders != [("label", "", None)]:
        raise QuestionError(
            INVALID_QUESTION, "Choice needs exactly one {label} placeholder"
        )
    keys = list(question.criteria)
    descriptions = [
        key if value is None else text(value)
        for key, value in question.criteria.items()
    ]
    return "choice", keys, [instruction.format(label=value) for value in descriptions]


class LogitsHead(SequenceHead):
    """Private readout sharing the public head's pooling and classifier tensors."""

    def readout(self, rows: Rows, sequences: Sequence[int]) -> list[Any]:
        return [tuple(row) for row in self.logits(rows, sequences).cpu().tolist()]


@dataclass(frozen=True)
class DecisionEntry:
    kind: Any
    keys: list[str]
    start: int
    end: int
    error: str | None = None
    message: str | None = None


def answer(
    kind: str, keys: list[str], rows: Sequence[Any], positive: int, label_count: int
) -> dict[str, Any]:
    """Stable softmax with NLI semantics; reject malformed/nonfinite readouts."""
    if not rows or any(
        not isinstance(row, (tuple, list))
        or len(row) != label_count
        or positive >= len(row)
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in row)
        for row in rows
    ):
        return {"type": kind, "error": INVALID_MODEL_OUTPUT}
    logits = [row[positive] for row in rows] if kind == "choice" else list(rows[0])
    maximum = max(logits)
    exps = [math.exp(value - maximum) for value in logits]
    total = sum(exps)
    probabilities = [value / total for value in exps]
    if kind == "noul":
        return {"type": kind, "noul": probabilities[positive]}
    if len(keys) != len(probabilities):
        return {"type": kind, "error": INVALID_MODEL_OUTPUT}
    winner = probabilities.index(max(probabilities))
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return {
        "type": kind,
        "choice": keys[winner],
        "probabilities": dict(zip(keys, probabilities, strict=True)),
        "confidence": max(0.0, min(1.0, 1.0 - entropy / math.log(len(keys)))),
    }


class NLIModel(LoadedModel[Item, Any]):
    """Decisions over a task model's shared classifier and execution path."""

    fuse_bundled_jobs = True

    def __init__(
        self, model: LoadedModel[Item, Any], head: SequenceHead, positive: int
    ):
        self.model = model
        self.head = head
        self.info = replace(
            model.info,
            surfaces=("classify", "decisions"),
            question_types=("choice", "noul"),
        )
        self.engine_model = model.engine_model
        self.packs_rows = model.packs_rows
        self.batch_invariant = model.batch_invariant
        self.limit = model.info.limits["max_input_tokens"]
        self.positive = positive

    def run(self, items: list[Item]) -> list[Any]:
        return self.model.run(items)

    def run_approximate(self, items: list[Item]) -> list[Any]:
        return self.model.run_approximate(items)

    def forward_token_budget(self) -> int | None:
        return self.model.forward_token_budget()

    def shared_context(self, items: list[Item], token_budget: int | None) -> int | None:
        return self.model.shared_context(items, token_budget)

    def plan_surface(self, surface: str, request: SurfaceRequest) -> SurfacePlan[Item]:
        if surface != "decisions":
            if request.body.get("head") == LOGITS_HEAD:
                raise ValueError("the NLI logits head is private")
            return self.model.plan_surface(surface, request)
        if request.options.get("max_tokens") is not None:
            raise ValueError(
                "NLI decisions reject overflow and do not accept a scan budget"
            )
        state, questions = request.body.get("state"), request.body.get("questions")
        if not valid_state(state) or not text(state).strip():
            raise ValueError("state must be non-blank text, an object or an array")
        premise = text(state)
        if (
            not isinstance(questions, dict)
            or not questions
            or any(not isinstance(key, str) or not key.strip() for key in questions)
        ):
            raise ValueError("questions must be a nonempty mapping of question IDs")
        items: list[Item] = []
        entries: dict[str, DecisionEntry] = {}
        tokens = 0
        head = self.head
        options = HeadOptions(overflow="reject", max_tokens=self.limit)
        for key, raw in questions.items():
            kind = raw.get("type") if isinstance(raw, dict) else None
            kind = kind if isinstance(kind, str) else None
            start = len(items)
            try:
                kind, keys, candidates = hypotheses(raw)
                if len(items) + len(candidates) > MAX_HYPOTHESES:
                    raise ValueError(
                        f"NLI decisions allow at most {MAX_HYPOTHESES} hypotheses per request"
                    )
                prepared = [
                    head.prepare(
                        {"text": premise, "text_pair": hypothesis},
                        options,
                        self.info.model_sha256,
                    )
                    for hypothesis in candidates
                ]
            except QuestionError as exc:
                entries[key] = DecisionEntry(kind, [], start, start, exc.code, str(exc))
                continue
            except InputTooLongError as exc:
                tokens += exc.tokens
                entries[key] = DecisionEntry(kind, [], start, start, exc.code)
                continue
            for pair in prepared:
                items.extend(pair.items)
                tokens += pair.usage["tokens"]
            entries[key] = DecisionEntry(kind, keys, start, len(items))
        if not request.part:
            errors = {
                key: {"error": entry.error, "message": entry.message or entry.error}
                for key, entry in entries.items()
                if entry.error
            }
            refuse_unanswerable(RequestPlan(list(questions), items, errors, tokens))
        return SurfacePlan("decisions", items, tokens, entries)

    def finish_surface(self, plan: SurfacePlan[Item], results: Any) -> dict[str, Any]:
        if plan.surface != "decisions":
            return self.model.finish_surface(plan, results)
        answers = {}
        for key, entry in plan.state.items():
            error = entry.error
            rows = [] if results is DEADLINE else results[entry.start : entry.end]
            if error is None and (
                results is DEADLINE or any(row is DEADLINE for row in rows)
            ):
                error = DEADLINE_EXCEEDED
            answers[key] = (
                {"type": entry.kind, "error": error}
                if error
                else answer(
                    entry.kind, entry.keys, rows, self.positive, len(self.head.labels)
                )
            )
            if entry.message:
                answers[key]["message"] = entry.message
        return {"answers": answers}

    def golden_values(self, surface: str, response: dict[str, Any]) -> dict[str, Any]:
        return (
            response["answers"]
            if surface == "decisions"
            else self.model.golden_values(surface, response)
        )

    def golden_compare(
        self,
        surface: str,
        values: dict[str, Any],
        reference: dict[str, Any],
        tolerance: float,
    ) -> tuple[int, int] | None:
        return (
            compare_answers(values, reference, tolerance)
            if surface == "decisions"
            else self.model.golden_compare(surface, values, reference, tolerance)
        )

    def outcomes(self, surface: str, body: dict[str, Any]) -> Iterable[tuple[str, str]]:
        if surface == "decisions":
            return [
                (str(value.get("type")), value.get("error", "answered"))
                for value in body.get("answers", {}).values()
            ]
        return self.model.outcomes(surface, body)
