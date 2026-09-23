"""Deterministic backend for contract tests and local API development."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence

from .backend import (
    BackendBatchRequest,
    BackendBatchResult,
    BackendPrediction,
    BackendResult,
    BackendUnavailableError,
    ModelDescriptor,
    UnknownModelError,
)
from .contracts import ChoiceQuestion, ScoreQuestion, SystemOneRequest


class FakeDecisionBackend:
    """Framework-free backend with stable, non-semantic probabilities."""

    def __init__(
        self,
        models: Sequence[ModelDescriptor],
        *,
        available: bool = True,
    ) -> None:
        if not models:
            raise ValueError("at least one fake model is required")
        if len({model.name for model in models}) != len(models):
            raise ValueError("fake model names must be unique")
        self._models = tuple(models)
        self._model_names = {model.name for model in models}
        self.available = available

    def models(self) -> Sequence[ModelDescriptor]:
        return self._models

    async def ready(self) -> bool:
        return self.available

    async def infer(self, request: SystemOneRequest) -> BackendResult:
        return self._infer_one(request)

    async def infer_batch(
        self, requests: tuple[BackendBatchRequest, ...]
    ) -> tuple[BackendBatchResult, ...]:
        """Exercise the batch seam without claiming physical microbatching."""

        return tuple(
            BackendBatchResult(
                state_id=item.state_id,
                result=self._infer_one(item.request),
            )
            for item in requests
        )

    def _infer_one(self, request: SystemOneRequest) -> BackendResult:
        if request.model not in self._model_names:
            raise UnknownModelError(request.model)
        if not self.available:
            raise BackendUnavailableError("the fake backend is not ready")

        predictions = []
        for question_id, question in request.questions.items():
            if isinstance(question, (ChoiceQuestion, ScoreQuestion)):
                candidate_count = len(question.criteria)
            else:
                candidate_count = 2
            question_payload = question.model_dump(mode="json")
            seed = _canonical_bytes(
                {
                    "model": request.model,
                    "state": request.state,
                    "question": question_payload,
                }
            )
            predictions.append(
                BackendPrediction(
                    question_id=question_id,
                    type=question.type,
                    probabilities=_distribution(seed, candidate_count),
                )
            )

        request_bytes = _canonical_bytes(request.model_dump(mode="json"))
        return BackendResult(
            model=request.model,
            predictions=tuple(predictions),
            input_tokens=max(1, (len(request_bytes) + 3) // 4),
            output_tokens=len(predictions),
        )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _distribution(seed: bytes, count: int) -> tuple[float, ...]:
    weights = []
    for index in range(count):
        digest = hashlib.sha256(seed + b"\x00" + str(index).encode("ascii")).digest()
        weights.append(int.from_bytes(digest[:8], "big") + 1)
    total = sum(weights)
    values = [weight / total for weight in weights[:-1]]
    values.append(1.0 - math.fsum(values))
    return tuple(values)
