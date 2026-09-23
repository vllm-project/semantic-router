"""Inference-backend protocol for the Decision runtime."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from .contracts import SystemOneRequest


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """Public metadata for one model exposed by the backend."""

    name: str
    description: str
    release_date: str


@dataclass(frozen=True, slots=True)
class BackendPrediction:
    """One ordered probability vector returned by a backend.

    Noul predictions always use ``(P(false), P(true))`` order. Choice and
    Score predictions preserve the request criteria order.
    """

    question_id: str
    type: Literal["noul", "choice", "score"]
    probabilities: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class BackendResult:
    """Backend-native result before public response adaptation."""

    model: str
    predictions: tuple[BackendPrediction, ...]
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class BackendBatchRequest:
    """One identified single-state request admitted to a backend batch."""

    state_id: str
    request: SystemOneRequest


@dataclass(frozen=True, slots=True)
class BackendBatchResult:
    """One identified backend result returned from an atomic batch."""

    state_id: str
    result: BackendResult


@runtime_checkable
class DecisionBackend(Protocol):
    """Minimal async boundary implemented by CUDA, ROCm, or MLX backends."""

    def models(self) -> Sequence[ModelDescriptor]:
        """Return the immutable public model inventory."""

    async def ready(self) -> bool:
        """Return whether the backend can accept inference work."""

    async def infer(self, request: SystemOneRequest) -> BackendResult:
        """Evaluate every question in one validated request."""


@runtime_checkable
class BatchDecisionBackend(DecisionBackend, Protocol):
    """Optional atomic batch capability implemented by batch-aware backends."""

    async def infer_batch(
        self, requests: tuple[BackendBatchRequest, ...]
    ) -> tuple[BackendBatchResult, ...]:
        """Evaluate an identified shared-question batch in one backend call."""


class UnknownModelError(ValueError):
    """The requested model is not served by this runtime."""


class BackendUnavailableError(RuntimeError):
    """The backend cannot currently perform inference."""


class BackendContractError(RuntimeError):
    """The backend returned malformed or mismatched predictions."""
