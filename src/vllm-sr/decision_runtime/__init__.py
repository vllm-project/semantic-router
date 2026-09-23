"""Backend-neutral Decision SystemOne runtime contracts.

The package root intentionally imports neither the HTTP server nor an inference
framework. Applications can use the protocol and engine without installing the
optional server dependencies.
"""

from .backend import (
    BatchDecisionBackend,
    BackendBatchRequest,
    BackendBatchResult,
    BackendPrediction,
    BackendResult,
    DecisionBackend,
    ModelDescriptor,
)
from .contracts import (
    BatchState,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneBatchRequest,
    SystemOneBatchResponse,
    SystemOneBatchResult,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    validate_batch_response_for_request,
    validate_response_for_request,
)
from .engine import DecisionEngine

__all__ = [
    "BatchDecisionBackend",
    "BackendBatchRequest",
    "BackendBatchResult",
    "BackendPrediction",
    "BackendResult",
    "BatchState",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "DecisionBackend",
    "DecisionEngine",
    "ModelDescriptor",
    "NoulAnswer",
    "NoulQuestion",
    "ScoreAnswer",
    "ScoreQuestion",
    "SystemOneBatchRequest",
    "SystemOneBatchResponse",
    "SystemOneBatchResult",
    "SystemOneRequest",
    "SystemOneResponse",
    "Usage",
    "validate_batch_response_for_request",
    "validate_response_for_request",
]
