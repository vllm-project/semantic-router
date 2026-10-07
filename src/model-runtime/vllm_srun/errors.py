"""Error codes shared by the API, the scheduler and model families.

Request-level errors become an HTTP status with ``{"error": {"code",
"message"}}``. Item-level errors (a question, an input, a document) are
reported in place, for example ``{"type": ..., "error": <code>}``, and never
fail sibling items.
"""

from __future__ import annotations

# Item-level codes (inside a 200 response).
INVALID_QUESTION = "invalid_question"
INVALID_INPUT = "invalid_input"
MAX_LENGTH_EXCEEDED = "max_length_exceeded"
INVALID_MODEL_OUTPUT = "invalid_model_output"
DEADLINE_EXCEEDED = "deadline_exceeded"
UNAVAILABLE = "unavailable"

QUESTION_ERROR_CODES = (
    INVALID_QUESTION,
    INVALID_INPUT,
    MAX_LENGTH_EXCEEDED,
    INVALID_MODEL_OUTPUT,
    DEADLINE_EXCEEDED,
    UNAVAILABLE,
)

# Request-level codes and their HTTP status.
REQUEST_ERROR_STATUS = {
    "invalid_request": 400,
    "model_not_found": 404,
    "request_too_large": 413,
    "unsupported_surface": 422,
    "overloaded": 429,
    "internal_error": 500,
    "not_ready": 503,
}


class RuntimeServiceError(Exception):
    """A request-level failure with a stable code and HTTP status."""

    def __init__(self, code: str, message: str):
        if code not in REQUEST_ERROR_STATUS:
            raise ValueError(f"unknown request error code {code!r}")
        super().__init__(message)
        self.code = code
        self.message = message

    @property
    def status(self) -> int:
        return REQUEST_ERROR_STATUS[self.code]

    def body(self) -> dict:
        return {"error": {"code": self.code, "message": self.message}}


class QuestionError(ValueError):
    """A question that cannot be answered; the code is returned in its answer."""

    def __init__(self, code: str, message: str = ""):
        if code not in QUESTION_ERROR_CODES:
            raise ValueError(f"unknown question error code {code!r}")
        super().__init__(message or code)
        self.code = code


def question_error(
    kind: str | None, error: QuestionError, code: str | None = None
) -> dict:
    """A failed question's answer: its type, error code and, if known, why."""
    answer: dict = {"type": kind, "error": code or error.code}
    reason = str(error)
    if reason and reason != error.code:
        answer["message"] = reason
    return answer


class PackageError(ValueError):
    """A model package failed verification; nothing from it was executed."""


class VerificationError(RuntimeError):
    """A loaded model whose golden answers are wrong; loading it again gives the same answers."""


class PlacementError(RuntimeError):
    """No device can hold the model under the configured budget."""


class UnsupportedDeviceError(PlacementError):
    """Every device the model may use lacks a capability it requires; loading again changes nothing."""
