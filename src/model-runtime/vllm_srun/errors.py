"""Error codes shared by the API, the scheduler and model families.

Request-level errors become an HTTP status with ``{"error": {"code",
"message"}}``; a rejected image or video adds its ``reason`` and, for a
broken limit, the ``limit``. Item-level errors (a question, an input, a
document) are reported in place, for example ``{"type": ..., "error":
<code>}``, and never fail sibling items.
"""

from __future__ import annotations

# Item-level codes (inside a 200 response).
INVALID_QUESTION = "invalid_question"
INVALID_INPUT = "invalid_input"
MAX_LENGTH_EXCEEDED = "max_length_exceeded"
# An input a model reads whole (windows) has more tokens than its scan budget;
# none of it was scanned.
SCAN_BUDGET_EXCEEDED = "scan_budget_exceeded"
INVALID_MODEL_OUTPUT = "invalid_model_output"
DEADLINE_EXCEEDED = "deadline_exceeded"
UNAVAILABLE = "unavailable"

QUESTION_ERROR_CODES = (
    INVALID_QUESTION,
    INVALID_INPUT,
    MAX_LENGTH_EXCEEDED,
    SCAN_BUDGET_EXCEEDED,
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


# Why an image or video was rejected. Clients read these words instead of the
# message, which may name a model or a path.
MEDIA_REASONS = (
    "image_format",
    "image_too_large",
    "image_pixels",
    "image_aspect_ratio",
    "image_malformed",
    "image_unsupported",
    "video_format",
    "video_too_large",
    "video_too_long",
    "video_pixels",
    "video_aspect_ratio",
    "video_malformed",
    "video_tokens",
    "video_unsupported",
)


class MediaError(ValueError):
    """A rejected image or video: its reason and, when it broke one, the limit."""

    def __init__(self, reason: str, message: str, limit: int | None = None):
        if reason not in MEDIA_REASONS:
            raise ValueError(f"unknown media reason {reason!r}")
        super().__init__(message)
        self.reason = reason
        self.limit = limit

    def at(self, where: str) -> MediaError:
        """The same rejection, its message prefixed with where it happened (``videos[1]``)."""
        return MediaError(self.reason, f"{where}: {self}", self.limit)


class RuntimeServiceError(Exception):
    """A request-level failure with a stable code and HTTP status."""

    def __init__(
        self,
        code: str,
        message: str,
        reason: str | None = None,
        limit: int | None = None,
    ):
        if code not in REQUEST_ERROR_STATUS:
            raise ValueError(f"unknown request error code {code!r}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.reason = reason
        self.limit = limit

    @classmethod
    def invalid(cls, exc: ValueError) -> RuntimeServiceError:
        """400 invalid_request for a request that cannot be served; a rejected image or video keeps its reason."""
        if isinstance(exc, MediaError):
            return cls("invalid_request", str(exc), exc.reason, exc.limit)
        return cls("invalid_request", str(exc))

    @property
    def status(self) -> int:
        return REQUEST_ERROR_STATUS[self.code]

    def body(self) -> dict:
        error: dict = {"code": self.code, "message": self.message}
        if self.reason is not None:
            error["reason"] = self.reason
        if self.limit is not None:
            error["limit"] = self.limit
        return {"error": error}


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
