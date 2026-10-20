"""Error codes shared by the API, the scheduler and model families.

Request-level errors become an HTTP status with ``{"error": {"code",
"message"}}``; a rejected image, video, asset or body adds its ``reason``
and, for a broken limit, the ``limit``. Item-level errors (a question, an
input, a document) are reported in place, for example ``{"type": ...,
"error": <code>, "reason": <word>}``, and never fail sibling items. Reasons
are fixed words that never carry request data, so a gateway may show them
where it must not show a message.
"""

from __future__ import annotations

from typing import Any

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

# Why a question failed, by its code; the first word is the code's default.
QUESTION_REASONS = {
    INVALID_QUESTION: (
        "question_malformed",
        "type_unsupported",
        "box_unsupported",
        "track_unsupported",
        "question_images_unsupported",
        "box_needs_image",
        "box_target_ambiguous",
        "video_needed",
        "video_target_ambiguous",
        "box_video_too_long",
        "at_out_of_range",
        "preset_unknown",
    ),
    INVALID_INPUT: ("input_malformed",),
    MAX_LENGTH_EXCEEDED: ("question_tokens", "require_full_input"),
    SCAN_BUDGET_EXCEEDED: ("scan_budget",),
    INVALID_MODEL_OUTPUT: ("model_output",),
    DEADLINE_EXCEEDED: ("deadline",),
    UNAVAILABLE: ("model_unavailable",),
}

# Request-level codes and their HTTP status.
REQUEST_ERROR_STATUS = {
    "invalid_request": 400,
    "model_not_found": 404,
    "request_too_large": 413,
    "unsupported_media_type": 415,
    "unsupported_surface": 422,
    "overloaded": 429,
    "internal_error": 500,
    "not_ready": 503,
}

# A System One request answered as a whole for one failed question (POST
# /v1/systemone) has the code ``question_failed`` and this status for the
# question's own code.
QUESTION_FAILED = "question_failed"
SYSTEMONE_QUESTION_STATUS = {
    INVALID_QUESTION: 422,
    INVALID_INPUT: 400,
    MAX_LENGTH_EXCEEDED: 400,
    SCAN_BUDGET_EXCEEDED: 400,
    INVALID_MODEL_OUTPUT: 500,
    DEADLINE_EXCEEDED: 504,
    UNAVAILABLE: 503,
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

# Why a request body, its assets or its token count were rejected.
REQUEST_REASONS = (
    "encoding_unsupported",
    "encoding_malformed",
    "multipart_malformed",
    "asset_malformed",
    "asset_unreferenced",
    "asset_reference",
    "asset_uses",
    "request_tokens",
)


def default_reason(code: str) -> str:
    """The reason a failed question gets when its family names none."""
    return QUESTION_REASONS[code][0]


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

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle by reason, message and limit, so a rejection crosses from a media worker process."""
        return (MediaError, (self.reason, str(self), self.limit))


class RequestError(ValueError):
    """A request rejected for its body, its assets or its size: a fixed reason, and the limit it broke."""

    def __init__(self, reason: str, message: str, limit: int | None = None):
        if reason not in REQUEST_REASONS:
            raise ValueError(f"unknown request reason {reason!r}")
        super().__init__(message)
        self.reason = reason
        self.limit = limit


class NoValidQuestionError(ValueError):
    """A request none of whose questions is valid: every failed question's answer, in request order."""

    def __init__(self, message: str, failures: list[tuple[str, dict[str, Any]]]):
        super().__init__(message)
        self.failures = failures


class RuntimeServiceError(Exception):
    """A request-level failure with a stable code and HTTP status.

    ``question`` names the question whose failure answers the request (a
    System One request answered as a whole, ``question_failed`` with the
    question's code as ``question_error``, or one in which no question is
    valid).
    """

    def __init__(
        self,
        code: str,
        message: str,
        reason: str | None = None,
        limit: int | None = None,
        question: str | None = None,
        question_error: str | None = None,
    ):
        if code not in REQUEST_ERROR_STATUS and code != QUESTION_FAILED:
            raise ValueError(f"unknown request error code {code!r}")
        if (code == QUESTION_FAILED) != (question_error in SYSTEMONE_QUESTION_STATUS):
            raise ValueError("question_failed needs the failed question's code")
        super().__init__(message)
        self.code = code
        self.message = message
        self.reason = reason
        self.limit = limit
        self.question = question
        self.question_error = question_error

    @classmethod
    def invalid(cls, exc: ValueError) -> RuntimeServiceError:
        """400 invalid_request for a request that cannot be served; a rejected image, video, asset or body keeps its reason."""
        if isinstance(exc, (MediaError, RequestError)):
            return cls("invalid_request", str(exc), exc.reason, exc.limit)
        if isinstance(exc, NoValidQuestionError) and exc.failures:
            question, answer = exc.failures[0]
            return cls(
                "invalid_request", str(exc), answer.get("reason"), question=question
            )
        return cls("invalid_request", str(exc))

    @property
    def status(self) -> int:
        if self.question_error is not None:
            return SYSTEMONE_QUESTION_STATUS[self.question_error]
        return REQUEST_ERROR_STATUS[self.code]

    def body(self) -> dict:
        error: dict = {"code": self.code, "message": self.message}
        if self.reason is not None:
            error["reason"] = self.reason
        if self.limit is not None:
            error["limit"] = self.limit
        if self.question is not None:
            error["question"] = self.question
        if self.question_error is not None:
            error["question_error"] = self.question_error
        return {"error": error}


class QuestionError(ValueError):
    """A question that cannot be answered; the code and its fixed reason are returned in its answer."""

    def __init__(self, code: str, message: str = "", reason: str | None = None):
        if code not in QUESTION_ERROR_CODES:
            raise ValueError(f"unknown question error code {code!r}")
        if reason is not None and reason not in QUESTION_REASONS[code]:
            raise ValueError(f"unknown reason {reason!r} for {code}")
        super().__init__(message or code)
        self.code = code
        self.reason = reason or default_reason(code)


def question_error(
    kind: str | None, error: QuestionError, code: str | None = None
) -> dict:
    """A failed question's answer: its type, error code, fixed reason and, if known, why."""
    final = code or error.code
    reason = error.reason if final == error.code else default_reason(final)
    answer: dict = {"type": kind, "error": final, "reason": reason}
    message = str(error)
    if message and message != error.code:
        answer["message"] = message
    return answer


def with_reason(answer: dict[str, Any]) -> dict[str, Any]:
    """A failed answer with its code's default reason when it has none (answers built without ``QuestionError``)."""
    code = answer.get("error")
    if code in QUESTION_REASONS and answer.get("reason") not in QUESTION_REASONS[code]:
        answer["reason"] = default_reason(code)
    return answer


class PackageError(ValueError):
    """A model package failed verification; nothing from it was executed."""


class VerificationError(RuntimeError):
    """A loaded model whose golden answers are wrong; loading it again gives the same answers."""


class PlacementError(RuntimeError):
    """No device can hold the model under the configured budget."""


class UnsupportedDeviceError(PlacementError):
    """Every device the model may use lacks a capability it requires; loading again changes nothing."""
