"""One identical client-side HTTP timing boundary for both Decision arms."""

from __future__ import annotations

import hashlib
import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlsplit

from .cases import Case

from pydantic import ValidationError

from decision_runtime.contracts import (
    ResponseContractError,
    SystemOneResponse,
    validate_response_for_request,
)

MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


OPENER = urllib.request.build_opener(_NoRedirect)


@dataclass(frozen=True)
class Endpoint:
    arm: str
    url: str
    token: str | None


@dataclass(frozen=True)
class Sample:
    arm: str
    phase: str
    round: int
    sequence: int
    case_id: str
    request_sha256: str
    started_ns: int
    ended_ns: int
    status_code: int | None
    error_code: str | None
    response_sha256: str | None

    @property
    def latency_ms(self) -> float:
        return (self.ended_ns - self.started_ns) / 1_000_000

    @property
    def success(self) -> bool:
        return self.error_code is None

    def public_record(self, origin_ns: int) -> dict[str, object]:
        return {
            "arm": self.arm,
            "phase": self.phase,
            "round": self.round,
            "sequence": self.sequence,
            "case_id": self.case_id,
            "request_sha256": self.request_sha256,
            "started_offset_ms": (self.started_ns - origin_ns) / 1_000_000,
            "completed_offset_ms": (self.ended_ns - origin_ns) / 1_000_000,
            "latency_ms": self.latency_ms,
            "status_code": self.status_code,
            "success": self.success,
            "error_code": self.error_code,
            "response_sha256": self.response_sha256,
        }


def validate_endpoint_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as error:
        raise ValueError("endpoint URL is invalid") from error
    if (
        url.strip() != url
        or parts.scheme not in {"http", "https"}
        or not parts.hostname
        or port == 0
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or parts.path != "/v1/systemone"
    ):
        raise ValueError(
            "endpoint must be an HTTP(S) /v1/systemone URL without credentials or query"
        )
    return url


def _consume(response) -> tuple[bytes | None, str]:
    """Read through EOF, hashing all bytes while bounding retained content."""

    digest = hashlib.sha256()
    chunks: list[bytes] = []
    retained = 0
    oversized = False
    while True:
        chunk = response.read(64 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        if retained + len(chunk) <= MAX_RESPONSE_BYTES:
            chunks.append(chunk)
            retained += len(chunk)
        else:
            oversized = True
    return (None if oversized else b"".join(chunks), digest.hexdigest())


def measure(
    endpoint: Endpoint,
    case: Case,
    *,
    phase: str,
    round_number: int,
    sequence: int,
    timeout_seconds: float,
) -> Sample:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if endpoint.token is not None:
        headers["Authorization"] = f"Bearer {endpoint.token}"
    request = urllib.request.Request(
        endpoint.url, data=case.body, headers=headers, method="POST"
    )
    status_code: int | None = None
    content: bytes | None = None
    response_sha256: str | None = None
    transport_error: str | None = None

    # The measured interval starts immediately before the client sends the
    # request and stops after the complete response body has been consumed.
    # Construction, JSON parsing, and contract checks are outside this interval.
    started_ns = time.perf_counter_ns()
    try:
        with OPENER.open(request, timeout=timeout_seconds) as response:
            status_code = response.status
            content, response_sha256 = _consume(response)
    except urllib.error.HTTPError as error:
        status_code = error.code
        with error:
            content, response_sha256 = _consume(error)
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError):
        transport_error = "transport_error"
    ended_ns = time.perf_counter_ns()

    error_code = transport_error
    if error_code is None and status_code != 200:
        error_code = f"http_{status_code}"
    elif error_code is None and content is None:
        error_code = "response_too_large"
    elif error_code is None:
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, UnicodeDecodeError):
            error_code = "invalid_json"
        else:
            try:
                response = SystemOneResponse.model_validate(parsed)
                validate_response_for_request(case.request, response)
            except (ValidationError, ResponseContractError):
                error_code = "response_contract"

    return Sample(
        arm=endpoint.arm,
        phase=phase,
        round=round_number,
        sequence=sequence,
        case_id=case.id,
        request_sha256=case.sha256,
        started_ns=started_ns,
        ended_ns=ended_ns,
        status_code=status_code,
        error_code=error_code,
        response_sha256=response_sha256,
    )
