"""Timed HTTP calls with strict request-relative single and batch validation."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from pydantic import ValidationError

# semantic_cases initializes the source checkout's runtime-contract path.
# isort: off
from .semantic_cases import RequestSpec
from .legacy_projection import project_legacy_preview
from .transport import Endpoint, OPENER, _consume, validate_endpoint_url
from decision_runtime.contracts import (
    ResponseContractError,
    SystemOneBatchRequest,
    SystemOneBatchResponse,
    SystemOneResponse,
    validate_batch_response_for_request,
    validate_response_for_request,
)

# isort: on

HTTP_OK = 200


def batch_url(single_url: str) -> str:
    parts = urlsplit(validate_endpoint_url(single_url))
    return urlunsplit(parts._replace(path="/v1/decision/batches"))


@dataclass(frozen=True)
class HttpSample:
    arm: str
    phase: str
    round: int
    sequence: int
    case_id: str
    concurrency: int
    state_id: str | None
    request_kind: str
    request_sha256: str
    request_bytes: int
    decisions: int
    started_ns: int
    ended_ns: int
    status_code: int | None
    error_code: str | None
    response_sha256: str | None

    @property
    def success(self) -> bool:
        return self.error_code is None

    @property
    def latency_ms(self) -> float:
        return (self.ended_ns - self.started_ns) / 1_000_000

    def public_record(self, origin_ns: int) -> dict[str, object]:
        return {
            "arm": self.arm,
            "phase": self.phase,
            "round": self.round,
            "sequence": self.sequence,
            "case_id": self.case_id,
            "concurrency": self.concurrency,
            "state_id": self.state_id,
            "request_kind": self.request_kind,
            "request_sha256": self.request_sha256,
            "request_bytes": self.request_bytes,
            "decisions": self.decisions,
            "started_offset_ms": (self.started_ns - origin_ns) / 1_000_000,
            "completed_offset_ms": (self.ended_ns - origin_ns) / 1_000_000,
            "latency_ms": self.latency_ms,
            "status_code": self.status_code,
            "success": self.success,
            "error_code": self.error_code,
            "response_sha256": self.response_sha256,
        }


def measure_http(
    endpoint: Endpoint,
    spec: RequestSpec,
    *,
    case_id: str,
    concurrency: int,
    phase: str,
    round_number: int,
    sequence: int,
    timeout_seconds: float,
) -> HttpSample:
    """Time only HTTP send through complete body read, including HTTP failures."""

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if endpoint.token is not None:
        headers["Authorization"] = f"Bearer {endpoint.token}"
    request = urllib.request.Request(
        endpoint.url, data=spec.body, headers=headers, method="POST"
    )
    status_code: int | None = None
    content: bytes | None = None
    response_sha256: str | None = None
    transport_error: str | None = None
    started_ns = time.perf_counter_ns()
    try:
        with OPENER.open(request, timeout=timeout_seconds) as response:
            status_code = response.status
            content, response_sha256 = _consume(response)
    except urllib.error.HTTPError as error:
        status_code = error.code
        with error:
            content, response_sha256 = _consume(error)
    except (urllib.error.URLError, TimeoutError, OSError):
        transport_error = "transport_error"
    ended_ns = time.perf_counter_ns()

    error_code = transport_error
    if error_code is None and status_code != HTTP_OK:
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
                if isinstance(spec.request, SystemOneBatchRequest):
                    if endpoint.response_mode != "decision_v1":
                        raise ValueError("legacy preview mode supports singles only")
                    response = SystemOneBatchResponse.model_validate(parsed)
                    validate_batch_response_for_request(spec.request, response)
                else:
                    if endpoint.response_mode == "legacy_preview":
                        parsed = project_legacy_preview(parsed, spec.request)
                    elif endpoint.response_mode != "decision_v1":
                        raise ValueError("unsupported benchmark response mode")
                    response = SystemOneResponse.model_validate(parsed)
                    validate_response_for_request(spec.request, response)
            except (
                ValidationError,
                ResponseContractError,
                TypeError,
                ValueError,
                OverflowError,
            ):
                error_code = "response_contract"

    return HttpSample(
        arm=endpoint.arm,
        phase=phase,
        round=round_number,
        sequence=sequence,
        case_id=case_id,
        concurrency=concurrency,
        state_id=spec.state_id,
        request_kind=(
            "batch" if isinstance(spec.request, SystemOneBatchRequest) else "single"
        ),
        request_sha256=spec.sha256,
        request_bytes=len(spec.body),
        decisions=spec.decisions,
        started_ns=started_ns,
        ended_ns=ended_ns,
        status_code=status_code,
        error_code=error_code,
        response_sha256=response_sha256,
    )
