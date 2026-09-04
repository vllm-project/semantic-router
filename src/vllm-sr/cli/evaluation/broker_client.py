"""Framed HTTP RPC client for the networkless Dashboard worker.

The sandbox has no socket syscalls. Live probes cross two inherited pipes and
the Dashboard performs only the exact server-owned HTTP operations admitted by
the run manifest.
"""

from __future__ import annotations

import json
import math
import os
import queue
import re
import struct
import threading
from datetime import datetime
from typing import Any

from cli.evaluation.canonical import strict_json_loads
from cli.evaluation.target_contracts import EvaluationTarget

_MAX_FRAME_BYTES = 4 * 1024 * 1024
_MAX_ERROR_BYTES = 256
_MAX_HEADER_BYTES = 256
_MIN_OBJECT_FRAME_BYTES = 2
_MIN_USER_DESCRIPTOR = 3
_MAX_TIMEOUT_SECONDS = 300
_MIN_HTTP_STATUS = 100
_MAX_HTTP_STATUS = 599
_MIN_SUCCESS_STATUS = 200
_MAX_SUCCESS_STATUS_EXCLUSIVE = 300
_MILLISECONDS_PER_SECOND = 1000
_UINT64_MAX = (1 << 64) - 1
_OPERATIONS = frozenset(
    {
        "models.list",
        "arm-chat.completions",
        "routed-chat.completions",
        "router.evaluate",
        "agent-task.ledger",
        "fault-recovery.ledger",
        "hard-policy.ledger",
        "production.experiment-ledger",
    }
)
_OPERATION_TRACKS = {
    "router.evaluate": frozenset({"routing"}),
    "arm-chat.completions": frozenset({"model_pool"}),
    "routed-chat.completions": frozenset({"joint", "multimodal", "capacity"}),
    "agent-task.ledger": frozenset({"agentic"}),
    "fault-recovery.ledger": frozenset({"agentic"}),
    "hard-policy.ledger": frozenset({"safety"}),
    "production.experiment-ledger": frozenset({"preference"}),
}
_READ_ONLY_OPERATIONS = frozenset(
    {
        "models.list",
        "agent-task.ledger",
        "fault-recovery.ledger",
        "hard-policy.ledger",
        "production.experiment-ledger",
    }
)
_RESPONSE_HEADERS = frozenset(
    {
        "x-vsr-selected-model",
        "x-vsr-selected-algorithm",
        "x-vsr-selected-recipe",
        "x-vsr-selected-decision",
    }
)
_EVIDENCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TRACKS = frozenset(
    {
        "routing",
        "model_pool",
        "joint",
        "agentic",
        "multimodal",
        "preference",
        "safety",
        "capacity",
    }
)


class BrokerProtocolError(RuntimeError):
    """The inherited Dashboard broker channel violated its fixed contract."""


class WorkerHTTPBroker:
    """Concurrent request multiplexer over one request and one response pipe."""

    def __init__(self, request_fd: int, response_fd: int):
        if (
            request_fd < _MIN_USER_DESCRIPTOR
            or response_fd < _MIN_USER_DESCRIPTOR
            or request_fd == response_fd
        ):
            raise BrokerProtocolError("worker broker descriptors are invalid")
        self._request_fd = request_fd
        self._response_fd = response_fd
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._next_id = 1
        self._pending: dict[int, queue.Queue[dict[str, Any] | BrokerProtocolError]] = {}
        self._closed_error: BrokerProtocolError | None = None
        self._reader = threading.Thread(
            target=self._read_responses,
            name="evaluation-http-broker",
            daemon=True,
        )
        self._reader.start()

    def request(
        self,
        operation: str,
        payload: dict[str, Any] | None,
        timeout: float,
        *,
        track_id: str | None,
        case_id: str | None,
        attempt_id: str | None,
    ) -> dict[str, Any]:
        _validate_request(operation, payload, track_id, case_id, attempt_id)
        timeout_seconds = _validate_timeout(timeout)
        result_queue: queue.Queue[dict[str, Any] | BrokerProtocolError] = queue.Queue(
            maxsize=1
        )
        request_id = self._publish_request(
            operation,
            payload,
            timeout_seconds,
            track_id,
            case_id,
            attempt_id,
            result_queue,
        )
        return self._await_response(result_queue, request_id, timeout_seconds)

    def _publish_request(
        self,
        operation: str,
        payload: dict[str, Any] | None,
        timeout: float,
        track_id: str | None,
        case_id: str | None,
        attempt_id: str | None,
        result_queue: queue.Queue[dict[str, Any] | BrokerProtocolError],
    ) -> int:
        request_id = 0
        try:
            # Identity allocation and publication share one lock so concurrent
            # callers cannot put request N+1 on the pipe before request N.
            with self._write_lock:
                with self._state_lock:
                    if self._closed_error is not None:
                        raise self._closed_error
                    request_id = self._next_id
                    if request_id > _UINT64_MAX:
                        raise BrokerProtocolError(
                            "worker broker request identity exhausted"
                        )
                    self._next_id += 1
                    self._pending[request_id] = result_queue
                self._write_frame(
                    {
                        "id": request_id,
                        "operation": operation,
                        "track_id": track_id,
                        "case_id": case_id,
                        "attempt_id": attempt_id,
                        "payload": payload,
                        "timeout_ms": max(
                            1, math.ceil(timeout * _MILLISECONDS_PER_SECOND)
                        ),
                    }
                )
        except Exception:
            with self._state_lock:
                if request_id:
                    self._pending.pop(request_id, None)
            raise
        return request_id

    def _await_response(
        self,
        result_queue: queue.Queue[dict[str, Any] | BrokerProtocolError],
        request_id: int,
        timeout: float,
    ) -> dict[str, Any]:
        try:
            response = result_queue.get(timeout=timeout + 5)
        except queue.Empty as exc:
            with self._state_lock:
                self._pending.pop(request_id, None)
            raise BrokerProtocolError(
                "Dashboard HTTP broker response timed out"
            ) from exc
        if isinstance(response, BrokerProtocolError):
            raise response
        return _validate_response(response, request_id)

    def _write_frame(self, value: dict[str, Any]) -> None:
        try:
            body = json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise BrokerProtocolError(
                "worker broker request is not canonical JSON"
            ) from exc
        if not body or len(body) > _MAX_FRAME_BYTES:
            raise BrokerProtocolError("worker broker request exceeds its frame bound")
        frame = struct.pack("!I", len(body)) + body
        _write_all(self._request_fd, frame)

    def _read_responses(self) -> None:
        try:
            while True:
                header = _read_exact(self._response_fd, 4)
                length = struct.unpack("!I", header)[0]
                if length < _MIN_OBJECT_FRAME_BYTES or length > _MAX_FRAME_BYTES:
                    raise BrokerProtocolError(
                        "Dashboard HTTP broker response exceeds its frame bound"
                    )
                body = _read_exact(self._response_fd, length)
                try:
                    value = strict_json_loads(body)
                except (UnicodeDecodeError, ValueError) as exc:
                    raise BrokerProtocolError(
                        "Dashboard HTTP broker returned invalid JSON"
                    ) from exc
                if not isinstance(value, dict):
                    raise BrokerProtocolError(
                        "Dashboard HTTP broker response must be an object"
                    )
                request_id = value.get("id")
                if isinstance(request_id, bool) or not isinstance(request_id, int):
                    raise BrokerProtocolError(
                        "Dashboard HTTP broker response identity is invalid"
                    )
                with self._state_lock:
                    pending = self._pending.pop(request_id, None)
                if pending is not None:
                    pending.put(value)
        except (OSError, BrokerProtocolError) as exc:
            error = (
                exc
                if isinstance(exc, BrokerProtocolError)
                else BrokerProtocolError("Dashboard HTTP broker channel closed")
            )
            with self._state_lock:
                self._closed_error = error
                pending = tuple(self._pending.values())
                self._pending.clear()
            for target in pending:
                target.put(error)


def _read_exact(descriptor: int, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            raise BrokerProtocolError("Dashboard HTTP broker channel closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _write_all(descriptor: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise BrokerProtocolError("Dashboard HTTP broker request write failed")
        remaining = remaining[written:]


def _validate_request(
    operation: str,
    payload: dict[str, Any] | None,
    track_id: str | None,
    case_id: str | None,
    attempt_id: str | None,
) -> None:
    _validate_operation_payload(operation, payload)
    _validate_attempt_binding(operation, track_id, case_id, attempt_id)


def _validate_operation_payload(operation: str, payload: dict[str, Any] | None) -> None:
    if operation not in _OPERATIONS:
        raise BrokerProtocolError("worker broker operation is invalid")
    if operation in _READ_ONLY_OPERATIONS:
        if payload is not None:
            raise BrokerProtocolError(
                "read-only broker operations cannot contain a payload"
            )
        return
    if not isinstance(payload, dict):
        raise BrokerProtocolError("worker broker operation requires an object payload")


def _validate_attempt_binding(
    operation: str,
    track_id: str | None,
    case_id: str | None,
    attempt_id: str | None,
) -> None:
    if operation == "models.list":
        if any(value is not None for value in (track_id, case_id, attempt_id)):
            raise BrokerProtocolError("model discovery cannot bind an evidence attempt")
        return
    if (
        track_id not in _TRACKS
        or not _valid_evidence_id(case_id)
        or not _valid_evidence_id(attempt_id)
    ):
        raise BrokerProtocolError(
            "worker broker request requires a canonical evidence attempt"
        )
    if track_id not in _OPERATION_TRACKS[operation]:
        raise BrokerProtocolError(
            "worker broker operation does not own this evaluation track"
        )


def _valid_evidence_id(value: str | None) -> bool:
    return isinstance(value, str) and _EVIDENCE_ID.fullmatch(value) is not None


def _validate_timeout(timeout: float) -> float:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise BrokerProtocolError("worker broker timeout is invalid")
    value = float(timeout)
    if not math.isfinite(value) or value <= 0 or value > _MAX_TIMEOUT_SECONDS:
        raise BrokerProtocolError("worker broker timeout is outside its bound")
    return value


def _validate_response(value: dict[str, Any], request_id: int) -> dict[str, Any]:
    expected = {
        "id",
        "success",
        "status_code",
        "payload",
        "latency_ms",
        "fetched_at",
        "headers",
        "error",
        "broker_receipt",
    }
    if set(value) != expected or value.get("id") != request_id:
        raise BrokerProtocolError("Dashboard HTTP broker response shape is invalid")
    success = value.get("success")
    status = value.get("status_code")
    payload = value.get("payload")
    latency = value.get("latency_ms")
    fetched_at = value.get("fetched_at")
    headers = value.get("headers")
    error = value.get("error")
    broker_receipt = value.get("broker_receipt")
    _validate_status_payload(success, status, payload, error)
    _validate_timing(latency, fetched_at)
    _validate_headers(headers)
    _validate_error(error)
    _validate_broker_receipt(broker_receipt)
    return value


def _validate_status_payload(
    success: Any,
    status: Any,
    payload: Any,
    error: Any,
) -> None:
    if not isinstance(success, bool):
        raise BrokerProtocolError("Dashboard HTTP broker success flag is invalid")
    if status is not None and (
        isinstance(status, bool)
        or not isinstance(status, int)
        or not _MIN_HTTP_STATUS <= status <= _MAX_HTTP_STATUS
    ):
        raise BrokerProtocolError("Dashboard HTTP broker status is invalid")
    if payload is not None and not isinstance(payload, dict):
        raise BrokerProtocolError("Dashboard HTTP broker payload is invalid")
    if success and (
        status is None
        or not _MIN_SUCCESS_STATUS <= status < _MAX_SUCCESS_STATUS_EXCLUSIVE
        or payload is None
        or error is not None
    ):
        raise BrokerProtocolError("Dashboard HTTP broker success is inconsistent")
    if not success and error is None:
        raise BrokerProtocolError("Dashboard HTTP broker failure omits its error")


def _validate_timing(latency: Any, fetched_at: Any) -> None:
    if (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(float(latency))
        or latency < 0
    ):
        raise BrokerProtocolError("Dashboard HTTP broker latency is invalid")
    if not isinstance(fetched_at, str):
        raise BrokerProtocolError("Dashboard HTTP broker fetched_at is invalid")
    try:
        parsed_fetched_at = datetime.fromisoformat(fetched_at)
    except ValueError as exc:
        raise BrokerProtocolError(
            "Dashboard HTTP broker fetched_at is invalid"
        ) from exc
    if parsed_fetched_at.tzinfo is None or parsed_fetched_at.utcoffset() is None:
        raise BrokerProtocolError("Dashboard HTTP broker fetched_at is invalid")


def _validate_headers(headers: Any) -> None:
    if (
        not isinstance(headers, dict)
        or not set(headers).issubset(_RESPONSE_HEADERS)
        or any(
            not isinstance(name, str)
            or not isinstance(header, str)
            or len(header.encode("utf-8")) > _MAX_HEADER_BYTES
            or "\r" in header
            or "\n" in header
            for name, header in headers.items()
        )
    ):
        raise BrokerProtocolError("Dashboard HTTP broker headers are invalid")


def _validate_error(error: Any) -> None:
    if error is not None and (
        not isinstance(error, str)
        or not error
        or len(error.encode("utf-8")) > _MAX_ERROR_BYTES
    ):
        raise BrokerProtocolError("Dashboard HTTP broker error is invalid")


def _validate_broker_receipt(broker_receipt: Any) -> None:
    if (
        not isinstance(broker_receipt, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", broker_receipt) is None
    ):
        raise BrokerProtocolError("Dashboard HTTP broker receipt is invalid")


class _BrokerInstallation:
    broker: WorkerHTTPBroker | None = None


_broker_installation = _BrokerInstallation()
_install_lock = threading.Lock()


def install_worker_broker(request_fd: int, response_fd: int) -> None:
    """Install the single inherited broker channel before evaluation starts."""

    with _install_lock:
        if _broker_installation.broker is not None:
            raise BrokerProtocolError("worker HTTP broker is already installed")
        _broker_installation.broker = WorkerHTTPBroker(request_fd, response_fd)


def worker_broker() -> WorkerHTTPBroker | None:
    return _broker_installation.broker


def require_broker_for_authenticated_target(target: EvaluationTarget) -> None:
    """Keep every SecretRef value on the Dashboard side of the worker boundary."""

    if target.credential_refs() and worker_broker() is None:
        raise BrokerProtocolError(
            "authenticated live evaluation requires the Dashboard HTTP broker"
        )
