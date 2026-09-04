"""Small credential-free HTTP adapter for evaluation traffic."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests

from cli.evaluation.broker_client import BrokerProtocolError, worker_broker

_HTTP_SUCCESS_MIN = 200
_HTTP_SUCCESS_MAX = 300
_MAX_DIAGNOSTIC_HEADER_LENGTH = 256
_RESPONSE_HEADER_ALLOWLIST = frozenset(
    {
        "x-vsr-selected-model",
        "x-vsr-selected-algorithm",
        "x-vsr-selected-recipe",
        "x-vsr-selected-decision",
    }
)


def _broker_operation(method: str, url: str, track_id: str | None) -> str:
    if method == "GET" and url.endswith("/v1/models"):
        return "models.list"
    if method == "POST" and url.endswith("/v1/chat/completions"):
        if track_id == "model_pool":
            return "arm-chat.completions"
        if track_id in {"joint", "multimodal", "capacity"}:
            return "routed-chat.completions"
        raise BrokerProtocolError("evaluation chat track has no broker operation")
    if method == "POST" and url.endswith("/api/v1/eval?trace=true"):
        return "router.evaluate"
    raise BrokerProtocolError("evaluation requested an unsupported broker operation")


@dataclass(frozen=True)
class HTTPResult:
    success: bool
    status_code: int | None
    payload: dict[str, Any] | None
    latency_ms: float
    headers: dict[str, str]
    error: str | None = None
    broker_receipt: str | None = None
    fetched_at: datetime | None = None


class EvaluationHTTPClient:
    def __init__(
        self,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ):
        self.timeout = timeout
        self.session = session or requests.Session()

    def _broker_request(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None,
        *,
        track_id: str | None,
        case_id: str | None,
        attempt_id: str | None,
        operation: str | None = None,
    ) -> HTTPResult | None:
        broker = worker_broker()
        if broker is None:
            return None
        operation = operation or _broker_operation(method, url, track_id)
        response = broker.request(
            operation,
            payload,
            self.timeout,
            track_id=track_id,
            case_id=case_id,
            attempt_id=attempt_id,
        )
        return HTTPResult(
            success=response["success"],
            status_code=response["status_code"],
            payload=response["payload"],
            latency_ms=float(response["latency_ms"]),
            headers=response["headers"],
            error=response["error"],
            broker_receipt=response["broker_receipt"],
            fetched_at=datetime.fromisoformat(response["fetched_at"]),
        )

    @staticmethod
    def _headers() -> dict[str, str]:
        # Authenticated Dashboard traffic is delegated to the Go-owned broker.
        # The Python worker never resolves SecretRef values or constructs an
        # Authorization header.
        return {"Content-Type": "application/json"}

    @staticmethod
    def _response_headers(response: requests.Response) -> dict[str, str]:
        headers: dict[str, str] = {}
        for raw_name, raw_value in response.headers.items():
            name = str(raw_name).casefold()
            value = str(raw_value)
            if (
                name in _RESPONSE_HEADER_ALLOWLIST
                and len(value) <= _MAX_DIAGNOSTIC_HEADER_LENGTH
                and "\r" not in value
                and "\n" not in value
            ):
                headers[name] = value
        return headers

    def post(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        track_id: str,
        case_id: str,
        attempt_id: str,
    ) -> HTTPResult:
        if broker_result := self._broker_request(
            "POST",
            url,
            payload,
            track_id=track_id,
            case_id=case_id,
            attempt_id=attempt_id,
        ):
            return broker_result
        started = time.perf_counter()
        try:
            response = self.session.post(
                url, json=payload, headers=self._headers(), timeout=self.timeout
            )
            latency_ms = (time.perf_counter() - started) * 1000
            try:
                body = response.json()
            except ValueError:
                body = None
            success = (
                _HTTP_SUCCESS_MIN <= response.status_code < _HTTP_SUCCESS_MAX
                and isinstance(body, dict)
            )
            return HTTPResult(
                success=success,
                status_code=response.status_code,
                payload=body if isinstance(body, dict) else None,
                latency_ms=latency_ms,
                headers=self._response_headers(response),
                error=None if success else f"HTTP {response.status_code}",
            )
        except requests.RequestException as exc:
            return HTTPResult(
                success=False,
                status_code=None,
                payload=None,
                latency_ms=(time.perf_counter() - started) * 1000,
                headers={},
                error=f"request_error:{type(exc).__name__}",
            )

    def get(
        self,
        url: str,
        *,
        track_id: str | None = None,
        case_id: str | None = None,
        attempt_id: str | None = None,
        broker_operation: str | None = None,
    ) -> HTTPResult:
        if broker_result := self._broker_request(
            "GET",
            url,
            None,
            track_id=track_id,
            case_id=case_id,
            attempt_id=attempt_id,
            operation=broker_operation,
        ):
            return broker_result
        started = time.perf_counter()
        try:
            response = self.session.get(
                url, headers=self._headers(), timeout=self.timeout
            )
            latency_ms = (time.perf_counter() - started) * 1000
            try:
                body = response.json()
            except ValueError:
                body = None
            success = (
                _HTTP_SUCCESS_MIN <= response.status_code < _HTTP_SUCCESS_MAX
                and isinstance(body, dict)
            )
            return HTTPResult(
                success=success,
                status_code=response.status_code,
                payload=body if isinstance(body, dict) else None,
                latency_ms=latency_ms,
                headers=self._response_headers(response),
                error=None if success else f"HTTP {response.status_code}",
            )
        except requests.RequestException as exc:
            return HTTPResult(
                success=False,
                status_code=None,
                payload=None,
                latency_ms=(time.perf_counter() - started) * 1000,
                headers={},
                error=f"request_error:{type(exc).__name__}",
            )
