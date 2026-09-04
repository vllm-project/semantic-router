"""Small OpenAI/router HTTP adapter with environment-only credentials."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import requests

_HTTP_SUCCESS_MIN = 200
_HTTP_SUCCESS_MAX = 300
_MAX_DIAGNOSTIC_HEADER_LENGTH = 256
_RESPONSE_HEADER_ALLOWLIST = frozenset(
    {
        "x-vsr-selected-model",
        "x-vsr-selected-algorithm",
    }
)


@dataclass(frozen=True)
class HTTPResult:
    success: bool
    status_code: int | None
    payload: dict[str, Any] | None
    latency_ms: float
    headers: dict[str, str]
    error: str | None = None


class EvaluationHTTPClient:
    def __init__(
        self,
        timeout: float = 30.0,
        session: requests.Session | None = None,
        credential_env: str | None = None,
    ):
        self.timeout = timeout
        self.session = session or requests.Session()
        self.credential_env = credential_env

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv(self.credential_env) if self.credential_env else None
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    @staticmethod
    def _response_headers(response: object) -> dict[str, str]:
        source = getattr(response, "headers", {})
        if not hasattr(source, "items"):
            return {}
        headers: dict[str, str] = {}
        for raw_name, raw_value in source.items():
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

    def post(self, url: str, payload: dict[str, Any]) -> HTTPResult:
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

    def get(self, url: str) -> HTTPResult:
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
