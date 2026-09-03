"""HTTP transport helpers shared by router calibration workflows."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request

HTTP_OK_MIN = 200
HTTP_REDIRECT_MIN = 300
MANAGEMENT_TOKEN_ENV = "VSR_MGMT_TOKEN"


def normalize_router_url(router_url: str) -> str:
    normalized = router_url.strip().rstrip("/")
    return normalized.removesuffix("/api/v1/eval")


def http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout_seconds: float = 60.0,
) -> tuple[int, dict[str, Any] | list[Any] | str]:
    body = None
    headers = {"Accept": "application/json"}
    token = os.getenv(MANAGEMENT_TOKEN_ENV, "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url=url, method=method.upper(), data=body, headers=headers)
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            status = response.getcode()
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        return exc.code, parsed
    except (error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"request to {url} failed: {exc}") from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw
    return status, parsed


def ensure_success(status: int, payload: Any, action: str) -> Any:
    if HTTP_OK_MIN <= status < HTTP_REDIRECT_MIN:
        return payload
    raise RuntimeError(
        f"{action} failed with status {status}: "
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
