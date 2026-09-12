"""HTTP request boundary and bounded request observation for the simulator."""

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from provider_contract import ContractViolationError, validate_provider_request

SESSION_HEADER = "x-vsr-test-session-id"
_OBSERVED_HEADER_PREFIX = "x-vsr-e2e-"
_MAX_REQUEST_STORE_SESSIONS = 32

router = APIRouter()


class RequestStore:
    def __init__(self) -> None:
        self._store: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def record(
        self,
        session_id: str,
        body: dict[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if session_id in self._store:
            self._store.move_to_end(session_id)
        elif len(self._store) >= _MAX_REQUEST_STORE_SESSIONS:
            self._store.popitem(last=False)
        observed_headers = {
            name.lower(): value
            for name, value in (headers or {}).items()
            if name.lower() == SESSION_HEADER
            or name.lower().startswith(_OBSERVED_HEADER_PREFIX)
        }
        self._store[session_id] = {
            "body": deepcopy(body),
            "headers": observed_headers,
        }

    def get(self, session_id: str) -> dict[str, Any] | None:
        observed = self._store.get(session_id)
        if observed is not None:
            self._store.move_to_end(session_id)
            return deepcopy(observed)
        return None


def invalid_request_response(message: str, field: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": field,
                "code": "invalid_request",
            }
        },
    )


async def parse_provider_request(
    request: Request, protocol: str
) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    try:
        body = await request.json()
    except ValueError:
        return None, invalid_request_response("request body is not valid JSON")
    try:
        return validate_provider_request(protocol, body), None
    except ContractViolationError as error:
        return None, invalid_request_response(str(error), error.field)


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/v1/models")
async def models() -> dict[str, list[dict[str, str]]]:
    return {"data": [{"id": "openai/gpt-oss-20b", "object": "model"}]}


@router.get("/debug/last-request")
async def debug_last_request(request: Request):
    session_id = (
        request.headers.get(SESSION_HEADER)
        or request.query_params.get(SESSION_HEADER)
        or "__global__"
    )
    observed = request.app.state.request_store.get(session_id)
    if observed is None:
        return JSONResponse(
            status_code=404,
            content={"error": "not_found", "session_id": session_id},
        )
    return {"session_id": session_id, **observed}
