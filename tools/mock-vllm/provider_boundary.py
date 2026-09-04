"""HTTP request boundary and bounded request observation for the simulator."""

from collections import OrderedDict
from copy import deepcopy
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from provider_contract import ContractViolationError, validate_provider_request

SESSION_HEADER = "x-vsr-test-session-id"
_MAX_REQUEST_STORE_SESSIONS = 32

router = APIRouter()


class RequestStore:
    def __init__(self) -> None:
        self._store: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def record(self, session_id: str, body: dict[str, Any]) -> None:
        if session_id in self._store:
            self._store.move_to_end(session_id)
        elif len(self._store) >= _MAX_REQUEST_STORE_SESSIONS:
            self._store.popitem(last=False)
        self._store[session_id] = deepcopy(body)

    def get(self, session_id: str) -> dict[str, Any] | None:
        body = self._store.get(session_id)
        if body is not None:
            self._store.move_to_end(session_id)
        return body


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
    body = request.app.state.request_store.get(session_id)
    if body is None:
        return JSONResponse(
            status_code=404,
            content={"error": "not_found", "session_id": session_id},
        )
    return {"session_id": session_id, "body": body}
