import asyncio
from collections import deque
from http import HTTPStatus
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

router = APIRouter()


class ShadowMode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["healthy", "hold", "malformed"]


class ShadowState:
    def __init__(self, mode: str = "healthy") -> None:
        self.mode = mode
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.received = 0
        self.active = 0
        self.expired = 0
        self.request_ids: deque[str] = deque(maxlen=8)

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "received": self.received,
            "active": self.active,
            "expired": self.expired,
            "request_ids": list(self.request_ids),
        }

    async def respond(self, request_id: str) -> JSONResponse | None:
        self.received += 1
        self.request_ids.append(request_id)
        self.active += 1
        self.started.set()
        try:
            if self.mode == "hold":
                try:
                    await asyncio.wait_for(self.release.wait(), timeout=60)
                except TimeoutError:
                    self.expired += 1
                    return JSONResponse(
                        status_code=HTTPStatus.GATEWAY_TIMEOUT,
                        content={"error": "shadow fixture barrier expired"},
                    )
            elif self.mode == "malformed":
                return JSONResponse(
                    content={"choices": {"message": {"content": "Hello."}}}
                )
            return None
        finally:
            self.active -= 1


class ShadowControl:
    def __init__(self) -> None:
        self.states: dict[str, ShadowState] = {
            scenario: ShadowState() for scenario in ("timeout", "malformed", "queue")
        }

    def state(self, scenario: str) -> ShadowState:
        if scenario not in self.states:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND, detail="unknown shadow scenario"
            )
        return self.states[scenario]

    async def respond(self, model: str, request_id: str) -> JSONResponse | None:
        state = (
            self.states.get(model.removeprefix("openai/shadow-"))
            if model.startswith("openai/shadow-")
            else None
        )
        if state is not None:
            return await state.respond(request_id=request_id)
        return None


@router.post("/debug/shadow/{scenario}/reset")
async def reset_shadow(
    scenario: str, mode: ShadowMode, request: Request
) -> dict[str, Any]:
    control: ShadowControl = request.app.state.shadow_control
    if control.state(scenario).active:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT,
            detail="release active shadow requests before resetting",
        )
    state = ShadowState(mode=mode.mode)
    control.states[scenario] = state
    return state.snapshot()


@router.post("/debug/shadow/{scenario}/release")
async def release_shadow(scenario: str, request: Request) -> dict[str, Any]:
    control: ShadowControl = request.app.state.shadow_control
    state = control.state(scenario)
    state.mode = "healthy"
    state.release.set()
    return state.snapshot()


@router.get("/debug/shadow/{scenario}")
async def shadow_status(scenario: str, request: Request) -> dict[str, Any]:
    control: ShadowControl = request.app.state.shadow_control
    return control.state(scenario).snapshot()
