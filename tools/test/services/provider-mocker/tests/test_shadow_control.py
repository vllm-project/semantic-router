import asyncio
from http import HTTPStatus
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from provider_mocker.app import app, create_app
from provider_mocker.settings import Settings

PRIMARY_MODEL = "openai/gpt-oss-20b"


@pytest.fixture
def controlled_app() -> FastAPI:
    return create_app(Settings(shadow_control=True))


def chat_body(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Please say hello."}],
    }


@pytest.mark.asyncio
async def test_primary_completes_while_shadow_barrier_is_held(
    controlled_app: FastAPI,
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=controlled_app),
        base_url="http://simulator",
    ) as client:
        reset = await client.post("/debug/shadow/queue/reset", json={"mode": "hold"})
        assert reset.status_code == HTTPStatus.OK
        state = controlled_app.state.shadow_control.state("queue")
        shadow = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                json=chat_body(model="openai/shadow-queue"),
                headers={"x-request-id": "fixture-shadow-request"},
            )
        )
        try:
            await asyncio.wait_for(state.started.wait(), timeout=2)
            status = await client.get("/debug/shadow/queue")
            assert status.json() == {
                "mode": "hold",
                "received": 1,
                "active": 1,
                "expired": 0,
                "request_ids": ["fixture-shadow-request"],
            }
            assert not shadow.done()
            primary = await asyncio.wait_for(
                client.post(
                    "/v1/chat/completions", json=chat_body(model=PRIMARY_MODEL)
                ),
                timeout=2,
            )
            assert primary.status_code == HTTPStatus.OK
            assert primary.json()["model"] == PRIMARY_MODEL
            assert primary.json()["choices"][0]["message"]["content"] == (
                f"Hello from {PRIMARY_MODEL}."
            )
            assert not shadow.done()
            reset = await client.post(
                "/debug/shadow/queue/reset", json={"mode": "healthy"}
            )
            assert reset.status_code == HTTPStatus.CONFLICT
            released = await client.post("/debug/shadow/queue/release")
            assert released.status_code == HTTPStatus.OK
            response = await asyncio.wait_for(shadow, timeout=2)
            assert response.status_code == HTTPStatus.OK
            assert response.json()["choices"][0]["message"]["content"] == (
                "Hello from openai/shadow-queue."
            )
            status = await client.get("/debug/shadow/queue")
            assert status.json()["active"] == 0
            assert status.json()["expired"] == 0
        finally:
            state.release.set()
            await asyncio.wait_for(shadow, timeout=2)


@pytest.mark.asyncio
async def test_malformed_shadow_shape_recovers_on_release(
    controlled_app: FastAPI,
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=controlled_app),
        base_url="http://simulator",
    ) as client:
        reset = await client.post(
            "/debug/shadow/malformed/reset", json={"mode": "malformed"}
        )
        assert reset.status_code == HTTPStatus.OK
        response = await client.post(
            "/v1/chat/completions",
            json=chat_body(model="openai/shadow-malformed"),
        )
        assert response.status_code == HTTPStatus.OK
        assert response.json() == {"choices": {"message": {"content": "Hello."}}}
        released = await client.post("/debug/shadow/malformed/release")
        assert released.status_code == HTTPStatus.OK
        recovered = await client.post(
            "/v1/chat/completions",
            json=chat_body(model="openai/shadow-malformed"),
        )
        assert recovered.status_code == HTTPStatus.OK
        assert recovered.json()["choices"][0]["message"]["content"] == (
            "Hello from openai/shadow-malformed."
        )
        status = await client.get("/debug/shadow/malformed")
        assert status.json()["received"] == 2
        assert status.json()["active"] == 0
        assert status.json()["expired"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenario", "payload", "expected_status"),
    [
        ("unknown", {"mode": "healthy"}, HTTPStatus.NOT_FOUND),
        ("timeout", {"mode": "unknown"}, HTTPStatus.UNPROCESSABLE_ENTITY),
        (
            "timeout",
            {"mode": "healthy", "extra": True},
            HTTPStatus.UNPROCESSABLE_ENTITY,
        ),
    ],
)
async def test_shadow_control_rejects_unknown_configuration(
    controlled_app: FastAPI,
    scenario: str,
    payload: dict[str, Any],
    expected_status: HTTPStatus,
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=controlled_app),
        base_url="http://simulator",
    ) as client:
        response = await client.post(f"/debug/shadow/{scenario}/reset", json=payload)
        assert response.status_code == expected_status


@pytest.mark.asyncio
async def test_default_simulator_does_not_enable_shadow_control() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://simulator"
    ) as client:
        response = await client.get("/debug/shadow/timeout")
        assert response.status_code == HTTPStatus.NOT_FOUND
        primary = await client.post(
            "/v1/chat/completions", json=chat_body(model=PRIMARY_MODEL)
        )
        assert primary.status_code == HTTPStatus.OK
        assert (
            '"mock":"provider-mocker"'
            in primary.json()["choices"][0]["message"]["content"]
        )
