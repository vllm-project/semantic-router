"""Scenario acceptance boundaries migrated from separate backend servers."""

import json
from unittest.mock import AsyncMock

import httpx
import pytest
from provider_mocker.app import create_app
from provider_mocker.memory import MemoryScenario
from provider_mocker.settings import Settings


def client_for(scenario, **kwargs):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(
            app=create_app(Settings(scenario=scenario, **kwargs))
        ),
        base_url="http://fixture",
    )


def body(content, model="test-model", **kwargs):
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        **kwargs,
    }


async def test_memory_echo_includes_injected_context_and_model():
    async with client_for("memory", model="memory-fixture") as client:
        request = body("What is my dog's name?", model="memory-fixture")
        request["messages"].insert(
            0, {"role": "system", "content": "Memory context: dog's name is Max"}
        )
        response = await client.post("/v1/chat/completions", json=request)
        assert response.status_code == 200
        assert (
            response.json()["choices"][0]["message"]["content"]
            == "[system]: Memory context: dog's name is Max\n[user]: What is my dog's name?"
        )
        assert response.json()["model"] == "memory-fixture"
        assert (await client.get("/v1/models")).json()["data"][0][
            "id"
        ] == "memory-fixture"


def test_memory_extraction_ignores_system_examples_and_deduplicates():
    content = MemoryScenario().content(
        [
            {
                "role": "system",
                "content": "You are a memory extraction system. Example: budget is $5000 at MIT",
            },
            {"role": "user", "content": "My dog Max is a golden retriever"},
        ]
    )
    assert json.loads(content) == [
        {"type": "semantic", "content": "User's dog's name is Max, a golden retriever"}
    ]


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Which hotel?", "Which hotel? (context: trip to Japan budget $5000)"),
        (
            "Explain a detailed method for allocating computational resources efficiently",
            "Explain a detailed method for allocating computational resources efficiently",
        ),
    ],
)
def test_memory_rewrite_only_adds_relevant_context(query, expected):
    content = MemoryScenario().content(
        [
            {"role": "system", "content": "You are a query rewriter"},
            {
                "role": "user",
                "content": "History:\n[user]: trip to Japan\n[user]: budget $5000\nQuery: "
                + query
                + "\nRewritten query:",
            },
        ]
    )
    assert content == expected


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("Compare the panel responses", "fusion-one-call-answer"),
        ("Synthesize a final answer directly", "fusion-none-answer"),
        ("Structured analysis:", "fusion-separate-answer"),
    ],
)
async def test_looper_judge_branches_and_observation(prompt, expected):
    async with client_for("looper") as client:
        response = await client.post(
            "/v1/chat/completions", json=body(prompt, "fusion-judge")
        )
        assert response.json()["choices"][0]["message"]["content"] == expected
        assert response.json()["usage"]["total_tokens"] == 22
        assert (await client.get("/test/calls")).json() == {
            "calls": {"fusion-judge": 1}
        }
        assert (await client.post("/test/reset")).status_code == 200
        assert (await client.get("/test/calls")).json() == {"calls": {}}


async def test_looper_failure_empty_slow_and_confidence(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr("provider_mocker.looper.asyncio.sleep", sleep)
    async with client_for("looper") as client:
        for model in ("fusion-panel-fail", "fusion-fallback-broken"):
            assert (
                await client.post("/v1/chat/completions", json=body("test", model))
            ).status_code == 502
        empty = await client.post(
            "/v1/chat/completions", json=body("test", "fusion-panel-empty")
        )
        assert (
            empty.json()["choices"] == []
            and empty.json()["usage"]["total_tokens"] == 13
        )
        slow = await client.post(
            "/v1/chat/completions", json=body("test", "fusion-panel-slow")
        )
        assert slow.json()["choices"][0]["message"]["content"] == "too late"
        sleep.assert_awaited_once_with(6)
        for model, expected in (
            ("confidence-model-low", -2.0),
            ("confidence-model-high", -0.1),
        ):
            response = await client.post(
                "/v1/chat/completions", json=body("test", model)
            )
            assert (
                response.json()["choices"][0]["logprobs"]["content"][0]["logprob"]
                == expected
            )
        assert len((await client.get("/test/calls")).json()["calls"]) == 6


@pytest.mark.parametrize(
    "path",
    [
        "/v1/chat/completions",
        "/v1beta/openai/chat/completions",
        "/v1/provider/chat/completions",
        "/v1/chat/chat/completions",
    ],
)
async def test_cli_auth_canary_and_prefixed_paths(path, capsys):
    async with client_for("cli", expected_authorization="canary") as client:
        request = body("hello")
        assert (await client.post(path, json=request)).status_code == 401
        response = await client.post(
            path, json=request, headers={"Authorization": "Bearer canary"}
        )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "ok"
        output = capsys.readouterr().out
        assert "authorization-canary-received" in output and path in output
        assert "Bearer canary" not in output


async def test_demo_toolcall_roundtrip_and_creative_bypass():
    async with client_for("toolcall") as client:
        request = body(
            "Eiffel tower",
            tools=[{"type": "function", "function": {"name": "web_search"}}],
        )
        first = (await client.post("/v1/chat/completions", json=request)).json()
        tool = first["choices"][0]["message"]["tool_calls"][0]
        assert tool["function"]["name"] == "web_search"
        assert first["choices"][0]["finish_reason"] == "tool_calls"
        request["messages"] += [
            first["choices"][0]["message"],
            {
                "role": "tool",
                "tool_call_id": tool["id"],
                "content": "completed in 1889",
            },
        ]
        second = (await client.post("/v1/chat/completions", json=request)).json()
        assert "1887" in second["choices"][0]["message"]["content"]
        creative = (
            await client.post(
                "/v1/chat/completions",
                json=body("write a haiku", tools=request["tools"]),
            )
        ).json()
        assert "tool_calls" not in creative["choices"][0]["message"]
        assert "Bits flow like water" in creative["choices"][0]["message"]["content"]


@pytest.mark.parametrize(
    "prompt,fragment", [("eiffel", "1887"), ("grounded fact", "1889")]
)
async def test_hallucination_keywords(prompt, fragment):
    async with client_for("hallucination") as client:
        response = await client.post("/v1/chat/completions", json=body(prompt))
        assert fragment in response.json()["choices"][0]["message"]["content"]


def test_unknown_scenario_rejected(monkeypatch):
    monkeypatch.setenv("PROVIDER_MOCKER_SCENARIO", "typo")
    with pytest.raises(ValueError, match="unknown PROVIDER_MOCKER_SCENARIO"):
        create_app()
