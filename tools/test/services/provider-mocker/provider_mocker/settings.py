"""Explicit, model-free fixture configuration."""

import os
from asyncio import sleep as sleep_for_fixture
from dataclasses import dataclass

SCENARIOS = {"default", "memory", "looper", "hallucination", "toolcall", "cli"}


@dataclass(frozen=True)
class Settings:
    scenario: str = "default"
    model: str = "openai/gpt-oss-20b"
    shadow_control: bool = False
    expected_authorization: str | None = None

    @classmethod
    def from_env(cls):
        scenario = os.getenv("PROVIDER_MOCKER_SCENARIO", "default")
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown PROVIDER_MOCKER_SCENARIO: {scenario}")
        return cls(
            scenario=scenario,
            model=os.getenv("PROVIDER_MOCKER_MODEL", "openai/gpt-oss-20b"),
            shadow_control=os.getenv("PROVIDER_MOCKER_SHADOW_CONTROL") == "true",
            expected_authorization=os.getenv("PROVIDER_MOCKER_EXPECT_AUTHORIZATION"),
        )


async def apply_fixture_delay() -> None:
    delay_ms = int(os.environ.get("PROVIDER_MOCKER_RESPONSE_DELAY_MS", "0"))
    if delay_ms < 0:
        raise ValueError("PROVIDER_MOCKER_RESPONSE_DELAY_MS must be non-negative")
    if delay_ms:
        await sleep_for_fixture(delay_ms / 1000)
