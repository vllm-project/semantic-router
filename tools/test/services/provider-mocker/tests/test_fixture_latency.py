"""Controlled fixture work makes latency aggregation observable without timing races."""

import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from provider_mocker import settings as mock_provider
from provider_mocker.app import app


class FixtureLatencyTests(unittest.IsolatedAsyncioTestCase):
    async def request(self):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://fixture",
        ) as client:
            return await client.post(
                "/v1/chat/completions",
                json={
                    "model": "fixture",
                    "messages": [{"role": "user", "content": "latency fixture"}],
                },
            )

    async def test_default_has_no_added_work_and_retains_usage(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(
                mock_provider, "sleep_for_fixture", new_callable=AsyncMock
            ) as sleep,
        ):
            response = await self.request()
        self.assertEqual(response.status_code, 200)
        sleep.assert_not_awaited()
        self.assertGreater(response.json()["usage"]["total_tokens"], 0)

    async def test_selected_latency_is_awaited_before_the_real_response(self):
        with (
            patch.dict(os.environ, {"PROVIDER_MOCKER_RESPONSE_DELAY_MS": "5"}),
            patch.object(
                mock_provider, "sleep_for_fixture", new_callable=AsyncMock
            ) as sleep,
        ):
            response = await self.request()
        self.assertEqual(response.status_code, 200)
        sleep.assert_awaited_once_with(0.005)
        self.assertEqual(response.json()["model"], "fixture")

    async def test_invalid_negative_fixture_work_is_rejected(self):
        with (
            patch.dict(os.environ, {"PROVIDER_MOCKER_RESPONSE_DELAY_MS": "-1"}),
            self.assertRaisesRegex(ValueError, "non-negative"),
        ):
            await mock_provider.apply_fixture_delay()
