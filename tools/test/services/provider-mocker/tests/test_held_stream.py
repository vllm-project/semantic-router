"""A held Chat stream sends its first chunk, then waits for the release."""

import asyncio
import unittest

import httpx
from provider_mocker import held_stream
from provider_mocker.app import app


class HeldStreamTest(unittest.IsolatedAsyncioTestCase):
    async def state(self, client, session):
        response = await client.get(f"/debug/held-stream/{session}")
        return response.json()["state"]

    async def test_the_rest_waits_for_the_release(self):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://t"
        ) as client:
            stream = held_stream.hold("held-1", iter(["one", "two", "three"]))
            self.assertEqual(await anext(stream), "one")
            waiting = asyncio.ensure_future(anext(stream))
            await asyncio.sleep(0.05)
            self.assertFalse(waiting.done())
            self.assertEqual(await self.state(client, "held-1"), "holding")
            await client.post("/debug/held-stream/held-1/release")
            self.assertEqual(await waiting, "two")
            self.assertEqual([chunk async for chunk in stream], ["three"])
            self.assertEqual(await self.state(client, "held-1"), "released")

    async def test_a_cancelled_stream_reports_it(self):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://t"
        ) as client:
            stream = held_stream.hold("held-2", iter(["one", "two"]))
            await anext(stream)
            waiting = asyncio.ensure_future(anext(stream))
            await asyncio.sleep(0.05)
            waiting.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiting
            self.assertEqual(await self.state(client, "held-2"), "cancelled")

    async def test_a_held_stream_survives_session_churn(self):
        stream = held_stream.hold("held-churn", iter(["one", "two"]))
        await anext(stream)
        for index in range(held_stream.MAX_SESSIONS + 10):
            held_stream._release_event(f"other-{index}")
        held_stream._release_event("held-churn").set()
        self.assertEqual(await asyncio.wait_for(anext(stream), timeout=1), "two")


if __name__ == "__main__":
    unittest.main()
