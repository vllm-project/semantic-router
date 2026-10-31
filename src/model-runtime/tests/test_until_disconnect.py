"""A disconnect and the work must not race each other inside one turn."""

import asyncio
import gc
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from starlette.requests import Request
from vllm_srun.api.app import CLIENT_CLOSED, create_app, until_disconnect


def _request(receive):
    """A request whose body was read already, as the routes read it."""
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/classify",
            "headers": [],
            "query_string": b"",
            "root_path": "",
        },
        receive,
    )


def _disconnecting():
    """A receive that reports the client's disconnect at once."""

    async def receive():
        return {"type": "http.disconnect"}

    return receive


def _staying():
    """A receive that never reports a disconnect."""

    async def receive():
        await asyncio.Event().wait()

    return receive


def test_a_disconnect_in_the_same_turn_as_the_work_wins():
    async def scenario():
        async def work():
            return "answer"

        return await until_disconnect(_request(_disconnecting()), work())

    assert asyncio.run(scenario()) is None


def test_a_disconnect_in_the_same_turn_as_a_failed_work_wins():
    async def scenario():
        async def work():
            raise RuntimeError("the work failed")

        return await until_disconnect(_request(_disconnecting()), work())

    assert asyncio.run(scenario()) is None


def test_the_work_result_is_kept_when_the_client_stays():
    async def scenario():
        async def work():
            return "answer"

        return await until_disconnect(_request(_staying()), work())

    assert asyncio.run(scenario()) == "answer"


def test_the_work_failure_is_raised_when_the_client_stays():
    async def scenario():
        async def work():
            raise RuntimeError("the work failed")

        return await until_disconnect(_request(_staying()), work())

    with pytest.raises(RuntimeError, match="the work failed"):
        asyncio.run(scenario())


def test_a_disconnect_cancels_the_work():
    cancelled = []

    async def scenario():
        async def work():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.append("cancelled")
                raise

        return await until_disconnect(_request(_disconnecting()), work())

    assert asyncio.run(scenario()) is None
    assert cancelled == ["cancelled"]


def test_a_disconnect_does_not_wait_for_work_that_suppresses_cancellation():
    async def scenario():
        async def work():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                await asyncio.sleep(60)
                return "late"

        started = time.perf_counter()
        outcome = await until_disconnect(_request(_disconnecting()), work())
        return outcome, time.perf_counter() - started

    outcome, elapsed = asyncio.run(scenario())
    assert outcome is None
    assert elapsed < 5


def test_a_discarded_work_failure_is_not_reported_as_unretrieved():
    seen = []

    async def scenario():
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, context: seen.append(context))

        async def work():
            raise RuntimeError("the work failed")

        assert await until_disconnect(_request(_disconnecting()), work()) is None
        gc.collect()
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert [context.get("message") for context in seen] == []


def test_a_disconnect_in_the_same_turn_as_the_work_answers_499():
    async def scenario():
        async def call(*args):
            return 200, {"labels": []}

        runtime = SimpleNamespace(
            config=SimpleNamespace(max_request_bytes=1024),
            metrics=MagicMock(),
            call=call,
            bundle=call,
        )
        messages = [{"type": "http.request", "body": b"{}", "more_body": False}]

        async def receive():
            if messages:
                return messages.pop()
            return {"type": "http.disconnect"}

        sent = []

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/classify",
            "headers": [],
            "query_string": b"",
            "root_path": "",
        }
        await create_app(runtime)(scope, receive, send)
        return runtime, sent

    runtime, sent = asyncio.run(scenario())
    assert sent[0]["status"] == CLIENT_CLOSED
    runtime.metrics.requests.labels.assert_called_with(
        endpoint="/v1/classify", status="499"
    )
