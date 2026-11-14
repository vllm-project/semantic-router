"""A Chat stream that stops after its first chunk until a test releases it.

A request carrying `__mock_held_stream__` streams its first chunk, then waits
for POST /debug/held-stream/{session}/release before the rest. A client that
sees the first chunk while the stream is held proves the path in between does
not buffer, and GET /debug/held-stream/{session} reports `cancelled` when the
client's disconnect reached this backend before the release.
"""

import asyncio
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterator

from fastapi import APIRouter

MARKER = "__mock_held_stream__"
HOLD_SECONDS = 60
MAX_SESSIONS = 256

router = APIRouter()
_releases: "OrderedDict[str, asyncio.Event]" = OrderedDict()
_states: "OrderedDict[str, str]" = OrderedDict()


def _remember(store: OrderedDict, session: str, value) -> None:
    """Keep the newest sessions, never one whose stream is still held."""
    store[session] = value
    store.move_to_end(session)
    for old in list(store):
        if len(store) <= MAX_SESSIONS:
            break
        if _states.get(old) != "holding":
            del store[old]


def _release_event(session: str) -> asyncio.Event:
    event = _releases.get(session)
    if event is None:
        event = asyncio.Event()
        _remember(_releases, session, event)
    return event


async def hold(session: str, chunks: Iterator[str]) -> AsyncIterator[str]:
    """Yield the first chunk, wait for the release, then yield the rest."""
    release = _release_event(session)
    _remember(_states, session, "holding")
    try:
        yield next(chunks)
        await asyncio.wait_for(release.wait(), timeout=HOLD_SECONDS)
        _remember(_states, session, "released")
        for chunk in chunks:
            yield chunk
    except (asyncio.CancelledError, GeneratorExit):
        _remember(_states, session, "cancelled")
        raise
    except asyncio.TimeoutError:
        _remember(_states, session, "expired")


@router.post("/debug/held-stream/{session}/release")
async def release_held_stream(session: str) -> dict[str, str]:
    _release_event(session).set()
    return {"session": session, "state": _states.get(session, "unknown")}


@router.get("/debug/held-stream/{session}")
async def held_stream_state(session: str) -> dict[str, str]:
    return {"session": session, "state": _states.get(session, "unknown")}
