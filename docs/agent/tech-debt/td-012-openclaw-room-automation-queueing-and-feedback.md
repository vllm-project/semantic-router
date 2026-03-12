# TD012: OpenClaw Room Automation Queueing and In-Flight Feedback

## Status

Open

## Scope

- `dashboard/backend/handlers/openclaw_rooms.go`
- `dashboard/backend/handlers/openclaw_websocket.go`
- `dashboard/frontend/src/components/ClawRoomChat.tsx`

## Summary

OpenClaw room automation currently serializes all trigger processing behind a per-room mutex and keeps that lock held while worker requests are in flight. When one `@worker` run hangs or stalls, later room messages are persisted but their automation pass waits behind the earlier run. The room UI also does not show a durable in-flight worker state, so queued or slow runs look like missing replies.

## Evidence

- `processRoomUserMessage()` acquires `roomAutomationLock(roomID)` and holds it across the full automation loop and worker round-trips in `dashboard/backend/handlers/openclaw_rooms.go`.
- Room WebSocket sends the user message immediately, then starts automation asynchronously in `dashboard/backend/handlers/openclaw_websocket.go`, so intake and execution are already decoupled at the transport layer but not at the automation scheduler.
- `ClawRoomChat.tsx` stores `streamingMessages`, but the transcript render path uses only persisted `message.content`, so in-flight chunks and placeholders are not surfaced in the room transcript.

## Why It Matters

- A single slow or stuck worker can make later `@worker` requests in the same room appear blocked.
- The current UX makes it hard to distinguish between `worker queued`, `worker running`, and `worker unavailable`.
- Operators end up debugging from backend logs because the room transcript does not expose enough execution state.

## Desired End State

- Room intake remains immediate, but automation scheduling no longer holds a room-wide lock across long worker calls.
- The room transcript shows clear in-flight state for worker replies and clears it deterministically on success or failure.
- Errors identify the failing worker/request without making later room messages appear dropped.

## Exit Criteria

- A user can submit another `@worker` message while a previous worker run is slow, and the system records and schedules it without waiting for the earlier network call to finish.
- The room UI shows an explicit in-flight worker state before the final reply is persisted.
- Backend and frontend tests cover slow worker, failed worker, and back-to-back mention scenarios.
