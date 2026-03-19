# TD024: OpenClaw Feature Slice Still Collapses Page, Transport, and Proxy Control Boundaries

## Status

Closed

## Scope

OpenClaw frontend pages/components and backend handlers/routes under `dashboard/`

## Summary

The OpenClaw feature is now a maintained product slice, but its implementation still concentrates several responsibilities into oversized frontend and backend hotspots. `OpenClawPage.tsx` mixes route orchestration, fallback graph shaping, modal state, and feature-level composition. `ClawRoomChat.tsx` combines room/session state, mention handling, WebSocket lifecycle, SSE fallback, reconnection, and message projection. On the backend, OpenClaw handlers and routes combine container health, proxy wiring, worker chat fallbacks, room state, and transport-specific recovery logic. Existing debt already tracks the structural size of these files and the broader enterprise-console gaps, but the feature slice still lacks explicit boundary debt for its control, transport, and presentation seams.

## Evidence

- [dashboard/frontend/src/pages/OpenClawPage.tsx](../../../dashboard/frontend/src/pages/OpenClawPage.tsx)
- [dashboard/frontend/src/components/ClawRoomChat.tsx](../../../dashboard/frontend/src/components/ClawRoomChat.tsx)
- [dashboard/frontend/src/components/ChatComponent.tsx](../../../dashboard/frontend/src/components/ChatComponent.tsx)
- [dashboard/backend/router/openclaw_routes.go](../../../dashboard/backend/router/openclaw_routes.go)
- [dashboard/backend/handlers/openclaw_rooms.go](../../../dashboard/backend/handlers/openclaw_rooms.go)
- [dashboard/backend/handlers/openclaw_worker_chat.go](../../../dashboard/backend/handlers/openclaw_worker_chat.go)
- [dashboard/frontend/src/pages/AGENTS.md](../../../dashboard/frontend/src/pages/AGENTS.md)
- [dashboard/frontend/src/components/AGENTS.md](../../../dashboard/frontend/src/components/AGENTS.md)
- [docs/agent/tech-debt/td-005-dashboard-enterprise-console-foundations.md](td-005-dashboard-enterprise-console-foundations.md)
- [docs/agent/tech-debt/td-006-structural-rule-target-vs-legacy-hotspots.md](td-006-structural-rule-target-vs-legacy-hotspots.md)

## Why It Matters

- Feature work in OpenClaw routinely crosses page orchestration, realtime transport, and backend proxy/control logic at once, which raises the cost of even narrow changes.
- UI behavior, transport recovery, and backend worker/proxy fallback semantics are harder to test or evolve independently when they share the same hotspots.
- Existing dashboard local rules do not yet name the OpenClaw hotspots explicitly, so contributors have less guidance near the real feature boundary.
- Structural ratchets alone do not explain which responsibilities should move where, so the repo still needs an architecture-level target for this slice.

## Desired End State

- OpenClaw has clearer slice boundaries between route/page composition, realtime room transport/session handling, and backend worker/proxy control services.
- Frontend page and component shells orchestrate sibling support modules instead of owning graph shaping, transport recovery, and message normalization inline.
- Backend handlers separate proxy/control concerns from room chat or worker-endpoint recovery helpers.
- Dashboard local rules name the OpenClaw hotspots and the extraction-first boundaries expected when they are touched.

## Exit Criteria

- OpenClaw route/page orchestration, realtime room transport, and backend proxy/control logic each have narrower primary owners.
- New room/chat/proxy work no longer requires reopening the same large frontend and backend hotspot files for unrelated responsibilities.
- Dashboard page/component local `AGENTS.md` files explicitly call out the OpenClaw hotspots and their extraction-first rules.
- The worst OpenClaw hotspots no longer need to be treated only as generic structural debt without a feature-slice boundary plan.

## Resolution

- `OpenClawPage.tsx` now acts as the route shell while tab and wizard ownership is split across `OpenClawPageTabs.tsx`, `OpenClawArchitectureTab.tsx`, `OpenClawTeamTab.tsx`, `OpenClawWorkerTab.tsx`, and `OpenClawWorkerProvisionWizard.tsx`, so page orchestration no longer shares a seam with the feature's full section composition.
- `ClawRoomChat.tsx` now delegates transport lifecycle and message support helpers through `useClawRoomTransport.ts` and `clawRoomChatSupport.ts`, separating realtime transport, message normalization, and mention support from the component shell.
- Backend OpenClaw room handling now delegates worker chat proxy behavior through `openclaw_room_worker_chat.go` and automation side effects through `openclaw_room_automation.go`, reducing the responsibility collapse inside `openclaw_rooms.go`.
- Dashboard local rules already named these hotspots, and the extracted files now align the code with those narrower page, transport, and backend-control seams.

## Validation

- `go test ./handlers -run 'Test(OpenClaw|Room|WorkerChat)'` from `dashboard/backend`
- `make dashboard-check`
- `make agent-serve-local ENV=cpu AGENT_STACK_NAME=pl0007-openclaw AGENT_PORT_OFFSET=600`
- `make agent-smoke-local AGENT_STACK_NAME=pl0007-openclaw AGENT_PORT_OFFSET=600`
- `make agent-stop-local ENV=cpu AGENT_STACK_NAME=pl0007-openclaw AGENT_PORT_OFFSET=600`
