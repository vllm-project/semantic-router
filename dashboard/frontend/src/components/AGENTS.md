# Dashboard Components Notes

## Scope

- `dashboard/frontend/src/components/**`
- local rules for shared dashboard component hotspots

## Responsibilities

- Component files should keep a single dominant responsibility.
- Treat `AgentPlayground.tsx` and `ExpressionBuilder.tsx` as orchestration hotspots that should shed display and helper code into adjacent modules.

## Change Rules

- `AgentPlayground.tsx` is the playground composition hotspot. Keep durable session and event transport in `useAgentSessionRuntime`; keep event projection, timeline rendering, target selection, and publication review on separate seams.
- Browser components must not execute Agent tools or persist durable session authority.
  Agent turns, tools, checkpoints, and publication plans belong to the optional Agent
  service contract. Model calls use the public OpenAI-compatible Envoy API; routing
  reads and mutations use the control-plane API. Neither path calls Router internals.
- `ClawRoomChat.tsx` is the OpenClaw realtime hotspot. Keep room/session orchestration there, but move WebSocket/SSE lifecycle helpers, mention parsing, message merge helpers, and sender-formatting support into adjacent hooks or utility modules.
- `ExpressionBuilder.tsx` is a ratcheted hotspot. Keep ReactFlow/container orchestration there, but move AST helpers, parsing/serialization, and display fragments into adjacent support modules when extending it.
- Prefer small presentational components over adding another conditional branch to a large JSX tree.
- If a component already mixes transport, storage, and UI rendering, extract pure display code first when extending it.
