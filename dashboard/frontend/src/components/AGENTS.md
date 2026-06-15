# Dashboard Components Notes

## Scope

- `dashboard/frontend/src/components/**`
- local rules for shared dashboard component hotspots

## Responsibilities

- Component files should keep a single dominant responsibility.
- Treat `ChatComponent.tsx` and `ExpressionBuilder.tsx` as orchestration hotspots that should shed display and helper code into adjacent modules.

## Change Rules

- `ChatComponent.tsx` is the playground orchestration hotspot. Keep network/tool orchestration there, but move display-only cards, citation rendering, toggles, and helper types into adjacent modules.
- Keep message rendering and transport orchestration on different seams when extending `ChatComponent.tsx`; prefer extracting display fragments before adding another async or state branch.
- `ClawRoomChat.tsx` is the OpenClaw realtime hotspot. Keep room/session orchestration there, but move WebSocket/SSE lifecycle helpers, mention parsing, message merge helpers, and sender-formatting support into adjacent hooks or utility modules.
- `ExpressionBuilder.tsx` is a ratcheted hotspot. Keep ReactFlow/container orchestration there, but move AST helpers, parsing/serialization, and display fragments into adjacent support modules when extending it.
- Prefer small presentational components over adding another conditional branch to a large JSX tree.
- If a component already mixes transport, storage, and UI rendering, extract pure display code first when extending it.
