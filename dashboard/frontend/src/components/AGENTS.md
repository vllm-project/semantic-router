# Dashboard Components Notes

## Scope

- `dashboard/frontend/src/components/**`
- local rules for shared dashboard component hotspots

## Responsibilities

- Component files should keep a single dominant responsibility.
- Treat `ChatComponent.tsx` and `ExpressionBuilder.tsx` as orchestration hotspots that should shed display and helper code into adjacent modules.

## Change Rules

- `ChatComponent.tsx` is the playground orchestration hotspot. Keep network/tool orchestration there, but move display-only cards, citation rendering, toggles, and helper types into adjacent modules.
- `ExpressionBuilder.tsx` is a ratcheted hotspot. Keep ReactFlow/container orchestration there, but move AST helpers, parsing/serialization, and display fragments into adjacent support modules when extending it.
- Prefer small presentational components over adding another conditional branch to a large JSX tree.
- If a component already mixes transport, storage, and UI rendering, extract pure display code first when extending it.
