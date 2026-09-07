# Dashboard frontend

- `App.tsx` owns route registration, auth/setup gates, providers, and layout.
  Page data shaping stays with the owning page or feature.
- Route pages own route-level state; shared components should not acquire page
  navigation or persistence policy.
- Keep transport/session orchestration separate from pure message and result
  rendering in chat components.
- Keep Expression Builder parsing/serialization separate from its ReactFlow UI.
- Reuse canonical config inventories; do not define another signal, plugin, or
  provider schema in a page component.
