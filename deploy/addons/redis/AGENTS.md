# Redis Addon Notes

## Scope

- `deploy/addons/redis/**`

## Responsibilities

- Keep addon manifest wiring, cache-deployment defaults, and Kubernetes helper code on separate seams.
- Treat Redis addon code as deployment wiring, not as a second owner for runtime cache semantics.
- Keep chart or addon-specific configuration translation separate from dashboard or router runtime handlers.

## Change Rules

- `redis-cache.go` is a ratcheted hotspot. New deployment helpers, manifest fragments, or environment wiring should move into adjacent helpers instead of widening that file.
- Do not mix runtime cache semantics or dashboard transport behavior into addon deployment code.
- If a change touches both addon wiring and runtime cache behavior, keep the runtime contract change in router-core and the deployment projection here.
