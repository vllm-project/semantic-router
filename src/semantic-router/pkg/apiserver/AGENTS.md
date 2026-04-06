# API Server Package Notes

## Scope

- `src/semantic-router/pkg/apiserver/**`

## Responsibilities

- Keep API-server bootstrap, runtime-registry bridging, and route registration separate from handler-specific transport or runtime helpers.
- Keep config, classification, embedding, memory, vector-store, and file endpoints on narrow registration seams instead of one monolithic server constructor.
- Keep request-time runtime lookups separate from startup-time compatibility wiring.

## Change Rules

- Keep `server.go` focused on startup orchestration and route registration; new runtime dependency lookups, route-family helpers, or compatibility bridges belong in sibling modules.
- `server_test.go` is a ratcheted regression hotspot. Prefer targeted helper builders or focused test files over widening the existing monolithic suite.
- Do not mix API transport changes with deep runtime construction when a `routerruntime` or service seam can own the behavior.
