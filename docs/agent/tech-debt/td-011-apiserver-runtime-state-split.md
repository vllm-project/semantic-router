# TD011: API Server Runtime State Is Split Between a Live Service Handle and a Stale Config Snapshot

## Status

Closed

## Scope

semantic-router API server runtime state ownership

## Summary

The API server previously followed router hot-reloads for classification behavior through a live service indirection, but config-sensitive endpoints still served a startup-time `config` snapshot. The server now resolves both reads and writes through one runtime-config seam, so deploy-triggered reloads update classification behavior and config-facing API responses together.

## Evidence

- [src/semantic-router/pkg/apiserver/server.go](../../../src/semantic-router/pkg/apiserver/server.go)
- [src/semantic-router/pkg/apiserver/config.go](../../../src/semantic-router/pkg/apiserver/config.go)
- [src/semantic-router/pkg/apiserver/runtime_config.go](../../../src/semantic-router/pkg/apiserver/runtime_config.go)
- [src/semantic-router/pkg/apiserver/route_models.go](../../../src/semantic-router/pkg/apiserver/route_models.go)
- [src/semantic-router/pkg/apiserver/route_model_info.go](../../../src/semantic-router/pkg/apiserver/route_model_info.go)
- [src/semantic-router/pkg/apiserver/route_system_prompt.go](../../../src/semantic-router/pkg/apiserver/route_system_prompt.go)
- [src/semantic-router/pkg/apiserver/route_config_deploy.go](../../../src/semantic-router/pkg/apiserver/route_config_deploy.go)
- [src/semantic-router/pkg/apiserver/runtime_state_test.go](../../../src/semantic-router/pkg/apiserver/runtime_state_test.go)
- [e2e/testcases/apiserver_runtime_config_endpoints.go](../../../e2e/testcases/apiserver_runtime_config_endpoints.go)
- [src/semantic-router/pkg/extproc/server.go](../../../src/semantic-router/pkg/extproc/server.go)

## Why It Matters

- DSL deploy relies on router hot-reload, so any API path that keeps its own stale config mirror can report or mutate state that no longer matches the active router.
- The current split makes bug fixes asymmetric: classification traffic can be fixed via a live service indirection while config-reporting and config-editing endpoints still read a different source of truth.
- This keeps reload behavior harder to reason about and raises the chance of future regressions when deploy, rollback, and config-edit APIs evolve independently.

## Desired End State

- The API server resolves runtime config and classification behavior from one explicit state owner instead of keeping a long-lived local config snapshot.
- Reload-aware read APIs and config-edit APIs share the same state seam and regression coverage.

## Exit Criteria

- Satisfied on 2026-03-09: `ClassificationAPIServer` no longer serves config-sensitive endpoints from a stale startup snapshot after DSL deploy or router hot-reload.
- Satisfied on 2026-03-09: read-only config/model info endpoints and mutable config endpoints use the same runtime state source.
- Satisfied on 2026-03-09: regression coverage demonstrates that deploy/reload updates both classification behavior and config-facing API responses.

## Resolution

- `src/semantic-router/pkg/apiserver/runtime_config.go` now owns the live runtime-config seam for apiserver reads and writes, with a startup fallback only when no resolver is available.
- `src/semantic-router/pkg/apiserver/route_models.go`, `route_model_info.go`, and `route_system_prompt.go` now resolve runtime config through that seam instead of serving `s.config` directly.
- `src/semantic-router/pkg/apiserver/runtime_state_test.go` adds stale-vs-live regression coverage for model list, classifier info, models info, and system-prompt update flows.
- `e2e/testcases/apiserver_runtime_config_endpoints.go` adds a baseline ai-gateway contract that verifies the live model list and classifier-config endpoints through the deployed router API service.
