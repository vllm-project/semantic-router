# TD011: API Server Runtime State Is Split Between a Live Service Handle and a Stale Config Snapshot

## Status

Open

## Scope

semantic-router API server runtime state ownership

## Summary

The API server now needs to follow router hot-reloads after DSL deploys, but it still owns two different views of runtime state: a live classification-service handle and a startup-time `config` snapshot. The service side can be refreshed independently; the config side still drifts after deploy-triggered reloads.

## Evidence

- [src/semantic-router/pkg/apiserver/server.go](../../../src/semantic-router/pkg/apiserver/server.go)
- [src/semantic-router/pkg/apiserver/config.go](../../../src/semantic-router/pkg/apiserver/config.go)
- [src/semantic-router/pkg/apiserver/route_models.go](../../../src/semantic-router/pkg/apiserver/route_models.go)
- [src/semantic-router/pkg/apiserver/route_model_info.go](../../../src/semantic-router/pkg/apiserver/route_model_info.go)
- [src/semantic-router/pkg/apiserver/route_system_prompt.go](../../../src/semantic-router/pkg/apiserver/route_system_prompt.go)
- [src/semantic-router/pkg/apiserver/route_config_deploy.go](../../../src/semantic-router/pkg/apiserver/route_config_deploy.go)
- [src/semantic-router/pkg/extproc/server.go](../../../src/semantic-router/pkg/extproc/server.go)

## Why It Matters

- DSL deploy relies on router hot-reload, so any API path that keeps its own stale config mirror can report or mutate state that no longer matches the active router.
- The current split makes bug fixes asymmetric: classification traffic can be fixed via a live service indirection while config-reporting and config-editing endpoints still read a different source of truth.
- This keeps reload behavior harder to reason about and raises the chance of future regressions when deploy, rollback, and config-edit APIs evolve independently.

## Desired End State

- The API server resolves runtime config and classification behavior from one explicit state owner instead of keeping a long-lived local config snapshot.
- Reload-aware read APIs and config-edit APIs share the same state seam and regression coverage.

## Exit Criteria

- `ClassificationAPIServer` no longer serves config-sensitive endpoints from a stale startup snapshot after DSL deploy or router hot-reload.
- Read-only config/model info endpoints and mutable config endpoints use the same runtime state source.
- Regression coverage demonstrates that deploy/reload updates both classification behavior and config-facing API responses.
