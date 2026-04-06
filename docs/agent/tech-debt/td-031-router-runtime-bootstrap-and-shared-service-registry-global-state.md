# TD031: Router Runtime Bootstrap and Shared Service Registry Still Depend on Process-Wide Globals

## Status

Closed

## Scope

`src/semantic-router/cmd/{main.go,runtime_bootstrap.go}`, `src/semantic-router/pkg/config/loader.go`, `src/semantic-router/pkg/routerruntime/**`, `src/semantic-router/pkg/apiserver/**`, `src/semantic-router/pkg/services/classification.go`, and adjacent runtime bootstrap or reload seams

## Summary

The common runtime path no longer suffers from the stale API-server config snapshot tracked by TD011, the single-consumer config-update channel, or implicit publication of classification, memory, vector-store, file-store, embedder, and ingestion-pipeline dependencies through constructor side effects or `cmd`-owned package globals. Those steady-state dependencies now flow through `pkg/routerruntime`, and the remaining compatibility globals are quarantined behind a narrow legacy API-server bootstrap adapter instead of being consulted during steady-state request handling or runtime config refresh.

## Evidence

- [src/semantic-router/cmd/main.go](../../../src/semantic-router/cmd/main.go)
- [src/semantic-router/cmd/runtime_bootstrap.go](../../../src/semantic-router/cmd/runtime_bootstrap.go)
- [src/semantic-router/pkg/config/loader.go](../../../src/semantic-router/pkg/config/loader.go)
- [src/semantic-router/pkg/routerruntime/registry.go](../../../src/semantic-router/pkg/routerruntime/registry.go)
- [src/semantic-router/pkg/routerruntime/vectorstore_runtime.go](../../../src/semantic-router/pkg/routerruntime/vectorstore_runtime.go)
- [src/semantic-router/pkg/apiserver/server.go](../../../src/semantic-router/pkg/apiserver/server.go)
- [src/semantic-router/pkg/apiserver/legacy_runtime_registry.go](../../../src/semantic-router/pkg/apiserver/legacy_runtime_registry.go)
- [src/semantic-router/pkg/apiserver/runtime_config.go](../../../src/semantic-router/pkg/apiserver/runtime_config.go)
- [src/semantic-router/pkg/apiserver/route_vectorstore.go](../../../src/semantic-router/pkg/apiserver/route_vectorstore.go)
- [src/semantic-router/pkg/apiserver/route_files.go](../../../src/semantic-router/pkg/apiserver/route_files.go)
- [src/semantic-router/pkg/services/classification.go](../../../src/semantic-router/pkg/services/classification.go)
- [src/semantic-router/pkg/services/classification_runtime_config.go](../../../src/semantic-router/pkg/services/classification_runtime_config.go)
- [src/semantic-router/pkg/extproc/router_runtime_services.go](../../../src/semantic-router/pkg/extproc/router_runtime_services.go)
- [src/semantic-router/pkg/apiserver/runtime_state_test.go](../../../src/semantic-router/pkg/apiserver/runtime_state_test.go)
- [src/semantic-router/pkg/services/classification_update_test.go](../../../src/semantic-router/pkg/services/classification_update_test.go)
- [src/semantic-router/pkg/extproc/server_reload_test.go](../../../src/semantic-router/pkg/extproc/server_reload_test.go)
- [src/semantic-router/pkg/memory/store.go](../../../src/semantic-router/pkg/memory/store.go)
- [docs/agent/tech-debt/td-011-apiserver-runtime-state-split.md](td-011-apiserver-runtime-state-split.md)

## Why It Matters

- The common startup and reload path is now explicit enough for classification, memory, vector-store, file-store, and config-fanout ownership, but the remaining compatibility globals still make it harder to reason about which code paths are fully registry-backed versus legacy fallback.
- Process-wide compatibility accessors still make multi-stack tests and partial-service recovery harder than they need to be because callers can still bypass the registry and touch package-global state directly.
- `config.Replace`/`config.Get` still form a broad shared contract across startup, reload, and request-time code, so the repo still lacks a complete composition root for all runtime-owned state.
- As the repo adds more deployment modes, background services, and observability hooks, this hidden registry becomes a concurrency and maintenance bottleneck rather than a simple bootstrap shortcut.

## Desired End State

- Runtime startup is expressed as an explicit composition root with typed component ownership and lifecycle hooks instead of package-global setters.
- API server, vector-store/file-store APIs, classification service, and memory service receive their dependencies through constructors or a narrow runtime registry object rather than through process-wide globals.
- Config updates support multiple consumers through an explicit fanout or subscription seam instead of a single implicit watcher.
- Startup, shutdown, and reload each have narrow responsibilities and can be tested independently for partial dependency availability.

## Exit Criteria

- `runtime_bootstrap.go` delegates service construction and lifecycle to narrower helpers or runtime components instead of remaining the default owner for cross-subsystem startup.
- `apiserver` no longer depends on package-global store/embedder/pipeline setters for its steady-state runtime behavior.
- Config reload and service-update paths support more than one observer without relying on a single global channel.
- Targeted tests can exercise bootstrap ordering, degraded startup, and runtime reload behavior without mutating process-wide globals across unrelated packages.

## Retirement Notes

- `apiserver.Init` now bridges legacy globals through [legacy_runtime_registry.go](../../../src/semantic-router/pkg/apiserver/legacy_runtime_registry.go) and then runs the same registry-backed startup path as `InitWithRuntime`; the steady-state server no longer falls back to `config.Get()`, `GetGlobalClassificationService()`, `GetGlobalMemoryStore()`, or package-global vector/file-store handles.
- `runtime_dependencies.go`, `route_vectorstore.go`, and `route_files.go` now treat package-global vector/file-store setters as legacy compatibility snapshots only; request-time API handlers resolve runtime services from `routerruntime.Registry`.
- `liveRuntimeConfig.Update`, `ClassificationService.UpdateConfig`, and `ClassificationService.RefreshRuntimeConfig` no longer write back through `config.Replace`, which keeps request-time and API-driven config refresh local to the explicit runtime owner.
- `extproc` request and startup paths now prefer runtime-owned config through `Server.currentRuntimeConfig()` and `OpenAIRouter.currentConfig()` instead of reading `config.Get()` directly in steady-state request handling.
- Targeted regressions now cover the legacy registry bridge, registry-backed shared dependency resolution, no-global-write runtime config refresh, and extproc config preference for the live runtime registry.

## Validation

- `cd /Users/bitliu/vs/src/semantic-router && go test ./pkg/apiserver ./pkg/services ./pkg/extproc ./pkg/routerruntime`
- `make agent-validate`
- `make agent-lint AGENT_CHANGED_FILES_PATH=/tmp/vsr_td031_changed.txt`
- `make agent-ci-gate AGENT_CHANGED_FILES_PATH=/tmp/vsr_td031_changed.txt`
