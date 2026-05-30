# TD031: Router Runtime Bootstrap and Shared Service Registry Still Depend on Process-Wide Globals

## Status

Open

## Owner Plan

PL0033 v0.3 Themis Release Closure

## Release Relevance

v0.3 Themis

## Scope

`src/semantic-router/cmd/{main.go,runtime_bootstrap.go}`,
`src/semantic-router/pkg/routerruntime/**`,
`src/semantic-router/pkg/apiserver/**`,
`src/semantic-router/pkg/extproc/**`,
`src/semantic-router/pkg/services/classification.go`, and adjacent runtime
bootstrap or reload seams.

## Summary

The main runtime path is now substantially better than the older package-global
shape: config, classification, memory, vector-store, file-store, embedder,
ingestion-pipeline, and model-selection dependencies can flow through
`pkg/routerruntime` for startup, API, ExtProc, and reload paths.

The debt that remains is narrower but still release-relevant. Several
non-registry entrypoints and fallback helpers still expose process-wide config,
classification, memory, and selector state. `runtime_bootstrap.go` also remains
the broad place where tracing, metrics, controller startup, service publication,
and reload coordination meet.

## Evidence

- [src/semantic-router/cmd/main.go](../../../src/semantic-router/cmd/main.go)
- [src/semantic-router/cmd/runtime_bootstrap.go](../../../src/semantic-router/cmd/runtime_bootstrap.go)
- [src/semantic-router/pkg/config/loader.go](../../../src/semantic-router/pkg/config/loader.go)
- [src/semantic-router/pkg/routerruntime/registry.go](../../../src/semantic-router/pkg/routerruntime/registry.go)
- [src/semantic-router/pkg/routerruntime/vectorstore_runtime.go](../../../src/semantic-router/pkg/routerruntime/vectorstore_runtime.go)
- [src/semantic-router/pkg/extproc/router_build.go](../../../src/semantic-router/pkg/extproc/router_build.go)
- [src/semantic-router/pkg/extproc/router_runtime_services.go](../../../src/semantic-router/pkg/extproc/router_runtime_services.go)
- [src/semantic-router/pkg/extproc/server.go](../../../src/semantic-router/pkg/extproc/server.go)
- [src/semantic-router/pkg/extproc/server_config_watch.go](../../../src/semantic-router/pkg/extproc/server_config_watch.go)
- [src/semantic-router/pkg/apiserver/server.go](../../../src/semantic-router/pkg/apiserver/server.go)
- [src/semantic-router/pkg/apiserver/runtime_config.go](../../../src/semantic-router/pkg/apiserver/runtime_config.go)
- [src/semantic-router/pkg/apiserver/route_vectorstore.go](../../../src/semantic-router/pkg/apiserver/route_vectorstore.go)
- [src/semantic-router/pkg/apiserver/route_files.go](../../../src/semantic-router/pkg/apiserver/route_files.go)
- [src/semantic-router/pkg/apiserver/route_feedback.go](../../../src/semantic-router/pkg/apiserver/route_feedback.go)
- [src/semantic-router/pkg/selection/selector.go](../../../src/semantic-router/pkg/selection/selector.go)
- [src/semantic-router/pkg/services/classification.go](../../../src/semantic-router/pkg/services/classification.go)
- [src/semantic-router/pkg/memory/store.go](../../../src/semantic-router/pkg/memory/store.go)

## Why It Matters

- Runtime-owned paths are easier to test and recover when dependencies are
  passed through explicit registries or constructors.
- Process-wide fallback accessors make multi-stack tests and partial-service
  recovery harder because callers can bypass the active runtime graph.
- Startup, reload, and service publication still share too much coordination
  logic in a small set of bootstrap files.

## Desired End State

- Runtime startup is expressed as an explicit composition root with typed
  component ownership and lifecycle hooks.
- API server, ExtProc, vector-store/file-store APIs, feedback/ratings/RL-state
  APIs, classification service, and memory service receive steady-state
  dependencies through constructors or a narrow runtime registry.
- Config update fanout remains explicit while runtime-owned paths stop relying
  on process-wide config publication for steady-state behavior.
- Startup, shutdown, and reload each have narrow responsibilities and focused
  tests for partial dependency availability.

## Exit Criteria

- `runtime_bootstrap.go` delegates service construction and lifecycle to
  narrower helpers or runtime components.
- `apiserver` and `extproc` do not depend on package-global store, embedder,
  pipeline, model-selector, config, or classification-service setters for
  steady-state runtime behavior.
- Runtime-owned reload and service-update paths publish through the runtime
  graph instead of process-wide state.
- Targeted tests exercise bootstrap ordering, degraded startup, and runtime
  reload behavior without mutating process-wide globals across unrelated
  packages.
