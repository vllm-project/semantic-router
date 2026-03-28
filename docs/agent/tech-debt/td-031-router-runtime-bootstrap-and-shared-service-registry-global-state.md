# TD031: Router Runtime Bootstrap and Shared Service Registry Still Depend on Process-Wide Globals

## Status

Open

## Scope

`src/semantic-router/cmd/{main.go,runtime_bootstrap.go}`, `src/semantic-router/pkg/config/loader.go`, `src/semantic-router/pkg/apiserver/**`, `src/semantic-router/pkg/services/classification.go`, and adjacent runtime bootstrap or reload seams

## Summary

The router runtime no longer suffers from the stale API-server config snapshot tracked by TD011, but startup and reload behavior still depend on a shallow process-wide service registry. `main.go` and `runtime_bootstrap.go` assemble model download, tracing, metrics, binding initialization, vector-store setup, API-server startup, and Kubernetes-controller startup in one composition chain. Cross-subsystem dependencies are then published through global package state: `config.Replace`/`config.Get`, a single config-update channel, global classification service pointers, global memory store accessors, and apiserver-global vector-store, file-store, embedder, and ingestion-pipeline setters. The API server still retries against those globals during startup, which means sequencing and lifecycle are encoded in start order and package-level state instead of an explicit runtime graph.

## Evidence

- [src/semantic-router/cmd/main.go](../../../src/semantic-router/cmd/main.go)
- [src/semantic-router/cmd/runtime_bootstrap.go](../../../src/semantic-router/cmd/runtime_bootstrap.go)
- [src/semantic-router/pkg/config/loader.go](../../../src/semantic-router/pkg/config/loader.go)
- [src/semantic-router/pkg/apiserver/server.go](../../../src/semantic-router/pkg/apiserver/server.go)
- [src/semantic-router/pkg/apiserver/runtime_config.go](../../../src/semantic-router/pkg/apiserver/runtime_config.go)
- [src/semantic-router/pkg/apiserver/route_vectorstore.go](../../../src/semantic-router/pkg/apiserver/route_vectorstore.go)
- [src/semantic-router/pkg/apiserver/route_files.go](../../../src/semantic-router/pkg/apiserver/route_files.go)
- [src/semantic-router/pkg/services/classification.go](../../../src/semantic-router/pkg/services/classification.go)
- [src/semantic-router/pkg/memory/store.go](../../../src/semantic-router/pkg/memory/store.go)
- [docs/agent/tech-debt/td-011-apiserver-runtime-state-split.md](td-011-apiserver-runtime-state-split.md)

## Why It Matters

- Startup and reload sequencing remain implicit. Adding a new runtime-owned service still tends to touch bootstrap, one or more global setter/getter packages, and API-server startup order in the same change.
- Process-wide globals make multi-stack tests, hot reload, and partial-service recovery harder because dependencies cannot be constructed, swapped, or observed in isolation.
- The single watcher channel in `config/loader.go` is an especially weak boundary for future observability, live-reload, and sidecar-style consumers because it encodes one-consumer assumptions into the core config path.
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
