# TD021: Milvus Lifecycle Logic Is Duplicated Across Runtime Stores

## Status

Closed (retired 2026-04-29): shared seam lives in [`src/semantic-router/pkg/milvus`](../../../src/semantic-router/pkg/milvus/); runtime stores and `extproc` memory wiring use `ConnectGRPC` / `Connect`, `EnsureCollectionLoaded` (+ hooks), and `Retry`; `router_memory` now calls the same connect path. Domain-specific HNSW/schema and MilvusStore transient retry heuristics remain in their packages by design. A future Response API Milvus backend must build on `pkg/milvus`.

## Scope

Milvus-backed runtime stores under `src/semantic-router/pkg/memory/`, `src/semantic-router/pkg/cache/`, `src/semantic-router/pkg/vectorstore/`, and `src/semantic-router/pkg/routerreplay/`

## Summary

Milvus integration logic is currently repeated across several runtime packages instead of being owned by one deep adapter seam. Multiple packages create clients, resolve collection names, bootstrap schemas, create indexes, load collections, and implement retry or flush policy independently. Each package has domain-specific behavior on top, but the backend lifecycle mechanics are largely the same. This duplicates complexity, makes failures inconsistent, and forces backend policy changes to be repeated in multiple places. The same pressure is already visible at the `responsestore` surface, which exposes Milvus as a backend type without a shared runtime implementation to inherit.

## Evidence

- [src/semantic-router/pkg/memory/milvus_store.go](../../../src/semantic-router/pkg/memory/milvus_store.go)
- [src/semantic-router/pkg/cache/milvus_cache.go](../../../src/semantic-router/pkg/cache/milvus_cache.go)
- [src/semantic-router/pkg/cache/hybrid_cache.go](../../../src/semantic-router/pkg/cache/hybrid_cache.go)
- [src/semantic-router/pkg/vectorstore/milvus_backend.go](../../../src/semantic-router/pkg/vectorstore/milvus_backend.go)
- [src/semantic-router/pkg/routerreplay/store/milvus.go](../../../src/semantic-router/pkg/routerreplay/store/milvus.go)
- [src/semantic-router/pkg/responsestore/interface.go](../../../src/semantic-router/pkg/responsestore/interface.go)
- [src/semantic-router/pkg/responsestore/factory.go](../../../src/semantic-router/pkg/responsestore/factory.go)

## Why It Matters

- Backend lifecycle changes such as retry policy, index tuning, connection defaults, or collection loading behavior require repeated edits across unrelated packages.
- Similar failure modes can surface with slightly different handling, logging, or defaults depending on which package reaches Milvus first.
- The repeated setup code obscures the truly domain-specific parts of each store, which makes the packages shallower and harder to reason about.
- Tests are forced to mock or verify the same lifecycle mechanics in several places instead of once behind a shared seam.

## Desired End State

- Shared Milvus lifecycle concerns such as client setup, collection bootstrap, index creation, loading, and retry policy live behind one reusable adapter or helper layer.
- Domain packages keep ownership of schema details, query semantics, and store-specific data translation instead of generic backend lifecycle steps.
- Cross-cutting backend defaults and failure handling are defined once and reused consistently.
- Tests exercise the shared lifecycle seam directly, while domain packages focus on contract-specific behavior.

## Exit Criteria

- A shared Milvus adapter or lifecycle helper is used by the main Go runtime packages that currently duplicate connection/bootstrap logic.
- Collection creation, index creation, loading, and retry policy are no longer independently reimplemented in each package without a strong domain-specific reason.
- Domain package tests shrink their mocked Milvus lifecycle surface and focus on schema/query semantics.
- The repository's architecture docs and local rules point contributors toward the shared backend seam instead of repeating Milvus setup in new packages.
