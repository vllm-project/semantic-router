# TD045: Router Runtime Resources Lack Generation Ownership and Bounded Shutdown

## Status

Open

## Owner Plan

PL0032 Architecture Debt Consolidation

## Release Relevance

Production hardening; remediation is split from the #2375 audit proof-of-fix.
Primary implementation tracker:
[#2470](https://github.com/vllm-project/semantic-router/issues/2470), with
bounded actor work in #2471 through #2475 and native contracts in #2396.

## Scope

`src/semantic-router/pkg/extproc`, `routerruntime`, `services`,
`selection`, `looper`, `memory`, `cache`, `vectorstore`,
`routerreplay`, and native runtime ownership seams.

## Summary

Router construction, reload, request admission, background work, and shutdown
do not share one ownership model. The router closes only a subset of its
children, construction has no rollback stack, reload can retire a router while
an ext-proc stream still uses it, and several stores or workers have no bounded
admission/drain protocol.

## Evidence

- [Router ownership and Close](../../../src/semantic-router/pkg/extproc/router.go)
- [Router construction](../../../src/semantic-router/pkg/extproc/router_build.go)
- [Reload and stream delegation](../../../src/semantic-router/pkg/extproc/server.go)
- [Replay stores](../../../src/semantic-router/pkg/routerreplay/store)
- [Vector ingestion pipeline](../../../src/semantic-router/pkg/vectorstore/pipeline.go)
- [Selection registry](../../../src/semantic-router/pkg/selection/selector.go)

## Why It Matters

Reload and shutdown can leak connections, goroutines, native handles, and
queued writes. Closing children without request leases would instead create
use-after-close races. Unbounded queues, bodies, cardinality, and worksets make
the same lifecycle gap an overload and data-durability risk.

## Desired End State

One immutable `RuntimeGeneration` owns every closeable child and bounded
background actor. Requests acquire a generation lease. Construction uses a
reverse-order rollback stack; reload stops admission, drains leases with a
deadline, and then shuts down the old generation. Actors expose explicit
open/closing/closed admission, cancellation, drain, error, and retention
contracts.

## Exit Criteria

- Every runtime child has one documented owner and idempotent close contract.
- Constructor fault injection proves exact-once reverse cleanup.
- Reload/stream and classification refresh tests pass under `go test -race`.
- Replay, workflow, cache, vector, and memory actors have bounded queues,
  contexts, drain semantics, and visible failures.
- Repeated reload/overload/SIGTERM tests keep goroutine, FD, connection, native
  handle, RSS, and GPU-memory counters stable and lose no acknowledged work.
