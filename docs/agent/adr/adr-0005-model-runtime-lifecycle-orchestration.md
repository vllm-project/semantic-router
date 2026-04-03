# ADR 0005: Adopt Explicit Model Runtime Lifecycle Planning and Orchestration

## Status

Accepted

## Context

The router currently manages model lifecycle through an implicit mix of config defaults, download preflight, startup bootstrap, extproc router construction, and classifier constructors.

- `cmd/main.go` and `runtime_bootstrap.go` sequence config load, download preflight, embedding initialization, vector-store startup, modality setup, router creation, and tools warmup in one startup chain.
- `pkg/extproc/router_build.go` and `router_components.go` assemble mappings, classifier construction, cache, replay, memory, response API, and selector state in one router build path.
- `pkg/classification` constructors still perform runtime initialization and preload work, which mixes model discovery, model init, and request-time inference ownership in the same package surface.
- Heavy preload work for embedding, complexity, and knowledge-base classifiers is hidden behind constructor calls, so the runtime cannot reason about it as a first-class startup phase.
- Native Candle bindings expose many model handles through `OnceLock` singletons. Some model families are independently safe to initialize in parallel, but the embedding factory path is order-sensitive because `GLOBAL_MODEL_FACTORY` can only be set once and mmBERT participation must be decided before that point.

The result is a lifecycle that works, but whose sequencing rules are distributed across too many packages and are difficult to parallelize, test, or evolve safely.

## Decision

Adopt an explicit model runtime lifecycle contract centered on a planner, a bounded-parallel task orchestrator, and explicit build-versus-init seams.

- Introduce a dedicated runtime-orchestration package for dependency-aware task execution instead of encoding startup order only in call order.
- Treat model lifecycle as three explicit phases:
  - asset readiness
  - runtime initialization
  - warmup or preload
- Separate classifier construction from classifier runtime initialization. Constructors may assemble rule-owned components, but higher layers must be able to choose when runtime initialization happens.
- Make preload-heavy classifier component construction parallel where dependencies allow it, so embedding, complexity, and knowledge-base preloads no longer serialize startup by default.
- Keep order-sensitive native-binding paths explicit. Embedding factory setup remains a dedicated controlled task instead of pretending every model family is equally parallel-safe.
- Preserve degraded-startup behavior where it is already part of the runtime contract, but express it as best-effort tasks rather than buried constructor side effects.
- Keep extproc orchestration thin: it should build router components from explicit ready or buildable dependencies, not rediscover hidden runtime init behavior from constructors.

## Consequences

- Startup behavior becomes inspectable as a runtime task graph instead of a long chain of implicit side effects.
- Classifier runtime init can be called explicitly by router assembly, while the legacy `NewClassifier` convenience path can remain as a compatibility wrapper during migration.
- Embedding-family and preload-heavy startup work can run concurrently without losing the ability to serialize order-sensitive native paths.
- Future work can migrate config reload, download planning, and backend capability checks onto the same lifecycle contract instead of adding more startup branches.
- The repository still carries broader global-state and native-binding lifecycle debt after this decision, especially around process-wide service registries and backend capability parity; that debt remains tracked in TD031 and TD033 until the migration closes.
