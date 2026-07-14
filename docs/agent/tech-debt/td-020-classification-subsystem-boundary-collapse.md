# TD020: Classification Subsystem Boundaries Have Collapsed Into Hotspot Orchestrators

## Status

Open

## Owner Plan

PL0032 Architecture Debt Consolidation

## Release Relevance

None - non-release debt

## Scope

`src/semantic-router/pkg/classification/**`,
`src/semantic-router/pkg/services/classification.go`, and adjacent classifier
bootstrap, discovery, signal-evaluation, and native-backend seams.

## Summary

Classification has been split into many narrower helpers, but the subsystem
still has broad request-time and construction seams. `classifier.go`,
embedding/classifier support files, MCP/hallucination/category paths, service
assembly, and signal dispatch still sit close enough together that backend or
signal-family work can reopen the same hotspots.

The valuable remaining work is not another broad audit. It is to keep
request-time signal orchestration, service assembly, classifier construction,
model discovery, and native-backend wiring from growing back into one central
orchestrator. Runtime refresh also needs to publish one immutable
config/classifier snapshot and retire the previous snapshot through an owned
lifecycle instead of mutating and replacing fields independently.

## Evidence

- [src/semantic-router/pkg/classification/classifier.go](../../../src/semantic-router/pkg/classification/classifier.go)
- [src/semantic-router/pkg/classification/embedding_classifier.go](../../../src/semantic-router/pkg/classification/embedding_classifier.go)
- [src/semantic-router/pkg/classification/model_discovery.go](../../../src/semantic-router/pkg/classification/model_discovery.go)
- [src/semantic-router/pkg/classification/keyword_classifier.go](../../../src/semantic-router/pkg/classification/keyword_classifier.go)
- [src/semantic-router/pkg/classification/mcp_classifier.go](../../../src/semantic-router/pkg/classification/mcp_classifier.go)
- [src/semantic-router/pkg/classification/hallucination_detector.go](../../../src/semantic-router/pkg/classification/hallucination_detector.go)
- [src/semantic-router/pkg/classification/unified_classifier.go](../../../src/semantic-router/pkg/classification/unified_classifier.go)
- [src/semantic-router/pkg/services/classification.go](../../../src/semantic-router/pkg/services/classification.go)
- [src/semantic-router/pkg/services/classification_runtime_config.go](../../../src/semantic-router/pkg/services/classification_runtime_config.go)
- [tools/agent/structure-rules.yaml](../../../tools/agent/structure-rules.yaml)

## Why It Matters

- Adding or changing a backend, signal family, or classifier option can still
  affect construction, discovery, scoring, metrics, and service DTOs at once.
- Broad request-time dispatch makes signal behavior harder to test in isolation.
- Native backend and unified-classifier wiring should not leak into every
  classifier feature.
- Unsynchronized refresh/read paths can expose a mixed config/classifier
  generation, while replaced keyword/MCP/native resources have no retirement
  owner.

## Desired End State

- Backend discovery, model initialization, request-time signal dispatch,
  projection/scoring, service DTOs, and native backend adapters each have narrow
  owner modules.
- New signal families extend family-owned evaluators and tests without touching
  unrelated classifier construction.
- Structure-rule exceptions for classification shrink as hotspots are split.
- Requests load one immutable classification snapshot; refresh builds off-path,
  atomically publishes it, and retires the old snapshot after readers drain.
- Classifier construction and retirement close every successfully created
  native or MCP child exactly once, including partial-build failures.

## Exit Criteria

- New classifier features can land through a family-owned module and focused
  tests without editing central constructors unless construction truly changes.
- `classifier.go` and sibling orchestrators no longer need structure exceptions
  for routine signal or backend changes.
- Service-layer classification endpoints consume narrow classifier APIs instead
  of duplicating discovery or bootstrap ownership.
- Refresh versus every request entrypoint is race-free and every response is
  wholly old-generation or new-generation.
