# TD020: Classification Subsystem Boundaries Have Collapsed Into Hotspot Orchestrators

## Status

Open

## Scope

`src/semantic-router/pkg/classification/**`, `src/semantic-router/pkg/services/classification.go`, and adjacent classifier bootstrap/discovery seams

## Summary

The classification subsystem has drifted into a shallow, high-churn design where giant orchestrator files own too many unrelated responsibilities at once. `classifier.go` mixes backend selection, model initialization, category and security mapping ownership, concurrency and metrics handling, and request-time family inference. The service layer partially duplicates discovery and bootstrap work, while the unified-classifier path and legacy path are layered unclearly. The result is a hotspot that resists change: adding a new backend or signal family tends to touch the same large constructor and shared state table instead of a narrow seam.

## Evidence

- [src/semantic-router/pkg/classification/classifier.go](../../../src/semantic-router/pkg/classification/classifier.go)
- [src/semantic-router/pkg/classification/unified_classifier.go](../../../src/semantic-router/pkg/classification/unified_classifier.go)
- [src/semantic-router/pkg/classification/model_discovery.go](../../../src/semantic-router/pkg/classification/model_discovery.go)
- [src/semantic-router/pkg/services/classification.go](../../../src/semantic-router/pkg/services/classification.go)
- [src/semantic-router/pkg/services/classification_test.go](../../../src/semantic-router/pkg/services/classification_test.go)
- [tools/agent/structure-rules.yaml](../../../tools/agent/structure-rules.yaml)

## Why It Matters

- A single hotspot becomes the edit point for backend discovery, mapping policy, inference flow, and metrics, which increases change risk and review cost.
- The current design makes it hard to test one classification seam in isolation because bootstrap, family logic, and runtime orchestration are entangled.
- Duplicate bootstrap logic between the service layer and classifier package encourages drift in mapping loading, model discovery, and fallback behavior.
- The structure-rule ratchet already treats these files as legacy hotspots, which confirms the code shape has exceeded the intended architecture.

## Desired End State

- Model discovery and bootstrap live behind dedicated helpers or adapters instead of the main request-time classifier orchestration.
- Per-family classification concerns such as category, jailbreak, and PII inference can evolve behind narrow seams rather than shared giant structs.
- The unified batch path and legacy path expose explicit interfaces and wiring points instead of being mixed into one monolithic runtime surface.
- Service-level composition owns assembly only; it does not reimplement mapping or discovery concerns that belong inside the classification package.

## Exit Criteria

- `classifier.go` and `services/classification.go` have materially reduced responsibility counts, with family adapters, discovery/bootstrap logic, and mapping ownership extracted into dedicated modules.
- New classifier backends or signal families can be added through narrow seams without editing a giant shared constructor or state table.
- Classification tests cover the extracted seams independently enough that bootstrap, mapping, and request-time inference can fail separately.
- The classification hotspot no longer requires relaxed structure treatment for the same responsibilities that are currently bundled together.
