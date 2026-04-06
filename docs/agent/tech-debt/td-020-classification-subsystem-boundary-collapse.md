# TD020: Classification Subsystem Boundaries Have Collapsed Into Hotspot Orchestrators

## Status

Closed

## Scope

`src/semantic-router/pkg/classification/**`, `src/semantic-router/pkg/services/classification*.go`, `src/semantic-router/pkg/services/classification_bootstrap_test.go`, and `tools/agent/structure-rules.yaml`

## Summary

The classification subsystem now has narrower seams for lifecycle, service assembly, and family-specific response handling. `classifier.go` no longer needs legacy structure relief, because classifier lifecycle and family composition already live on adjacent helpers such as `classifier_lifecycle.go`, `classifier_construction.go`, and `category_classifier.go`. The remaining service hotspot has been decomposed: auto-discovery and mapping/bootstrap live in `classification_bootstrap.go`, intent evaluation stays with the signal contract in `classification_signal_contract.go`, and PII, security, unified-batch, fact-check, and user-feedback handlers each own their own response seams. The old `classifier.go` / `services/classification.go` relaxed structure entry has been removed from `tools/agent/structure-rules.yaml`.

## Evidence

- [src/semantic-router/pkg/classification/classifier.go](../../../src/semantic-router/pkg/classification/classifier.go)
- [src/semantic-router/pkg/classification/classifier_lifecycle.go](../../../src/semantic-router/pkg/classification/classifier_lifecycle.go)
- [src/semantic-router/pkg/classification/classifier_construction.go](../../../src/semantic-router/pkg/classification/classifier_construction.go)
- [src/semantic-router/pkg/classification/category_classifier.go](../../../src/semantic-router/pkg/classification/category_classifier.go)
- [src/semantic-router/pkg/classification/unified_classifier.go](../../../src/semantic-router/pkg/classification/unified_classifier.go)
- [src/semantic-router/pkg/classification/model_discovery.go](../../../src/semantic-router/pkg/classification/model_discovery.go)
- [src/semantic-router/pkg/services/classification.go](../../../src/semantic-router/pkg/services/classification.go)
- [src/semantic-router/pkg/services/classification_bootstrap.go](../../../src/semantic-router/pkg/services/classification_bootstrap.go)
- [src/semantic-router/pkg/services/classification_signal_contract.go](../../../src/semantic-router/pkg/services/classification_signal_contract.go)
- [src/semantic-router/pkg/services/classification_pii.go](../../../src/semantic-router/pkg/services/classification_pii.go)
- [src/semantic-router/pkg/services/classification_security.go](../../../src/semantic-router/pkg/services/classification_security.go)
- [src/semantic-router/pkg/services/classification_unified.go](../../../src/semantic-router/pkg/services/classification_unified.go)
- [src/semantic-router/pkg/services/classification_feedback.go](../../../src/semantic-router/pkg/services/classification_feedback.go)
- [src/semantic-router/pkg/services/classification_bootstrap_test.go](../../../src/semantic-router/pkg/services/classification_bootstrap_test.go)
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

## Validation

- `go test ./pkg/services ./pkg/classification`
- `python3 tools/agent/scripts/structure_check.py src/semantic-router/pkg/classification/classifier.go src/semantic-router/pkg/services/classification.go src/semantic-router/pkg/services/classification_bootstrap.go src/semantic-router/pkg/services/classification_pii.go src/semantic-router/pkg/services/classification_security.go src/semantic-router/pkg/services/classification_feedback.go src/semantic-router/pkg/services/classification_unified.go src/semantic-router/pkg/services/classification_signal_contract.go`
