# TD028: Operator Config Contract Still Collapses Across CRD Schema, Webhook Validation, Canonical Translation, and Sample Fixtures

## Status

Open

## Scope

`deploy/operator/api/v1alpha1/**`, `deploy/operator/controllers/canonical_config_builder.go`, and adjacent generated/sample contract tests

## Summary

The Kubernetes operator still spreads one config contract across too many ownership seams. `semanticrouter_types.go` defines a broad CRD schema inventory, `semanticrouter_webhook.go` owns semantic validation for several unrelated spec families, and sample or webhook tests encode additional contract expectations. A single operator config change can therefore require synchronized edits across CRD fields, webhook validators, canonical conversion helpers, and sample fixtures. Current-source recheck shows operator-local AGENT rules now exist for API and controller ownership, so the remaining debt is implementation boundary relief rather than missing near-code guidance.

The controller-side canonical translation portion is narrower after the
2026-05-24 ASR008 slice. `controllers/canonical_config_builder.go` now owns base
canonical config construction and orchestration only; discovered backend fan-in
lives in `controllers/canonical_config_backends.go`, and CRD spec-family
translation lives in `controllers/canonical_config_spec.go`. The debt remains
open because CRD schema size, webhook validation-family ownership, and
sample-fixture ownership still need the same boundary relief.

## Evidence

- [deploy/operator/api/v1alpha1/semanticrouter_types.go](../../../deploy/operator/api/v1alpha1/semanticrouter_types.go)
- [deploy/operator/api/v1alpha1/semanticrouter_webhook.go](../../../deploy/operator/api/v1alpha1/semanticrouter_webhook.go)
- [deploy/operator/api/v1alpha1/sample_validation_test.go](../../../deploy/operator/api/v1alpha1/sample_validation_test.go)
- [deploy/operator/api/v1alpha1/semanticrouter_webhook_test.go](../../../deploy/operator/api/v1alpha1/semanticrouter_webhook_test.go)
- [deploy/operator/controllers/canonical_config_builder.go](../../../deploy/operator/controllers/canonical_config_builder.go)
- [deploy/operator/controllers/canonical_config_backends.go](../../../deploy/operator/controllers/canonical_config_backends.go)
- [deploy/operator/controllers/canonical_config_spec.go](../../../deploy/operator/controllers/canonical_config_spec.go)
- [deploy/operator/controllers/canonical_config_spec_test.go](../../../deploy/operator/controllers/canonical_config_spec_test.go)
- [deploy/operator/api/v1alpha1/AGENTS.md](../../../deploy/operator/api/v1alpha1/AGENTS.md)
- [deploy/operator/controllers/AGENTS.md](../../../deploy/operator/controllers/AGENTS.md)
- [deploy/operator/config/samples/](../../../deploy/operator/config/samples)
- [docs/agent/module-boundaries.md](../module-boundaries.md)
- [docs/agent/repo-map.md](../repo-map.md)
- [docs/agent/tech-debt/td-006-structural-rule-target-vs-legacy-hotspots.md](td-006-structural-rule-target-vs-legacy-hotspots.md)

## Why It Matters

- Operator config evolution is harder than it should be because schema declaration, semantic validation, and sample-fixture coverage do not have clear primary owners.
- Drift risk is high: the CRD can accept or document shapes that the controller-side canonical builder or sample fixtures do not fully reflect.
- The operator tree is already covered by changed-file validation and generation gates, and it now has near-code rules explaining how to keep these seams narrow. The implementation files still need extraction so those rules are reflected in code shape.

## Desired End State

- CRD schema families, webhook validation helpers, controller-side canonical config translation, and sample-fixture expectations each have clearer module owners.
- `semanticrouter_types.go` stays focused on schema contracts, webhook files delegate family-specific validation to narrower helpers, and controller-side canonical config translation remains split between base construction, backend discovery, and family-specific conversion helpers.
- Operator-local AGENT rules already tell contributors where schema, validation, and canonical conversion logic belong; the implementation should now converge to those rules.

## Exit Criteria

- New operator config features no longer require parallel edits across `semanticrouter_types.go`, `semanticrouter_webhook.go`, and `canonical_config_builder.go` unless the change is intentionally cross-cutting.
- Sample-fixture or webhook regressions are updated through clear family-specific helpers instead of one broad CRD or builder hotspot.
- The operator API and controller directories both have local AGENT rules that reflect the narrower contract ownership. This criterion is satisfied; the remaining exit criteria are code and fixture boundary changes.
