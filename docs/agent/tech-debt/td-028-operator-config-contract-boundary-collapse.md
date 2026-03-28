# TD028: Operator Config Contract Still Collapses Across CRD Schema, Webhook Validation, Canonical Translation, and Sample Fixtures

## Status

Open

## Scope

`deploy/operator/api/v1alpha1/**`, `deploy/operator/controllers/canonical_config_builder.go`, and adjacent generated/sample contract tests

## Summary

The Kubernetes operator still spreads one config contract across too many ownership seams. `semanticrouter_types.go` defines a broad CRD schema inventory, `semanticrouter_webhook.go` owns semantic validation for several unrelated spec families, `controllers/canonical_config_builder.go` translates the CRD into the router's canonical v0.3 config, and sample or webhook tests encode additional contract expectations. A single operator config change can therefore require synchronized edits across CRD fields, webhook validators, canonical conversion helpers, and sample fixtures. TD006 already records operator-side structural relief, but the repo still lacks a subsystem-specific debt entry for this operator config contract collapse.

## Evidence

- [deploy/operator/api/v1alpha1/semanticrouter_types.go](../../../deploy/operator/api/v1alpha1/semanticrouter_types.go)
- [deploy/operator/api/v1alpha1/semanticrouter_webhook.go](../../../deploy/operator/api/v1alpha1/semanticrouter_webhook.go)
- [deploy/operator/api/v1alpha1/sample_validation_test.go](../../../deploy/operator/api/v1alpha1/sample_validation_test.go)
- [deploy/operator/api/v1alpha1/semanticrouter_webhook_test.go](../../../deploy/operator/api/v1alpha1/semanticrouter_webhook_test.go)
- [deploy/operator/controllers/canonical_config_builder.go](../../../deploy/operator/controllers/canonical_config_builder.go)
- [deploy/operator/config/samples/](../../../deploy/operator/config/samples)
- [docs/agent/module-boundaries.md](../module-boundaries.md)
- [docs/agent/repo-map.md](../repo-map.md)
- [docs/agent/tech-debt/td-006-structural-rule-target-vs-legacy-hotspots.md](td-006-structural-rule-target-vs-legacy-hotspots.md)

## Why It Matters

- Operator config evolution is harder than it should be because schema declaration, semantic validation, canonical translation, and sample-fixture coverage do not have clear primary owners.
- Drift risk is high: the CRD can accept or document shapes that the controller-side canonical builder or sample fixtures do not fully reflect.
- The operator tree is already covered by changed-file validation and generation gates, but it still lacks near-code rules explaining how to keep these seams narrow.

## Desired End State

- CRD schema families, webhook validation helpers, controller-side canonical config translation, and sample-fixture expectations each have clearer module owners.
- `semanticrouter_types.go` stays focused on schema contracts, webhook files delegate family-specific validation to narrower helpers, and `canonical_config_builder.go` delegates family-specific conversion into adjacent translation helpers.
- Operator-local AGENT rules explicitly tell contributors where schema, validation, and canonical conversion logic belong before the files widen further.

## Exit Criteria

- New operator config features no longer require parallel edits across `semanticrouter_types.go`, `semanticrouter_webhook.go`, and `canonical_config_builder.go` unless the change is intentionally cross-cutting.
- Sample-fixture or webhook regressions are updated through clear family-specific helpers instead of one broad CRD or builder hotspot.
- The operator API and controller directories both have local AGENT rules that reflect the narrower contract ownership.
