# TD028: Operator Config Contract Still Collapses Across CRD Schema, Webhook Validation, Canonical Translation, and Sample Fixtures

## Status

Open

## Owner Plan

PL0033 v0.3 Themis Release Closure

## Release Relevance

v0.3 Themis

## Scope

Operator API schema, webhook validation, controller canonical translation, and
generated/sample contract fixtures.

## Summary

The operator still spreads one config contract across too many owner seams.
`semanticrouter_types.go` carries a broad CRD schema inventory,
`semanticrouter_webhook.go` validates multiple spec families, controller
canonical translation lives in adjacent controller helpers, and sample/webhook
tests encode additional contract expectations.

Controller-side canonical translation has been narrowed into base construction,
backend discovery, and spec-family conversion helpers. The remaining debt is to
give CRD schema families, webhook validation, and sample fixtures the same clear
ownership.

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
- [docs/agent/tech-debt/td-006-structural-rule-exceptions.md](td-006-structural-rule-exceptions.md)

## Why It Matters

- Operator config evolution should not require synchronized edits across broad
  schema, validation, conversion, and sample-fixture hotspots for routine
  feature work.
- Drift risk is high when accepted CRD shapes, canonical conversion, and sample
  fixtures do not have clear primary owners.
- v0.3 release confidence depends on operator config behavior matching the
  canonical router contract.

## Desired End State

- CRD schema families, webhook validation helpers, controller-side canonical
  translation, and sample-fixture expectations each have clear module owners.
- `semanticrouter_types.go` stays focused on schema contracts.
- Webhook validation delegates family-specific behavior to narrow helpers.
- Controller canonical conversion remains split between base construction,
  backend discovery, and family-specific conversion.

## Exit Criteria

- New operator config features do not require unrelated edits across schema,
  webhook, canonical builder, and sample fixture files.
- Sample-fixture and webhook regressions are updated through family-owned
  helpers.
- Operator local rules and implementation shape agree on schema, validation, and
  canonical conversion ownership.
