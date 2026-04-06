# TD028: Operator Config Contract Still Collapses Across CRD Schema, Webhook Validation, Canonical Translation, and Sample Fixtures

## Status

Closed

## Scope

`deploy/operator/api/v1alpha1/**`, `deploy/operator/controllers/canonical_config_builder*.go`, and adjacent generated/sample contract tests

## Summary

The operator contract now has explicit family seams. `semanticrouter_types.go` remains the schema inventory, admission validation fans out from `semanticrouter_webhook.go` into family-specific helpers, controller-side canonical translation fans out from `canonical_config_builder.go` into adjacent model-catalog, routing, services, stores, provider-default, discovery, and support helpers, and sample-fixture assertions now route through sample-specific helpers instead of one broad filename switch. Operator-local `AGENTS.md` files and shared module-boundary docs now describe the same ownership split, so this debt no longer needs to stay open.

## Evidence

- [deploy/operator/api/v1alpha1/semanticrouter_types.go](../../../deploy/operator/api/v1alpha1/semanticrouter_types.go)
  - schema declaration remains the CRD-owned seam
- [deploy/operator/api/v1alpha1/semanticrouter_webhook.go](../../../deploy/operator/api/v1alpha1/semanticrouter_webhook.go)
  - webhook entrypoint now dispatches family-specific validation helpers instead of owning all rule bodies inline
- [deploy/operator/api/v1alpha1/semanticrouter_validation_autoscaling.go](../../../deploy/operator/api/v1alpha1/semanticrouter_validation_autoscaling.go)
- [deploy/operator/api/v1alpha1/semanticrouter_validation_persistence.go](../../../deploy/operator/api/v1alpha1/semanticrouter_validation_persistence.go)
- [deploy/operator/api/v1alpha1/semanticrouter_validation_probes.go](../../../deploy/operator/api/v1alpha1/semanticrouter_validation_probes.go)
- [deploy/operator/api/v1alpha1/semanticrouter_validation_ingress.go](../../../deploy/operator/api/v1alpha1/semanticrouter_validation_ingress.go)
- [deploy/operator/api/v1alpha1/sample_validation_test.go](../../../deploy/operator/api/v1alpha1/sample_validation_test.go)
- [deploy/operator/api/v1alpha1/sample_validation_support_test.go](../../../deploy/operator/api/v1alpha1/sample_validation_support_test.go)
  - sample contract assertions now route through sample-specific helpers
- [deploy/operator/controllers/canonical_config_builder.go](../../../deploy/operator/controllers/canonical_config_builder.go)
  - controller entrypoint now orchestrates family translators instead of owning each conversion body inline
- [deploy/operator/controllers/canonical_config_builder_model_catalog.go](../../../deploy/operator/controllers/canonical_config_builder_model_catalog.go)
- [deploy/operator/controllers/canonical_config_builder_stores_integrations.go](../../../deploy/operator/controllers/canonical_config_builder_stores_integrations.go)
- [deploy/operator/controllers/canonical_config_builder_provider_defaults.go](../../../deploy/operator/controllers/canonical_config_builder_provider_defaults.go)
- [deploy/operator/controllers/canonical_config_builder_services.go](../../../deploy/operator/controllers/canonical_config_builder_services.go)
- [deploy/operator/controllers/canonical_config_builder_routing.go](../../../deploy/operator/controllers/canonical_config_builder_routing.go)
- [deploy/operator/controllers/canonical_config_builder_discovery.go](../../../deploy/operator/controllers/canonical_config_builder_discovery.go)
- [deploy/operator/controllers/canonical_config_builder_support.go](../../../deploy/operator/controllers/canonical_config_builder_support.go)
- [deploy/operator/api/v1alpha1/AGENTS.md](../../../deploy/operator/api/v1alpha1/AGENTS.md)
- [deploy/operator/controllers/AGENTS.md](../../../deploy/operator/controllers/AGENTS.md)

## Why It Matters

- Operator config evolution is harder than it should be because schema declaration, semantic validation, canonical translation, and sample-fixture coverage do not have clear primary owners.
- Drift risk is high: the CRD can accept or document shapes that the controller-side canonical builder or sample fixtures do not fully reflect.
- The operator tree is already covered by changed-file validation and generation gates, but it still lacks near-code rules explaining how to keep these seams narrow.

## Desired End State

- CRD schema families, webhook validation helpers, controller-side canonical config translation, and sample-fixture expectations each have clearer module owners.
- `semanticrouter_types.go` stays focused on schema contracts, webhook files delegate family-specific validation to narrower helpers, and `canonical_config_builder.go` delegates family-specific conversion into adjacent translation helpers.
- Operator-local AGENT rules explicitly tell contributors where schema, validation, and canonical conversion logic belong before the files widen further.

## Exit Criteria

- Satisfied on 2026-04-06: new operator config features no longer default to parallel edits across `semanticrouter_types.go`, `semanticrouter_webhook.go`, and `canonical_config_builder.go` because webhook validation and canonical translation now fan out through family-specific helpers.
- Satisfied on 2026-04-06: sample-fixture and webhook regressions now route through explicit sample or validation-family helpers instead of one broad hotspot.
- Satisfied on 2026-04-06: the operator API and controller directories both have local `AGENTS.md` rules that reflect the narrower contract ownership.

## Retirement Notes

- `semanticrouter_webhook.go` is now a thin admission entrypoint; autoscaling, persistence, probe, and ingress validation each live in their own helper file.
- `canonical_config_builder.go` is now a thin controller-side translation orchestrator; model-catalog, stores, provider-defaults, services, routing, discovery, and generic typed-conversion support live in adjacent files.
- Sample assertions now route through `sample_validation_support_test.go`, so sample-specific contract checks do not widen the primary sample-validation runner.

## Validation

- `make generate-crd`
- `go test ./deploy/operator/api/v1alpha1 ./deploy/operator/controllers`
- `make agent-ci-gate AGENT_CHANGED_FILES_PATH=<changed-files>`
