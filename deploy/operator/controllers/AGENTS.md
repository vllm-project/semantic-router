# Operator Controller Notes

## Scope

- `deploy/operator/controllers/**`

## Responsibilities

- Keep reconciler orchestration, backend discovery, platform integration, and canonical config translation on separate seams.
- Treat `canonical_config_builder.go` as **orchestration only**: default canonical skeleton, `applyDiscoveredBackends`, and LoRA helpers for discovered models.
- Treat `canonical_operator_config_translate.go` as **`spec.config` → canonical v0.3** (model catalog, stores, routing, services, typed YAML bridge).
- Keep Kubernetes discovery or platform-specific wiring outside family-specific canonical config translation when possible.

## Change Rules

- Do not add new CRD schema or admission-validation logic into controller helpers.
- When a config family grows, extend `canonical_operator_config_translate.go` (or add a sibling `canonical_operator_config_*` helper) instead of widening `canonical_config_builder.go`.
- Keep backend discovery, OpenShift or gateway integration, and canonical translation on separate seams even when a feature touches more than one of them.
- If a change requires updating both controller translation and API schema semantics, update the nearest API and controller local rules together and keep the ownership boundary explicit.
