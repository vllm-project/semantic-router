# Operator Controller Notes

## Scope

- `deploy/operator/controllers/**`

## Responsibilities

- Keep reconciler orchestration, backend discovery, platform integration, and canonical config translation on separate seams.
- Treat `canonical_config_builder.go` as the operator-to-router translation hotspot, not as the default home for all controller-side helper logic.
- Keep Kubernetes discovery or platform-specific wiring outside family-specific canonical config translation when possible.

## Change Rules

- Do not add new CRD schema or admission-validation logic into controller helpers.
- When a config family grows, prefer dedicated translation helpers instead of widening `canonical_config_builder.go`.
- Keep backend discovery, OpenShift or gateway integration, and canonical translation on separate seams even when a feature touches more than one of them.
- If a change requires updating both controller translation and API schema semantics, update the nearest API and controller local rules together and keep the ownership boundary explicit.
