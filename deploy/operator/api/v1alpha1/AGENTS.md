# Operator API Notes

## Scope

- `deploy/operator/api/v1alpha1/**`

## Responsibilities

- Keep CRD schema declaration, admission validation, and generated contract expectations distinct.
- Treat `semanticrouter_types.go` as the **workload/platform** schema slice (image, probes, ingress, `vllmEndpoints`, …).
- Treat `semanticrouter_types_config_spec.go` and `semanticrouter_types_config_*_family.go` as the **router config contract** schema (everything under `spec.config`).
- Treat `semanticrouter_webhook.go` as webhook wiring only; put admission rules in `semanticrouter_webhook_validation.go` (or future family-specific validation helpers).
- Keep sample and webhook regression tests aligned with the operator contract without pushing fixture-specific logic into production types.

## Change Rules

- Do not add controller-side canonical config translation logic into API type or webhook files.
- When a spec family grows, add or extend the matching **config family** file (`*_config_*_family.go`) or validation helper instead of widening `semanticrouter_types.go` or `semanticrouter_webhook.go`.
- Keep generated CRD, sample fixtures, and webhook tests aligned with the API contract in the same change; do not leave schema drift for a later patch.
- If a change requires edits in both schema declaration and semantic validation, keep the files separate and make the shared contract seam explicit instead of widening one hotspot.
