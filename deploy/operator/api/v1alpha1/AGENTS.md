# Operator API Notes

## Scope

- `deploy/operator/api/v1alpha1/**`

## Responsibilities

- Keep CRD schema declaration, admission validation, and generated contract expectations distinct.
- Treat `semanticrouter_types.go` as the schema hotspot and `semanticrouter_webhook.go` as the admission-validation hotspot, not as catch-all homes for every operator config change.
- Keep sample and webhook regression tests aligned with the operator contract without pushing fixture-specific logic into production types.

## Change Rules

- Do not add controller-side canonical config translation logic into API type or webhook files.
- When a spec family grows, prefer dedicated schema-family or validation-helper files over widening `semanticrouter_types.go` or `semanticrouter_webhook.go`.
- Keep generated CRD, sample fixtures, and webhook tests aligned with the API contract in the same change; do not leave schema drift for a later patch.
- If a change requires edits in both schema declaration and semantic validation, keep the files separate and make the shared contract seam explicit instead of widening one hotspot.
