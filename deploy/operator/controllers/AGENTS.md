# Operator Controller Notes

## Scope

- `deploy/operator/controllers/**`

## Responsibilities

- Keep reconciler orchestration, immutable bootstrap validation, and platform integration on separate seams.
- Treat the selected immutable ConfigMap as the only Router configuration source; the Operator must not synthesize routing or access resources.
- Keep OpenShift, Gateway, storage, and workload wiring in narrow platform-specific helpers.

## Change Rules

- Do not add new CRD schema or admission-validation logic into controller helpers.
- Keep bootstrap checks at the deployment boundary lightweight; full manifest compilation belongs to Router startup.
- A bootstrap change requires a new immutable ConfigMap reference and a Pod rollout, never an in-process reload path.
- If a change updates both API schema and controller semantics, update the nearest API and controller local rules together and keep the ownership boundary explicit.
