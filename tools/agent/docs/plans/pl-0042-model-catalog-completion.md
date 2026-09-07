# PL-0042: Model Catalog Completion

## Goal

Land the unified built-in model catalog with accurate model reasoning controls,
provider-native request projection, complete public Hub surfaces, and verified
runtime behavior.

## Scope

- Audit built-in model limits and configurable reasoning controls against
  primary sources.
- Validate every model-family/provider-transport combination at generation and
  runtime boundaries.
- Cover final Chat Completions, Responses, Messages, gateway, and local-runtime
  request shapes.
- Complete Dashboard and Website Model Hub regression, mobile QA, PR CI, and
  preview deployment.

## Non-Goals

- The separate GPT-6 Astra Day-0 example change.
- Fabricating unavailable evaluation measurements.
- Adding low-signal model creators outside the reviewed catalog policy.

## Exit Criteria

- Catalog generation, audit, Go, Python, Dashboard, Website, operator, and
  affected E2E gates pass.
- A real request traverses vLLM Semantic Router to a model backend.
- PR is rebased, pushed, and all required checks are green.
- Website and Dashboard previews are reachable and visually inspected at
  desktop and mobile widths.

## Task List

- [x] `TASK-01` Close metadata and reasoning-control audit findings.
- [x] `TASK-02` Complete provider wire-contract and validation tests.
- [x] `TASK-03` Run local and AMD validation, fixing failures locally.
- [ ] `TASK-04` Rebase, push, and drive PR CI green.
- [x] `TASK-05` Deploy and inspect Website and Dashboard previews.

## Next Action

Drive the latest rebased pull-request head through every required CI check.

## Operating Rules

- Keep the local checkout as implementation source of truth.
- Use a dynamically discovered AMD host only as an exact validation mirror.
- Do not merge the pull request without explicit authorization.

## Related Docs

- [Unified model catalog proposal](../../../website/docs/proposals/unified-model-catalog-and-evaluation-index.md)
- [Day-0 support guide](../../../website/docs/community/model-provider-day-0-support.md)
