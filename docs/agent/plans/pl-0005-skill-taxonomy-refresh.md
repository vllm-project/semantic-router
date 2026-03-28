# Skill Taxonomy Refresh Execution Plan

## Goal

- Align the harness skill taxonomy with the repository's current active module boundaries.
- Eliminate fallback-only routing for the representative subsystem paths captured in TD018.
- Close TD018 after the registry, skill docs, surface catalog, and routing validation converge.

## Scope

- `tools/agent/skill-registry.yaml`
- `docs/agent/change-surfaces.md`
- `docs/agent/skill-catalog.md`
- relevant `tools/agent/skills/**/SKILL.md` files
- routing fixtures and validation under `tools/agent/scripts/**` and `tools/agent/routing-fixtures.yaml`
- representative active subsystems and deployment areas now missing first-class ownership:
  - `src/fleet-sim/**`
  - `src/semantic-router/pkg/apiserver/**`
  - `src/semantic-router/pkg/memory/**`
  - `src/semantic-router/pkg/responseapi/**`
  - `src/semantic-router/pkg/authz/**`
  - `src/semantic-router/pkg/ratelimit/**`
  - `deploy/kubernetes/**` paths that are not operator CRDs
  - currently uncovered router runtime modules such as `imagegen`, `promptcompression`, `routerreplay`, `mcp`, and related storage/runtime packages
- rename or retirement work for stale skill families such as classifier-only training and plugin-runtime terminology

## Exit Criteria

- Representative `agent-report` runs for `src/fleet-sim/**`, `src/semantic-router/pkg/memory/**`, `src/semantic-router/pkg/responseapi/**`, `src/semantic-router/pkg/authz/**`, `src/semantic-router/pkg/apiserver/**`, and `deploy/kubernetes/response-api/**` resolve to dedicated or intentionally shared non-fallback primary skills.
- `docs/agent/change-surfaces.md` reflects current package ownership instead of stale path examples such as `src/semantic-router/pkg/plugins/**`.
- The skill catalog and SKILL.md corpus use current, defensible names for the training and router-runtime extension families.
- Routing fixtures and validation fail if the representative subsystem paths regress back to `cross-stack-bugfix`.
- TD018 is updated to `Closed` with a concrete resolution summary.

## Task List

- [x] `S001` Create TD018 and this execution plan, and capture the initial routing evidence from the current harness state.
- [x] `S002` Define the target taxonomy and naming map for active subsystems, including which skills are added, renamed, merged, or retired.
- [x] `S003` Implement first-class router-service routing for apiserver, memory/response-api, and authz/rate-limit changes, including matching surface definitions and skill docs.
- [x] `S004` Implement first-class routing for fleet-sim and non-CRD deployment/profile changes so these paths stop falling through to the cross-stack fallback.
- [x] `S005` Replace stale classifier-only and plugin-runtime terminology with current training-stack and router-runtime-extension naming, then update the affected skill docs and surface docs.
- [x] `S006` Add routing fixtures and validation coverage for every representative path needed to keep TD018 closed.
- [x] `S007` Run the canonical validation sequence, update TD018 with the closure evidence, and retire this plan once the exit criteria hold.

## Current Loop

- Completed on 2026-03-18: the skill taxonomy refresh landed in the executable harness, representative routing fixtures were added, and TD018 was closed with validation evidence.

## Decision Log

- Prefer a small set of current subsystem-aligned skills over accumulating legacy aliases that describe outdated code boundaries.
- Treat `cross-stack-bugfix` as a true fallback only. If a subsystem has its own code area, local rules, or dedicated E2E/workflow coverage, it should not rely on fallback routing by default.
- Land routing fixtures in the same loop as taxonomy changes so the new ownership model becomes mechanically enforced.
- Keep skill prose concise and use `## Gotchas` plus linked references for progressive disclosure instead of copying large playbook content into each skill.

## Follow-up Debt / ADR Links

- [../tech-debt-register.md](../tech-debt-register.md)
- [../tech-debt/README.md](../tech-debt/README.md)
- [../tech-debt/td-018-skill-surface-coverage-drift.md](../tech-debt/td-018-skill-surface-coverage-drift.md)
- [../adr/README.md](../adr/README.md) (no dedicated ADR yet)
