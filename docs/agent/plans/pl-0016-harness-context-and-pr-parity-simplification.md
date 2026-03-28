# Harness Context and PR Parity Simplification

## Goal

- Reduce default harness context pressure so `agent-report` stops front-loading low-value skill and governance references.
- Make fragment-skill activation task-scoped instead of unconditional.
- Remove or merge low-value skills and downgrade stock `Must Read` sets so the default harness context carries less dead weight.
- Remove local E2E execution from the default harness completion gate.
- Provide a repo-native local PR gate that reproduces the baseline CI requirements for `Pre-commit / Run pre-commit hooks` and `Test And Build`.

## Scope

- `tools/agent/scripts/agent_context_pack.py`
- `tools/agent/scripts/agent_resolution.py`
- `tools/agent/scripts/agent_models.py`
- `tools/agent/scripts/agent_skill_validation.py`
- `tools/agent/context-map.yaml`
- `tools/agent/skill-registry.yaml`
- `tools/agent/repo-manifest.yaml`
- `tools/agent/skills/**`
- `tools/make/agent.mk`
- `tools/make/pre-commit.mk`
- `.github/workflows/pre-commit.yml`
- `.github/workflows/test-and-build.yml`
- `docs/agent/context-management.md`
- `docs/agent/testing-strategy.md`
- `docs/agent/feature-complete-checklist.md`
- `docs/agent/skill-catalog.md`
- `docs/agent/playbooks/e2e-selection.md`
- `CONTRIBUTING.md`

## Exit Criteria

- Default `agent-report` summary emits a compact context pack and stops surfacing debt-index or low-value catalog docs by default.
- Resolved fragment skills are filtered to the active surfaces and environment instead of inheriting the full primary-skill fragment list.
- Redundant or low-value stock skills are removed or merged, and high-traffic skills stay within the tightened `Must Read` budget.
- `make agent-feature-gate` no longer runs local E2E profiles automatically.
- Contributors have a single repo-native local command that covers the baseline PR requirements for pre-commit parity and test-and-build parity.
- Harness docs, manifest indexing, and validation rules stay aligned with the simplified contract.

## Task List

- [x] `HS001` Create and index the canonical execution plan for this workstream.
  - Done when this file is indexed from the plan README and repo manifest.
- [x] `HS002` Shrink default context-pack disclosure and remove low-value default refs.
  - Done when `agent-report` summary is compact by default and the default pack no longer injects debt-index or catalog-heavy refs for ordinary tasks.
- [x] `HS003` Make fragment activation surface-scoped.
  - Done when `resolve_skill()` only emits active fragments relevant to impacted surfaces and environment.
- [x] `HS004` Remove local E2E from the default completion gate.
  - Done when `agent-feature-gate` stops auto-running local E2E while the explicit `agent-e2e-affected` path remains available.
- [x] `HS005` Add a repo-native local PR parity gate.
  - Done when contributors can run one local command that covers the CI pre-commit path plus the local reproduction of `Test And Build`.
- [x] `HS006` Validate the simplified harness contract and record the outcome here.
  - Done when the applicable repo-native validation commands pass or any remaining heavy-runtime blocker is recorded explicitly.
- [x] `HS007` Clean up the stock skill inventory and downgrade unnecessary `Must Read` refs.
  - Done when redundant or low-value skills are deleted or merged, `cross-stack-bugfix` stops hard-coding the entire fragment inventory, and the tightened skill budgets pass validation.
- [x] `HS008` Collapse overlapping primary skills into broader task archetypes.
  - Done when redundant primary skills are deleted or merged, routing fixtures and reports resolve to the new archetypes, and the primary inventory stays materially smaller.
- [x] `HS009` Collapse overlapping fragment skills into subsystem-aligned implementation slices.
  - Done when decision, dashboard, and Kubernetes fragment clusters are merged into broader runtime slices, architecture guardrails stop loading by default, and the fragment inventory is materially smaller.

## Current Loop

- 2026-03-26: Started the harness simplification loop for context pressure, fragment fanout, local E2E defaults, and PR parity.
- 2026-03-26: Completed the context-pack compaction, surface-scoped fragment activation, local-E2E removal from default gates, local PR parity entrypoints, and the stock skill cleanup pass.
- 2026-03-26: Collapsed overlapping primary skills into broader archetypes, shrinking the primary inventory from 16 to 12 while keeping fragment-level specificity.
- 2026-03-26: Collapsed overlapping fragment skills into broader runtime slices, shrinking the fragment inventory from 20 to 14 and demoting architecture guardrails out of the default fragment path.

## Decision Log

- Keep this work in one loop because context shaping, gate selection, and PR parity are coupled parts of the same harness contract.
- Prefer shrinking default disclosure and default execution over deleting explicit escape hatches; explicit heavy commands can remain available without staying on the default path.
- Treat the completion checklist as canonical documentation instead of a support skill so it stays available without expanding default skill context.
- Merge dashboard config, routing visibility, and console backend guidance into one `dashboard-platform-runtime` fragment so dashboard work routes as one console contract instead of three overlapping sub-fragments.
- Let `cross-stack-bugfix` resolve fragments from impacted surfaces instead of hand-maintaining a catch-all fragment list.
- Remove `header-contract-change` and let router-originated header contracts route through `signal-end-to-end`, while dashboard-only presentation work routes through the dashboard primary.
- Merge dashboard frontend and backend primaries into `dashboard-platform-change` so console work routes by subsystem instead of by UI/server split.
- Merge operator and deployment-profile primaries into `k8s-platform-change` so Kubernetes-facing work routes by platform boundary instead of by manifest bucket.
- Narrow `config-platform-change` back to router config, CLI schema, and dashboard config UI; Kubernetes-facing translation is now a separate archetype.
- Merge decision/selection, dashboard, and Kubernetes fragment clusters into broader runtime slices so the fragment layer mirrors the new coarse primary archetypes instead of reintroducing fine-grained context sprawl underneath them.
- Demote `architecture-guardrails` from a default fragment to a manual support skill; structural review still exists, but it no longer pollutes every non-trivial task by default.

## Follow-up Debt / ADR Links

- [README.md](README.md)
- [../tech-debt-register.md](../tech-debt-register.md)
- [../tech-debt/README.md](../tech-debt/README.md)
- [../adr/README.md](../adr/README.md)

## Validation Summary

- `make agent-validate`
- `make agent-report ENV=cpu CHANGED_FILES='tools/agent/skill-registry.yaml,docs/agent/skill-catalog.md,tools/agent/scripts/agent_skill_validation.py'`
- `make agent-report ENV=cpu CHANGED_FILES='dashboard/frontend/src/pages/topology/TopologyPageEnhanced.tsx,dashboard/frontend/src/components/HeaderReveal.tsx'`
- `make agent-report ENV=cpu CHANGED_FILES='dashboard/backend/handlers/status_collectors.go,dashboard/frontend/src/pages/topology/TopologyPageEnhanced.tsx'`
- `make agent-report ENV=cpu CHANGED_FILES='deploy/operator/controllers/router_controller.go,deploy/kubernetes/response-api/redis.yaml,src/semantic-router/pkg/dsl/compiler.go'`
- `make agent-report ENV=cpu CHANGED_FILES='src/semantic-router/pkg/decision/tree.go,src/semantic-router/pkg/modelselection/selector.go'`
- `make agent-pr-gate` reached the repo-native PR baseline but failed in `golang-lint` because `proxy.golang.org` returned `unexpected EOF` while fetching `github.com/valkey-io/valkey-glide/go/v2@v2.2.7`
- Heavy lint and full CI gates were intentionally skipped in this loop because the active request explicitly prioritized finishing the cleanup without running the heavier local lint or test stack.
