# Execution Plans

This directory stores canonical execution plans for long-horizon work that needs repeated loop execution across multiple sessions or contributors.

## When to Use an Execution Plan

- Use an execution plan when the work cannot be completed reliably in one pass and needs a durable file-backed loop.
- Use an execution plan when an agent should be able to resume from the repo alone, without relying on chat memory.
- Use an execution plan when the work has ordered tasks, loop-level progress, or a durable completion boundary.

Typical examples:

- multi-step harness refactors
- large architectural cleanup programs
- staged subsystem migrations
- extended testing or coverage rationalization loops

## What Belongs in an Execution Plan

- the long-horizon goal
- scope boundaries
- exit criteria
- a stable task list with IDs
- the current loop state
- brief decision log notes for the loop
- links to related ADRs or technical debt items

## What Does Not Belong in an Execution Plan

- one-off branch scratch notes
- private contributor reminders
- durable architectural decisions that belong in ADRs
- unresolved architecture gaps that belong in the debt register

## Execution Plan Versus Other Governance Files

- `docs/agent/plans/*.md`
  - track active long-horizon execution loops
- `docs/agent/adr/*.md`
  - record durable decisions
- `docs/agent/tech-debt-register.md` and `docs/agent/tech-debt/*.md`
  - record the landing page plus detailed entries for durable unresolved gaps
- `docs/agent/README.md`
  - indexes the canonical harness

Rule of thumb:

- if the repo is still executing, use a plan
- if the repo has decided, use an ADR
- if the repo knows the gap but has not retired it, use the debt register

## Execution Plan Template

Every execution plan should include:

- `# <title>`
- `## Goal`
- `## Scope`
- `## Exit Criteria`
- `## Task List`
- `## Current Loop`
- `## Decision Log`
- `## Follow-up Debt / ADR Links`

Tasks should use stable IDs and explicit status markers such as:

- `- [ ]`
- `- [x]`

Prefer one active execution plan per workstream.

Use file names such as `pl-0001-example.md`.
Keep the numeric index unique within `docs/agent/plans/`.

## Current Execution Plans

- [pl-0001-harness-roadmap.md](pl-0001-harness-roadmap.md)
- [pl-0002-e2e-rationalization-roadmap.md](pl-0002-e2e-rationalization-roadmap.md)
- [pl-0003-v0-3-config-contract-rollout.md](pl-0003-v0-3-config-contract-rollout.md)
- [pl-0004-fleet-sim-migration.md](pl-0004-fleet-sim-migration.md)
- [pl-0005-skill-taxonomy-refresh.md](pl-0005-skill-taxonomy-refresh.md)
- [pl-0006-design-elegance-ratchet.md](pl-0006-design-elegance-ratchet.md)
- [pl-0007-hotspot-boundary-ratchet.md](pl-0007-hotspot-boundary-ratchet.md)
- [pl-0008-dashboard-backend-and-go-config-boundary-ratchet.md](pl-0008-dashboard-backend-and-go-config-boundary-ratchet.md)
- [pl-0009-fleet-sim-optimizer-and-operator-config-boundary-ratchet.md](pl-0009-fleet-sim-optimizer-and-operator-config-boundary-ratchet.md)
- [pl-0010-extproc-response-and-dashboard-frontend-boundary-ratchet.md](pl-0010-extproc-response-and-dashboard-frontend-boundary-ratchet.md)
- [pl-0011-runtime-and-dashboard-state-durability-and-telemetry-ratchet.md](pl-0011-runtime-and-dashboard-state-durability-and-telemetry-ratchet.md)
- [pl-0012-dsl-conflict-free-routing-workstream.md](pl-0012-dsl-conflict-free-routing-workstream.md)
- [pl-0013-openclaw-vsr-install-import-workstream.md](pl-0013-openclaw-vsr-install-import-workstream.md)
