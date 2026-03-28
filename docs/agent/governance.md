# Agent Harness Governance

This document defines how the repository's agent rules are layered and maintained.

## Rule Layers

- `AGENTS.md`
  - short entrypoint only
  - tells agents what to read first, what commands exist, and where the real source of truth lives
- `docs/agent/*`
  - human-readable system of record
  - owns repo map, environments, change surfaces, testing expectations, and governance
- `tools/agent/*.yaml`, `tools/agent/scripts/*`, `tools/make/agent.mk`, `.github/workflows/*`
  - executable rule layer
  - owns matching, validation, gating, and CI behavior
- nearest local `AGENTS.md`
  - module-boundary supplements for hotspots or special subtrees
- `CONTRIBUTING.md`, the PR template, issue templates, and `.prowlabels.yaml`
  - contributor-facing wrappers and label taxonomy around the shared harness

## Source of Truth Policy

- Define each rule once.
- Keep summary guidance in `AGENTS.md`, not the full rule body.
- Put durable prose guidance in `docs/agent/*`.
- Put important repeated rules into executable checks when possible.
- Keep task-first context disclosure executable through `tools/agent/context-map.yaml` and `make agent-report`, not as a second prose-only handbook.
- Keep gitignored local harness outputs under `.agent-harness/` as derived helper artifacts only; they must not replace canonical plans, ADRs, debt entries, or indexed docs.
- If prose and executable rules disagree, fix them in the same change.
- Record durable unresolved gaps in the debt entry files indexed from [tech-debt/README.md](tech-debt/README.md), and keep [tech-debt-register.md](tech-debt-register.md) as the landing page for that workflow.
- If desired architecture and current implementation still diverge after a change, promote that gap there instead of leaving it only in PR text or chat.
- Record long-horizon loop execution in [plans/README.md](plans/README.md) and the plan files it indexes.
- Record durable harness decisions in [adr/README.md](adr/README.md) and the ADR files it indexes.
- Do not use ADRs for temporary execution plans, one-off migrations, or debt items that have not yet reached a durable decision.

## Default Task Loop

- Every task follows the canonical loop surfaced by `make agent-report`.
- Start from the resolved skill and context, make the smallest viable change, run the applicable gate, and treat failing runs as continuation signals instead of handoff points.
- `tools/agent/task-matrix.yaml` classifies rule sets as `lightweight` or `completion`.
- `lightweight` tasks close when the applicable gates for the current change pass or a real external blocker is recorded.
- `completion` tasks close when the active subtask and its applicable gates pass; if the work spans multiple ordered loops or must resume across sessions, create or update an execution plan.
- Stop a loop early only for real external blockers or when another unfinished in-scope subtask can advance while the blocker is pending.

## What Does Not Belong in the Canonical Harness

- one-off CI repair loops
- branch-local cleanup backlogs
- temporary migration checklists
- notes that only make sense for one contributor or one PR
- raw generated report or handoff files under `.agent-harness/`

These can exist as PR descriptions, issue notes, or temporary working docs, but they must not be listed as default entrypoints in `AGENTS.md`, `repo-manifest.yaml`, or `task-matrix.yaml`.

## Maintenance Rules

- Every canonical agent doc must be reachable from [README.md](README.md) or [AGENTS.md](../../AGENTS.md).
- Every canonical executable rule must be reachable from [repo-manifest.yaml](../../tools/agent/repo-manifest.yaml).
- Every task-first context reference must point back to a canonical doc, skill, or executable rule instead of creating a parallel source of truth.
- Canonical docs and local rule supplements carry machine-readable stewardship and freshness metadata in `repo-manifest.yaml`.
- Keep `AGENTS.md` short enough to be a navigation entrypoint instead of a handbook.
- When a repeated failure mode appears, prefer adding a script, Make target, or CI check over adding more prose.
- When a document stops being durable shared guidance, remove it from the canonical layer.
- Keep technical debt discoverable through the landing page and the indexed `docs/agent/tech-debt/*.md` entry files.
- When a code/spec mismatch is real but not fixed yet, treat it as tracked technical debt and update the matching debt entry in the same change that documents the gap.
- Keep execution plans task-focused and resumable instead of turning them into a second architecture handbook.
- Keep ADRs small, decision-focused, and indexed from `docs/agent/adr/README.md` instead of turning them into implementation handbooks.

## Standard Validation Entry

- `make agent-validate`
- `make agent-scorecard`
- `make agent-report ENV=cpu|amd CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`

Use `make agent-validate` for harness-only changes before wider gates. Use `make agent-scorecard` when you need a compact snapshot of canonical docs, local rules, skills, and current validation status.
