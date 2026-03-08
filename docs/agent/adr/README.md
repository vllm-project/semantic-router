# Agent ADRs

This directory stores architecture decision records for the repository's agent harness.

## When to Create or Update an ADR

- Create a new ADR when the harness makes a durable structural decision that should outlive one PR, one contributor, or one refactor loop.
- Update an existing ADR when the original decision still stands but its context, consequences, or status materially changed.
- Replace or supersede an ADR when the harness intentionally changes direction and the previous decision is no longer the current source of truth.

Typical ADR-worthy changes:

- changing harness layering or source-of-truth boundaries
- introducing or retiring a major governance mechanism
- changing how skills, surfaces, validation, or canonical docs are organized
- formalizing a durable decision about what belongs in docs, executable rules, or tracked debt

## What Belongs in an ADR

- stable architectural or governance decisions
- the context that made the decision necessary
- the decision itself
- the long-lived consequences of adopting it
- status changes such as accepted, superseded, or retired

## What Does Not Belong in an ADR

- temporary execution plans for one refactor loop
- branch-local notes or repair checklists
- one-off bug triage
- technical debt items that describe a gap but do not yet record a chosen decision
- implementation details that are better kept in playbooks, code comments, or PR descriptions

## ADR Versus Other Governance Files

- `docs/agent/adr/*.md`
  - record durable harness decisions
- `docs/agent/plans/*.md`
  - record active long-horizon execution loops and current execution state
- `docs/agent/tech-debt-register.md`
  - records known durable gaps between the desired architecture and the current implementation
- temporary execution plans
  - belong in working notes, issues, or PR descriptions until promoted into a durable canonical artifact

Rule of thumb:

- if the repo has already made a long-lived choice, use an ADR
- if the repo has not fixed the gap yet, use the debt register
- if the repo is actively executing a loop, use an execution plan

## ADR Template

Every ADR should include:

- `# ADR NNNN: <title>`
- `## Status`
- `## Context`
- `## Decision`
- `## Consequences`

Use zero-padded numeric prefixes such as `0001-...md`, `0002-...md`.

## Current ADRs

- [0001-harness-layering.md](0001-harness-layering.md)
