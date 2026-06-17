# Execution Plans

Execution plans are the repo's current execution ledger. They are not an
archive and they are not a second architecture handbook.

## Maintainer Model

- Every active release has exactly one release plan.
- Non-release architecture debt is tracked by exactly one current debt plan.
- Work that is not owned by a current plan is not current work.
- Plan files in this directory describe only current work.

## When to Use an Execution Plan

- Use a release plan when maintainer-approved work belongs to a named release or
  milestone.
- Use the debt plan when architecture debt matters but is not release-bound.
- Use a plan only when the work needs multiple resumable loops or multiple
  contributors.

## What Belongs in an Execution Plan

- release or debt objective
- scope boundaries and non-goals
- linked technical debt entries and related governance docs
- issue or milestone ownership where applicable
- stable task IDs
- current loop state and next action
- exit criteria

## What Does Not Belong in an Execution Plan

- completed workstreams
- branch-local notes
- one-off bug triage
- unresolved architecture gaps that belong in TD entries
- daily GitHub issue or PR status; that belongs under `.agent-harness/maintainer/`

## Execution Plan Versus Other Governance Files

- `docs/agent/plans/*.md`: current execution state.
- `docs/agent/tech-debt/*.md`: unresolved architecture gaps.
- `.agent-harness/maintainer/*`: local, gitignored issue and PR operating
  board.

Rule of thumb:

- if the repo is executing it now, use a plan
- if the repo knows the gap, use TD
- if the state changes daily, keep it out of git

## Execution Plan Template

Every execution plan should include:

- `# <title>`
- `## Goal`
- `## Scope`
- `## Exit Criteria`
- `## Task List`
- `## Next Action`
- `## Operating Rules`
- `## Related Docs`

Tasks use stable IDs and checkbox status:

- `- [ ]`
- `- [x]`

## Current Release Plans

- [pl-0033-v0-3-themis-release-closure.md](pl-0033-v0-3-themis-release-closure.md)

## Current Debt Plans

- [pl-0032-architecture-scorecard-ratchet.md](pl-0032-architecture-scorecard-ratchet.md)

## Current Execution Plans

- [pl-0033-v0-3-themis-release-closure.md](pl-0033-v0-3-themis-release-closure.md)
- [pl-0032-architecture-scorecard-ratchet.md](pl-0032-architecture-scorecard-ratchet.md)
- [pl-0034-fusion-api.md](pl-0034-fusion-api.md)
