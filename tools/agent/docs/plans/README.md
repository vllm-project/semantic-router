# Execution Plans

Execution plans hold only work that is active and needs to resume across
contributors or sessions. They are not release notes, test reports, or an
archive; completed plans remain available in Git history.

## Maintainer Model

- Every active release has one release plan.
- Non-release architecture debt has one current debt plan.
- Other feature or evaluation plans exist only while their multi-step work is
  active.
- Plan files describe current work, not project history.

## When to Use an Execution Plan

Use a plan when work needs multiple ordered loops, contributors, or sessions.
Use a release plan for a named milestone and the debt plan for non-release
architecture cleanup. A small change that can finish in one loop does not need
a plan.

## What Belongs in an Execution Plan

- a goal, scope, and non-goals;
- stable task IDs and current task state;
- exit criteria and one next action;
- links to the durable design, debt, issue, and benchmark sources it depends
  on.

## What Does Not Belong in an Execution Plan

- completed workstreams or validation receipts;
- branch names, commits, pull-request handoff notes, host details, or PID/log
  locations;
- architecture explanations that belong in product or harness documentation;
- benchmark results that belong beside the benchmark artifacts;
- daily GitHub state, which belongs under `.agent-harness/maintainer/`.

## Execution Plan Versus Other Governance Files

- `tools/agent/docs/plans/*.md`: active, resumable execution.
- `tools/agent/docs/tech-debt/*.md`: unresolved architecture gaps.
- `.agent-harness/maintainer/*`: local, changing issue and pull-request state.

Use a plan for current execution, a debt entry for a known gap, and the local
board for state that changes daily.

## Execution Plan Template

```markdown
# PL-XXXX: Title

## Goal

## Scope

## Non-Goals

## Exit Criteria

## Task List

- [ ] `TASK-01` First resumable task

## Next Action

## Operating Rules

## Related Docs
```

## Current Release Plans

None. Create a release plan when maintainers open the next release milestone;
do not revive a completed release plan.

## Current Debt Plans

- [PL-0032: Architecture Debt Consolidation](pl-0032-architecture-scorecard-ratchet.md)

## Current Execution Plans

- [PL-0032: Architecture Debt Consolidation](pl-0032-architecture-scorecard-ratchet.md)
- [PL-0037: Router Flow Evaluation Campaign](pl-0037-router-flow-eval-campaign.md)
- [PL-0038: Dashboard Modeling Experience](pl-0038-dashboard-modeling-experience.md)
- [PL-0039: Evaluation Plane](pl-0039-evaluation-plane.md)
- [PL-0040: MoM Routing Hardening](pl-0040-mom-routing-hardening.md)
- [PL-0041: Agent-Aware Router Contracts (Epic #2994)](pl-0041-agent-aware-router-contracts.md)
