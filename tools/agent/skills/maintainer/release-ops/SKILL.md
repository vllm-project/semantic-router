---
name: maintainer-release-ops
category: support
description: Maintainer release and milestone operating workflow. Use when a maintainer wants to plan a release, assess milestone health, coordinate release blockers, or generate a release-focused review brief.
---

# Maintainer Release Ops

## Trigger

- Use when release planning needs milestone structure, release issue creation,
  or blocker coordination
- Use when the maintainer asks for milestone health or a release-focused brief
- Use `maintainer-issue-pr-management` for day-to-day intake, labels, titles,
  acceptance, and stale-work triage

## Required Surfaces

- `maintainer_ops`

## Stop Conditions

- GitHub writes are requested but the generated action payload has not been
  reviewed by the maintainer
- A public issue, PR, or comment body would include private infrastructure
  details or AI/tool attribution

## Workflow

1. Read the active release plan and maintainer policy.
2. Run `maintainer_board.py sync` to refresh local state, or `brief` to use the
   latest snapshot.
3. Run `release-report` when an active release plan exists, then compare plan
   tasks, milestone issues, and PR blockers before proposing work.
4. Review accepted milestone work and release blockers.
5. Generate proposed actions only; do not mutate GitHub by default.
6. Apply actions only after maintainer confirmation.

## Must Read

- [tools/agent/docs/maintainer-ops.md](../../../../../tools/agent/docs/maintainer-ops.md)
- [tools/agent/maintainer-policy.yaml](../../../maintainer-policy.yaml)
- [.prowlabels.yaml](../../../../../.prowlabels.yaml)
- [.github/PULL_REQUEST_TEMPLATE.md](../../../../../.github/PULL_REQUEST_TEMPLATE.md)

## Standard Commands

- `python3 tools/agent/scripts/maintainer_board.py sync --milestone "<name>"`
- `python3 tools/agent/scripts/maintainer_board.py brief`
- `python3 tools/agent/scripts/maintainer_board.py release-report --release-plan tools/agent/docs/plans/RELEASE_PLAN.md --write`
- `python3 tools/agent/scripts/maintainer_board.py create-issues --release-plan tools/agent/docs/plans/RELEASE_PLAN.md --dry-run`

## Gotchas

- Milestone assignment is a release commitment and requires accepted work;
  there is no separate candidate state label.
- Close-candidate PRs should get an explicit grace-period comment before close
  unless the maintainer asks for immediate closure.
- The generated board is local and gitignored; do not link to local board paths
  from public GitHub artifacts.

## Acceptance

- The local maintainer board groups issues and PRs into actionable buckets
- Public GitHub actions are generated separately from read-only sync
- Release issue creation follows the current release plan and default label
  policy
