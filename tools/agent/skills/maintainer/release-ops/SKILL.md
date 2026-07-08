---
name: maintainer-release-ops
category: support
description: Maintainer release and milestone operating workflow. Use when a maintainer wants to plan a release, create milestone issues, sync GitHub issue or PR state, generate a daily review brief, or manage stale PRs and backlog routing.
---

# Maintainer Release Ops

## Trigger

- Use when release planning needs GitHub milestone, issue, or PR management
- Use when the maintainer asks for a daily board, stale PR review, backlog
  grooming, or milestone health check

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
4. Review issue and PR groups by milestone, lifecycle, and stale status.
5. Generate proposed actions only; do not mutate GitHub by default.
6. Apply actions only after maintainer confirmation.

## Must Read

- [docs/agent/maintainer-ops.md](../../../../../docs/agent/maintainer-ops.md)
- [tools/agent/maintainer-policy.yaml](../../../maintainer-policy.yaml)
- [.prowlabels.yaml](../../../../../.prowlabels.yaml)
- [.github/PULL_REQUEST_TEMPLATE.md](../../../../../.github/PULL_REQUEST_TEMPLATE.md)

## Standard Commands

- `python3 tools/agent/scripts/maintainer_board.py sync --milestone "<name>"`
- `python3 tools/agent/scripts/maintainer_board.py brief`
- `python3 tools/agent/scripts/maintainer_board.py release-report --release-plan docs/agent/plans/pl-0033-v0-3-themis-release-closure.md --write`
- `python3 tools/agent/scripts/maintainer_board.py create-issues --release-plan docs/agent/plans/pl-0033-v0-3-themis-release-closure.md --dry-run`

## Gotchas

- Milestone assignment is a release commitment; `milestone-candidate` is not.
- Close-candidate PRs should get an explicit grace-period comment before close
  unless the maintainer asks for immediate closure.
- The generated board is local and gitignored; do not link to local board paths
  from public GitHub artifacts.

## Acceptance

- The local maintainer board groups issues and PRs into actionable buckets
- Public GitHub actions are generated separately from read-only sync
- Release issue creation follows the current release plan and default label
  policy
