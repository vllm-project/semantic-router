# Maintainer Ops

Maintainer ops is the bridge between release plans and GitHub execution state.

## Why This Exists

The repo has more work than a maintainer can track from memory. Release plans,
GitHub milestones, user issues, bug reports, PR review queues, stale PRs, and
backlog candidates all need one daily operating board.

The canonical state split is:

- release intent lives in `docs/agent/plans/`
- architecture gaps live in `docs/agent/tech-debt/`
- durable operating rules live in the relevant governance docs
- daily issue and PR state lives in `.agent-harness/maintainer/`

## Local Board

Generated files live under the gitignored `.agent-harness/maintainer/`
directory:

- `current.json`: raw issue, PR, milestone, and proposed-action snapshot
- `today.md`: maintainer action brief for the current day
- `milestone-<slug>.md`: milestone-specific issue and PR grouping
- `release-readiness.md`: active release plan versus milestone progress and
  PR blocker summary
- `proposed-actions.json`: write actions that require explicit maintainer apply
- `snapshots/YYYY-MM-DD.json`: daily historical snapshots

These files are local operating artifacts. They are not canonical repo docs and
must not be committed.

## Issue Groups

- `milestone-bound`: assigned to the active milestone
- `milestone-candidate`: labelled as a candidate for the active milestone
- `incoming-triage`: new or unclassified bug/feature/user report
- `backlog`: valuable but not current-release work
- `stale`: inactive or directionally obsolete issue that needs maintainer action

## PR Groups

- `merge-candidate`: approved and green
- `review-now`: ready for maintainer review
- `unblock`: failing, blocked, or waiting on maintainer decision
- `needs-rebase`: dirty or stale against the base branch
- `close-candidate`: inactive or no longer aligned with current mainline

## Release Issue Creation

Seed issues should come from the release plan, not from scattered historical
notes. The default creation mode is dry-run. Public issue bodies must not
include private infrastructure paths, private hosts, local workspace paths, or
AI/tool attribution. Newly created issues receive `help wanted` by default
unless the maintainer explicitly disables it.

Maintainer ops owns two release-management actions that should not appear as
active release-plan tasks:

- Sync GitHub milestone, issue, PR, label, review, and CI state into the local
  board and classify the result by release track.
- Propose missing release seed issues from the active release plan, review the
  dry-run payload, and apply only after explicit maintainer approval.

## Commands

```bash
python3 tools/agent/scripts/maintainer_board.py sync --milestone "v0.3 - Themis"
python3 tools/agent/scripts/maintainer_board.py brief
python3 tools/agent/scripts/maintainer_board.py release-report --release-plan docs/agent/plans/pl-0033-v0-3-themis-release-closure.md --write
python3 tools/agent/scripts/maintainer_board.py create-issues --release-plan docs/agent/plans/pl-0033-v0-3-themis-release-closure.md --dry-run
```

`sync` requires the GitHub CLI to be authenticated. `brief` and
`create-issues` run from the latest local snapshot. Issue creation proposes only
release-plan tasks that do not already match an open milestone issue unless
`--include-matched` is passed explicitly.

## Daily Cron Prompt

```text
Run semantic-router maintainer ops for v0.3. Use docs/agent/maintainer-ops.md
and the maintainer release skill. Sync GitHub issues, PRs, milestones, labels,
review state, and CI state for the v0.3 milestone. Regenerate
.agent-harness/maintainer/current.json, today.md, milestone notes,
release-readiness.md, and proposed-actions.json. Compare the active release
plan with the milestone and summarize blockers, missing issues, PRs needing
review, PRs needing rebase, close candidates, and the next coding-agent tasks.
If the active release includes session-aware agentic routing, run the GA
readiness report with `--allow-blockers`; the maintainer board will include the
latest GA blocker summary in today.md and release-readiness.md, with a link to
the generated `ga-readiness.json` for details.
Do not mutate GitHub.
```

## Scheduled CI Workflow

`.github/workflows/maintainer-board.yml` runs the maintainer board on a daily
schedule and via `workflow_dispatch`. The sync job calls
`tools/agent/scripts/run_maintainer_board_ci.sh`, which wraps
`maintainer_board.py sync` and publishes:

- the GitHub Actions job summary (`today.md`)
- downloadable artifacts: `today.md`, `current.json`,
  `proposed-actions.json`, and milestone notes

The scheduled cron path is read-only: it does not label, comment on, or close
issues or pull requests.

### Optional CI apply mode

Maintainers can optionally apply a previously reviewed
`proposed-actions.json` payload from CI in two steps:

1. Run a read-only sync via `workflow_dispatch` (leave `source_run_id` empty).
   Note the workflow run ID and download or inspect `proposed-actions.json`.
2. After review, run `workflow_dispatch` again with `source_run_id=<sync run ID>`.
   The apply job downloads that artifact and calls
   `tools/agent/scripts/run_maintainer_board_apply_ci.sh`, which validates the
   payload and applies labels through `maintainer_board_ci_apply.py`.

Apply mode rules:

- runs only on manual `workflow_dispatch`, never on the daily cron
- applies only a reviewed artifact from a prior sync run, never from the same
  workflow run that just generated the payload
- CI allowlist is `label_issue` and `label_pr` only; labels must come from
  `tools/agent/maintainer-policy.yaml` lifecycle and PR state mappings
- rejects malformed payloads, non-`#<number>` targets, and synthetic actions such
  as `create_issue`
- reports attempted, succeeded, and failed actions in the job summary even when
  some `gh` commands fail
- does not create, comment on, or close issues or pull requests

### Relationship to `stale.yml`

- `.github/workflows/stale.yml` mutates GitHub directly: it marks inactive
  issues and pull requests as stale and closes them after the grace period.
- `.github/workflows/maintainer-board.yml` sync job classifies the current queue
  using `tools/agent/maintainer-policy.yaml` and gives maintainers a daily brief
  without changing GitHub state by default.

Use the maintainer board to decide what needs review, rebase, unblock, or
close-candidate follow-up. Use `stale.yml` only for the automated stale/close
lifecycle.

Manual trigger examples:

```bash
# Read-only sync
gh workflow run maintainer-board.yml -f milestone=v0.4 -f issue_limit=100 -f pr_limit=50

# Apply a reviewed artifact from sync run 123456789 (maintainer-only)
gh workflow run maintainer-board.yml -f source_run_id=123456789
```

## Apply Policy

GitHub mutations are never implicit. Applying proposed labels, comments, issue
creation, or close actions requires an explicit apply command and a maintainer
review of the generated payload. Local apply remains available via
`maintainer_board.py apply --actions .agent-harness/maintainer/proposed-actions.json --confirm`.
