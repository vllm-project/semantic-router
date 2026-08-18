# Maintainer Ops

Maintainer ops turns an active release plan and current GitHub state into a
local review board. It is read-only unless a maintainer separately reviews and
applies a proposed action.

## Why This Exists

Release intent, architecture debt, and changing GitHub state have different
lifecycles. The local board gives maintainers one current view without copying
daily issue and pull-request state into versioned plans.

The canonical state split is:

- release intent lives in `tools/agent/docs/plans/`
- architecture gaps live in `tools/agent/docs/tech-debt/`
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

## Built-in Model Catalog Releases

`config/recipes/built-in/latest/` is the authoring source for the catalog that
ships with `vllm-sr`. The package mirror under
`src/vllm-sr/cli/model_assets/latest/` is generated; update it with
`tools/release/sync_model_catalog.py` rather than editing it directly.

Immediately before a stable `vX.Y.Z` tag, create the matching catalog snapshot:

```bash
make built-in-model-snapshot RELEASE_VERSION=X.Y.Z
```

The command creates `config/recipes/built-in/vX.Y/`, updates its release
metadata and bundle digests, and generates the matching package resources. It
refuses to overwrite an existing snapshot. Commit both generated trees in the
release-preparation change.

Before tagging, verify the version contract and source/package parity. Published
snapshots are release inputs and must not be rewritten; policy changes belong
in `latest` or a new catalog version. User-facing Model Cards should explain
catalog versions and compatibility without reproducing these release steps.

## Release Promotion

Stable releases are created from an explicitly reviewed candidate; nightly
artifacts are never promoted automatically.

1. Confirm the candidate commit passes the required CI and release checks.
2. Update the repository's version-bearing surfaces and validate their shared
   version contract.
3. Create the matching built-in catalog snapshot as described above.
4. Push the reviewed `v<version>` tag to start the canonical Docker, Helm,
   Python, crate, and Operator publishers.
5. Verify every publisher before treating the GitHub release as complete.

Fleet Simulator uses its own package version and tag stream. Keep that release
independent from the main Router version unless a documented compatibility
constraint requires coordinated updates.

## Commands

```bash
python3 tools/agent/scripts/maintainer_board.py sync --milestone "MILESTONE_NAME"
python3 tools/agent/scripts/maintainer_board.py brief
python3 tools/agent/scripts/maintainer_board.py release-report \
  --release-plan tools/agent/docs/plans/RELEASE_PLAN.md --write
python3 tools/agent/scripts/maintainer_board.py create-issues \
  --release-plan tools/agent/docs/plans/RELEASE_PLAN.md --dry-run
```

`sync` requires the GitHub CLI to be authenticated. `brief` and
`create-issues` run from the latest local snapshot. Issue creation proposes only
release-plan tasks that do not already match an open milestone issue unless
`--include-matched` is passed explicitly.

## Automation Prompt Template

```text
Run semantic-router maintainer ops for MILESTONE_NAME. Use
tools/agent/docs/maintainer-ops.md and the maintainer release skill. Sync GitHub
issues, PRs, milestones, labels, review state, and CI state. Regenerate
.agent-harness/maintainer/current.json, today.md, milestone notes,
release-readiness.md, and proposed-actions.json. Compare the active release
plan with the milestone and summarize blockers, missing issues, PRs needing
review, PRs needing rebase, close candidates, and the next coding-agent tasks.
Do not mutate GitHub.
```

## Scheduled CI Workflow

`.github/workflows/maintenance.yml` runs the read-only maintainer board on its
daily cadence and can select it through `workflow_dispatch`. The lifecycle
workflow invokes the reusable `.github/workflows/maintainer-board.yml`, which calls
`tools/agent/scripts/run_maintainer_board_ci.sh`, which wraps
`maintainer_board.py sync` and publishes:

- the GitHub Actions job summary (`today.md`)
- downloadable artifacts: `today.md`, `current.json`,
  `proposed-actions.json`, and milestone notes

The scheduled workflow does not label, comment on, or close issues or pull
requests. `proposed-actions.json` is informational only in CI; use the local
`apply` command after maintainer review when mutations are intended.

### Relationship to `stale.yml`

- `.github/workflows/stale.yml` is a reusable job invoked by
  `.github/workflows/maintenance.yml`; it mutates GitHub directly by marking inactive
  issues and pull requests as stale and closes them after the grace period.
- `.github/workflows/maintainer-board.yml` is visibility-only: it classifies
  the current queue using `tools/agent/maintainer-policy.yaml` and gives
  maintainers a daily brief without changing GitHub state.

Use the maintainer board to decide what needs review, rebase, unblock, or
close-candidate follow-up. Use `stale.yml` only for the automated stale/close
lifecycle.

Manual trigger example:

```bash
gh workflow run maintenance.yml -f task=board -f milestone=MILESTONE_NAME
```

## Apply Policy

GitHub mutations are never implicit. Applying proposed labels, comments, issue
creation, or close actions requires an explicit apply command and a maintainer
review of the generated payload.
