---
name: project-change
category: primary
description: Handles a focused repository change when no specialized primary skill applies. Use when changed-file routing selects this fallback for a feature, fix, refactor, documentation update, or subsystem-local task.
---

# Project Change

## Trigger

- Changed-file routing selects this fallback because no specialized
  cross-surface workflow applies.

## Workflow

1. Read the agent report and the nearest `AGENTS.md` for the files being changed.
2. Inspect the current implementation and complete the requested outcome within its natural boundary.
3. Run the smallest relevant checks named by the report; widen validation only when behavior or contracts cross surfaces.

## Gotchas

- The report is guidance for finding affected contracts, not a ban on necessary implementation work.
- Keep the change focused, but update dependent code, tests, docs, or tracked debt when the contract genuinely crosses a boundary.
- Stop only for a real product decision, missing authority, or unavailable environment that prevents meaningful progress.

## Must Read

- [tools/agent/docs/change-surfaces.md](../../../../tools/agent/docs/change-surfaces.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`

## Acceptance

- The requested behavior is complete.
- The smallest relevant validation passes, and affected contracts remain aligned.
