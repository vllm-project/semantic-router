# TD047: Workflows and Ratings Still Duplicate Panel-Execution Semantics

## Status

Open. Fusion and ReMoM migrated onto the shared panel executor; Workflows and
Ratings still carry their own independent goroutine/semaphore/collector
implementations.

## Owner Plan

[PL0032 Architecture Debt Consolidation](../plans/pl-0032-architecture-scorecard-ratchet.md)

## Release Relevance

Not release-blocking. Issue #2856 ("unify Looper candidate-panel execution
semantics") explicitly prioritized Fusion and ReMoM migration first; this
entry tracks the deferred remainder so it isn't lost.

## Scope

- `src/semantic-router/pkg/looper/workflows.go`
  (`executeWorkflowStep`/`startWorkflowStepWorkers`/`workflowStepCollector`)
- `src/semantic-router/pkg/looper/ratings.go` (`Execute`'s inline
  goroutine/`sync.WaitGroup` fan-out)
- `src/semantic-router/pkg/looper/panel_executor.go` (`RunPanel`, the shared
  executor Fusion and ReMoM now use)

## Summary

`RunPanel` (added in #2856) now owns concurrency limiting, quorum, timeout,
cancellation, and deterministic ordering for Fusion's and ReMoM's panel
dispatch. Workflows' concurrent step-worker path
(`startWorkflowStepWorkers`/`workflowStepCollector`) and Ratings' `Execute`
still have their own independent, hand-rolled versions of the same pattern,
with the same kind of behavioral quirks the original issue flagged.

## Evidence

- Workflows' concurrent path already matches Fusion's shape closely (same
  cancel-aware semaphore acquire, same quorum-triggers-cancel behavior,
  deterministic index-ordered results) - the mechanical part of migrating it
  onto `RunPanel` should be low-risk. It additionally has a non-concurrent
  `executeWorkflowStepSequential` fallback (used whenever a step involves
  tool calls) that has no counterpart in `RunPanel` and would need to stay
  outside the executor, or `RunPanel` would need a genuine
  `MaxConcurrent==1`-equivalent mode audited against that path's specific
  tool-interrupt semantics.
- Ratings' semaphore acquire (`sem <- struct{}{}`, ratings.go) is a plain
  blocking send, not cancellation-aware like the other three algorithms -
  migrating it fixes that gap the same way this PR fixed ReMoM's ordering
  and cancellation gaps, but Ratings also has no timeout or quorum concept
  today (`wg.Wait()`s for all calls unconditionally) and its `OnError=fail`
  handling is a post-hoc check after every call has already completed,
  not `RunPanel`'s abort-in-progress `FailFast`. Migrating it is a genuine
  behavior change to decide deliberately, not a mechanical port.

## Why It Matters

Until these two migrate, the codebase has three different concurrent-panel
implementations instead of one (`RunPanel` plus the two originals) - the
duplication and inconsistency risk the original issue raised is only half
resolved. Ratings in particular still cannot be preempted while queued
behind its concurrency limit.

## Desired End State

Workflows' concurrent step-worker path and Ratings' `Execute` both dispatch
through `RunPanel`, with Workflows' sequential tool-bearing path either left
as an explicitly-documented exception or given its own deliberate design,
and Ratings' `OnError=fail` semantics reconciled with `FailFast` as an
explicit, reviewed decision rather than a silent behavior change.

## Exit Criteria

- Workflows' concurrent step-worker path migrated onto `RunPanel`, with
  `workflowStepCollector` deleted and existing Workflows tests passing
  unmodified.
- A written decision (in the migration PR description, at minimum) on how
  `executeWorkflowStepSequential` relates to the shared executor.
- Ratings migrated onto `RunPanel`, with an explicit decision on `OnError`
  semantics (abort in progress vs. today's post-hoc check) called out and
  reviewed rather than silently changed.
- This file retired once both land.
