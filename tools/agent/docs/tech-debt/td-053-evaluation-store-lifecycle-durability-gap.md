# TD053: Evaluation Store Lifecycle Durability Gap

## Status

Open

## Owner Plan

[PL-0039: Evaluation Plane](../plans/pl-0039-evaluation-plane.md)

## Release Relevance

Evaluation runs can be retained in controlled development and CI stores. A
long-running multi-tenant Dashboard requires retention, garbage collection,
and crash-durability work before evaluation storage is considered operationally
complete.

## Scope

- content-addressed object reference accounting and garbage collection
- retention, quotas, and protected evidence holds
- directory fsync and terminal-state error propagation
- evidence ownership, deletion authority, and lifecycle audit records

## Summary

Deleting a run removes its index and run directory but does not reclaim
unreferenced content-addressed objects. Atomic rename protects readers from
partial files, but parent directories are not fsynced and some terminal append
errors cannot be surfaced after worker exit. Ordinary evaluation writers can
also delete completed evidence because runs do not yet carry an owner, evidence
hold, or auditable delete authorization decision.

## Evidence

- The store has immutable objects and atomic mutable indexes but no mark-and-
  sweep or reference ledger.
- Large private records can retain substantial disk space after run deletion.
- Crash consistency and lifecycle authorization are not covered by
  fault-injection or multi-principal audit tests.
- The SSE path does persist numeric event cursors, honors `Last-Event-ID`, and
  deduplicates replay in the browser; lifecycle authorization and audit remain
  independent unresolved store concerns.

## Why It Matters

Evaluation evidence is intentionally rich and can grow much faster than normal
Dashboard state. Unbounded orphan retention can exhaust disk, while an
acknowledged terminal state that was not durably persisted undermines audit and
comparison guarantees.

## Desired End State

The store has explicit retention classes, quotas, evidence holds, safe object
reference accounting, ownership-aware lifecycle authorization, an append-only
audit trail, and crash-consistent transitions.

## Exit Criteria

- Add protected retention policies, per-run/store quotas, usage reporting, and
  a dry-run/apply garbage collector that never removes referenced objects.
- Fsync files and parent directories at required manifest, index, and terminal
  state boundaries; propagate persistence failures to the owner process.
- Record the creating principal and policy decision for create, start, cancel,
  compare, hold, release, export, and delete; require owner or administrator
  authority for destructive lifecycle operations.
- Add power-loss/fault-injection tests for create, append, finalize, cancel,
  compare, delete, and garbage collection.
- Validate bounded disk growth across repeated large-run create/delete cycles.
