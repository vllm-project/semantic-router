# TD048: Dashboard Jobs Lack Aggregate Lifecycle Ownership

## Status

Open

## Owner Plan

PL0038 Router Hardening Audit

## Release Relevance

High - durable control-plane resource ownership

## Scope

Dashboard Evaluation and ML job metadata, progress events, artifact storage,
recovery, pagination, retention, cancellation, and aggregate quota.

## Summary

The Dashboard now bounds individual Evaluation and ML requests, active jobs,
live progress streams, subprocess output, uploads, and per-job artifacts. It
also stores private job-owned training snapshots instead of identifying every
training result through one mutable deployment directory. The durable SQLite
rows and artifact trees still have no complete aggregate lifecycle owner:
completed history can grow indefinitely, an interrupted process cannot resume
or deterministically reconcile in-flight work, and list/result APIs do not all
share cursor pagination, retention, and tenant or installation quotas.

## Evidence

- `dashboard/backend/handlers/evaluation.go` and
  `dashboard/backend/handlers/evaluation_lifecycle.go` own process-local run,
  cancellation, and SSE admission while task and result rows remain durable.
- `dashboard/backend/evaluation/db.go` and the Evaluation result directory have
  per-operation limits and private storage, but no aggregate expiration or
  restart reconciliation policy.
- `dashboard/backend/mlpipeline/runner.go`,
  `dashboard/backend/mlpipeline/runner_admission.go`, and
  `dashboard/backend/workflowstore/**` persist jobs and progress while active
  execution remains process-local.
- `dashboard/backend/mlpipeline/runner_train_artifacts.go` gives each
  successful training job a private immutable snapshot, but no collector owns
  the total number or total bytes of historical job directories.

## Why It Matters

Per-request limits prevent one request from exhausting the service, but they do
not bound long-running disk or database growth. A restart can also leave a
durable `running` record without a live worker, and unpaginated history makes
the cost of an otherwise authorized read depend on installation age. These are
resource, operability, and data-lifecycle failures rather than reasons to
weaken the newly added request-level controls.

## Desired End State

One durable job repository owns Evaluation and ML state transitions, leases or
restart reconciliation, cursor-based reads, cancellation, artifact references,
retention, and installation or tenant quotas. Artifact deletion follows a
reference-aware garbage-collection transaction, and every terminal state is
recoverable without depending on an in-process map.

## Exit Criteria

- Startup deterministically reconciles every persisted non-terminal job and
  proves crash behavior with restart tests.
- Task, result, progress, and artifact list APIs use stable cursor pagination
  with hard page limits.
- Operator-configured age, row-count, and byte quotas cover SQLite records and
  artifact trees, with observable fail-closed admission at the limit.
- Cancellation and terminal transitions are idempotent across process restart.
- Garbage collection deletes only unreferenced artifacts and has fault,
  concurrent-download, and interrupted-collection tests.
- Single-replica SQLite and any future shared database implementation pass the
  same lifecycle contract suite.
