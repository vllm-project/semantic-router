# TD047: Access Control High-Volume Follow-Ups

## Status

Open.

## Owner Plan

[PL-0032: Architecture Debt Consolidation](../plans/pl-0032-architecture-scorecard-ratchet.md)

## Release Relevance

Not release-blocking for the initial PostgreSQL and Redis multi-replica control
plane. It tracks the next scale boundary for multi-host and sustained
high-volume installations.

## Scope

Dashboard member identity, API-key authorization reads, usage ingestion,
request-log retention, and distributed quota reservations.

## Summary

The control plane now persists inference identities, teams, API keys, access
groups, budgets, usage, and audit records in PostgreSQL and enforces shared
quotas through Redis. Remaining scale work includes a networked Dashboard
member store, partitioned or asynchronous usage ingestion, cursor pagination,
request-id-keyed quota leases, and a revisioned authorization cache.

## Evidence

- Dashboard members, sessions, and one-time invitations still use SQLite, so
  replicas on different hosts cannot share identity state without an external
  store.
- The gateway writes each usage event synchronously to PostgreSQL and reads
  authorization state for every request.
- Usage queries are bounded and indexed, but raw-event retention has no table
  partition lifecycle or pre-aggregated long-range rollups.
- Quota reservation and reconciliation are atomic across replicas, but a
  retried client request is not yet represented by an idempotent lease.

## Why It Matters

The current design favors immediate revocation and accurate accounting. At
large request volumes or across independent hosts, synchronous raw writes,
shared-file Dashboard auth, and uncached policy reads become operational
limits.

## Desired End State

Move Dashboard member identity to a networked transactional store or external
identity provider. Add partitioned ingestion and rollups, cursor-based scans,
idempotent quota leases, and revisioned bounded caches without weakening
immediate revocation or durable per-request accounting.

## Exit Criteria

- Dashboard member invite, session, role, and audit operations work across
  independent hosts without a shared filesystem.
- Usage ingestion and retention remain bounded at sustained high volume and
  long-range views use verified rollups.
- List and log APIs use stable cursors for high-cardinality scans.
- Retried request IDs cannot reserve RPM, TPM, or daily quota twice.
- Authorization caching has explicit revision invalidation and preserves
  immediate key revocation across replicas.
