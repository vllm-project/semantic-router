# TD047: Router-Native Access Control Cutover

## Status

In progress.

## Owner Plan

[PL-0038: Router-Native Access Control](../plans/pl-0038-router-native-access-control.md)

## Release Relevance

Not blocking for existing releases while access control remains explicitly
experimental. It is an exit criterion before that product can graduate or be
described as Router-native, Dashboard-optional, or globally enforced across replicas.
Implementation must move to an active execution or release plan before graduation.

## Scope

Inference authentication and authorization, API-key lifecycle, model visibility,
global quota admission and settlement, management identity, usage ingestion,
request-log retention, Dashboard integration, and removal of the old enforcement
paths.

## Summary

The target is specified by
[Router-Native Access Control and Quota Accounting](../../../../website/docs/proposals/router-native-access-control.md)
and its normative appendices. The implementation must replace every old authority;
it must not add a compatibility layer beside them.

## Evidence

- Router-owned domain, persistence, publication, admission, settlement, and
  Management API seams live under `src/semantic-router/pkg/access*`,
  `pkg/quotaruntime`, `pkg/usage*`, and `pkg/managementapi`.
- `make management-api-contract-check` verifies the generated Management API
  artifacts consumed by the Dashboard.
- `dashboard-managed-access-lifecycle` covers authentication, scoped model
  discovery, cross-replica quota enforcement, rotation, disablement, and actual
  usage settlement through public interfaces.
- Docker, Helm, and operator contracts have focused tests. The debt remains open
  until the complete release validation matrix and operator cutover record satisfy
  every exit criterion below.

## Why It Matters

Dashboard is an optional control-plane client. Authentication, model visibility,
quota correctness, and accounting must remain identical when the Dashboard is
absent, when clients call inference directly, and when Router replicas scale
independently. Ten thousand dynamic keys cannot be represented in static Router or
Kubernetes configuration.

## Desired End State

The Router owns a versioned Management API, PostgreSQL desired state and ledger,
Valkey credential/policy projections, exact global request counters, actual-token
settlement, usage stream, and audit. Public inference always passes through the same
AccessRuntime. Dashboard, CLI, and custom consoles use generated Management clients;
Playground uses a short-lived delegated credential and the public inference path.

Routing persists only Model, Recipe, and Entrypoint. Entrypoint rule actions own
decision assignments. Old Dashboard proxy/enforcement packages, static provider
paths, model bindings, process-local counters, and header-selected identity are
removed at an explicit operator cutover.

## Exit Criteria

- Every public inference path enforces the same API-key authentication, model
  discovery/invocation policy, admission, and actual-usage settlement.
- PostgreSQL and Valkey contracts, publication barriers, recovery, Docker, and
  Kubernetes behavior meet the normative proposal and its validation matrix.
- Management identity exchange, permissions/scopes, User/Team/key ownership, Team
  inheritance, self-service onboarding, and delegated inference pass API-level
  tests.
- Usage and quota remain correct for streaming, retries, disconnects, and internal
  multi-dispatch execution across multiple Router replicas.
- Dashboard contains no authoritative access store or public inference proxy and can
  be removed without changing data-plane behavior.
- Model-binding, legacy access API, static enforcement, compatibility, and duplicate
  configuration paths are absent from code, generated schemas, tests, and docs.
- A non-secret operator cutover record proves resource counts, explicit mappings or
  resets, effective-policy equivalence, credential verification, quota cutover,
  usage totals, and secret redaction. The Router keeps no runtime compatibility
  reader for historical Dashboard schemas.
