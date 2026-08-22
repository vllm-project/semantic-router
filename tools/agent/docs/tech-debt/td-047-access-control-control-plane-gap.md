# TD047: Router-Native Access Control Cutover

## Status

Open.

## Owner Plan

[PL-0032: Architecture Debt Consolidation](../plans/pl-0032-architecture-scorecard-ratchet.md)

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

The current implementation owns inference identities, keys, policies, quota, and
public inference proxy behavior in the Dashboard backend, while the Router retains
separate static authorization and rate-limit configuration paths. That boundary lets
a client bypass Dashboard-owned enforcement and prevents one clean contract for
Docker, Kubernetes, CLI automation, and independent consoles.

The target and migration are now specified by
[Router-Native Access Control and Quota Accounting](../../../../website/docs/proposals/router-native-access-control.md)
and its normative appendices. The implementation must replace the current paths; it
must not add a compatibility layer beside them.

## Evidence

- The Dashboard backend owns authoritative access tables and inference gateway
  handlers instead of consuming a Router Management API.
- Direct public Router inference and Dashboard-proxied inference do not yet share one
  mandatory credential, discovery, invocation, quota, and settlement evaluator.
- Dynamic Users, Teams, API keys, access grants, and rate policies do not yet project
  from Router-owned PostgreSQL state into one shared Valkey runtime contract.
- Dashboard browser identity does not yet exchange into a scoped Router
  ManagementPrincipal, and Playground does not yet use delegated inference
  credentials against the public listener.
- Existing Entrypoint model-binding state has not yet been replaced by rule-action
  assignments across configuration, DSL, API, and UI.

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
removed after one verified migration.

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
- The migration report proves resource counts, effective-policy equivalence,
  credential verification, quota cutover, usage totals, and secret redaction.
