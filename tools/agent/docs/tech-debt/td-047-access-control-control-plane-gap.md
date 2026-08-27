# TD047: Control-Plane and Data-Plane Separation

## Status

In progress.

## Owner Plan

[PL-0038: Replaceable Control Plane and Inference Enforcement](../plans/pl-0038-router-native-access-control.md)

## Release Relevance

Not blocking for existing releases while access control remains explicitly
experimental. It is an exit criterion before that product can graduate or be
described as Dashboard-optional or globally enforced across replicas.
Implementation must move to an active execution or release plan before graduation.

## Scope

Envoy dispatch, ExtProc selection and enforcement, API-key lifecycle, model
visibility, global quota admission and settlement, control-plane identity, optional
Agent services, usage ingestion, request-log retention, deployment recovery, and
scale gates.

## Summary

The target contract is specified by
[Access Control and Quota Accounting](../../../../website/docs/proposals/router-native-access-control.md)
and its normative appendices. Runtime code has one authority for each state class and
one publication path into the data plane. The target separates Envoy transport,
ExtProc execution, control-plane desired state, and optional Agent orchestration.

## Evidence

- Product CRUD, PostgreSQL persistence, publication, Management identity, and Agent
  packages are still composed into the Router under
  `src/semantic-router/pkg/managementcomposition`, `pkg/managementserver`,
  `pkg/managementapi`, and `pkg/agent*`. This violates the target dependency
  direction even though the request-time access evaluator is reusable.
- `src/semantic-router/pkg/config/agent_service_config.go` and
  `global.services.agent` still configure Agent behavior in Router YAML.
- The frontend still calls `/api/router/management/v1/agent-*`; the separate
  `/api/agent/v1` service contract is not implemented.
- The bundled backend invoker still lets Router own upstream Model HTTP calls instead
  of returning a logical DispatchPlan to an Envoy/external-gateway adapter.
- `make management-api-contract-check` verifies the generated Management API
  artifacts consumed by the Dashboard.
- `dashboard-managed-access-lifecycle` covers authentication, scoped model
  discovery, cross-replica quota enforcement, rotation, disablement, and actual
  usage settlement through public interfaces.
- Reusable knowledge bases are currently valid only as startup assets declared by
  `global.model_catalog.kbs[]`. The durable routing snapshot and `/management/v1`
  routing resources contain Models, Recipes, and Entrypoints, but no KnowledgeBase
  resource or revision. Router API tests explicitly reject the retired
  `/config/kbs` mutation paths, and Dashboard navigation intentionally omits the old
  file-backed editor. Dynamic KnowledgeBase management is outside PL0038's current
  delivery scope and must not be presented as implemented; it needs a separate
  Router-owned resource proposal and explicit future plan scope before a Build
  surface can return.
- Docker, Helm, and operator contracts have focused tests. The debt remains open
  until the complete release validation matrix satisfies every exit criterion below.

## Why It Matters

Dashboard is the reference replaceable control plane. Authentication, model visibility,
quota correctness, and accounting must remain identical when the Dashboard is
absent, when clients call inference directly, and when Router replicas scale
independently. Ten thousand dynamic keys cannot be represented in static Router or
Kubernetes configuration. Agent and product CRUD in Router make the core binary,
configuration, API, database privileges, and scaling model unnecessarily coupled.

## Desired End State

The control plane owns the versioned Management API, PostgreSQL desired state and
ledger, policy compilation, publication, and audit. ExtProc consumes immutable
routing/access snapshots and owns only request-time verification, selection,
admission, and settlement. Valkey owns applied policy state, global counters, and the
usage stream. Envoy owns upstream transport. Public inference always passes through
the same evaluator.

Dashboard, CLI, and custom consoles use generated control-plane clients. Playground
uses a short-lived delegated credential and the public Envoy inference path. Its
optional Agent API, session store, worker, Skills, Tools, and Builder live in the
control-plane deployment and can be omitted without changing Router configuration.

Routing persists Model, Recipe, and Entrypoint resources. Entrypoint rule actions own
decision assignments; there is no detached Model-to-Recipe association resource or API.
Dashboard is an optional client, API-key identity comes only from Router verification,
and globally enforced counters have one Valkey authority.

## Exit Criteria

- Every public inference path enforces the same API-key authentication, model
  discovery/invocation policy, admission, and actual-usage settlement.
- Router binary and OpenAPI contain no User/Team/key CRUD, invitation, provider
  catalog, Agent session, Skill, Tool, or publication-coordination API.
- PostgreSQL and Valkey contracts, publication barriers, recovery, Docker, and
  Kubernetes behavior meet the normative proposal and its validation matrix.
- Management identity exchange, permissions/scopes, User/Team/key ownership, Team
  inheritance, self-service onboarding, and delegated inference pass API-level
  tests.
- Usage and quota remain correct for streaming, retries, disconnects, and internal
  multi-dispatch execution across multiple Router replicas.
- The reference control plane can be removed or replaced without changing already
  applied data-plane behavior; it is never a public inference proxy.
- ExtProc opens no upstream Model connection. Envoy or an external gateway executes
  the DispatchPlan and returns authenticated attempt/usage evidence.
- Agent code imports only public inference and control-plane contracts; Router code
  imports no Agent package and Router YAML has no Agent field.
- Entrypoint assignments are the only persistent Model-to-Recipe association; code,
  generated schemas, tests, and docs expose no duplicate configuration or
  enforcement path.
- A public-safe operator validation record proves resource counts, effective-policy
  equivalence, credential verification, global quota behavior, usage totals, and
  secret redaction. The serving runtime reads only the target contracts.
