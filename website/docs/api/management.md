---
title: Management API contract
---

The Router Management API is the versioned control-plane contract for Models,
Recipes, Entrypoints, identities, API keys, policy, quota, usage, logs, Agent
sessions, and operations. The Dashboard is one client of this API; an independent
console or automation service can use the same contract without running the
Dashboard.

## Choose the schema

The Router registry generates two checked artifacts in every source release:

- [OpenAPI 3.1 document](/openapi/management/v1/management.openapi.json) for
  standard client generators and API tooling.
- `dashboard/frontend/src/generated/managementApiContract.ts`, the compact
  TypeScript operation catalog used by the Dashboard. It contains the canonical
  base path, media type, protocol headers, HTTP methods, operation IDs, and a
  typed path builder.

The checked artifacts and the live Management listener's `GET /openapi.json` are
generated from the same Router registry. Use the checked document to build a
client for a release; use the live document to confirm the contract exposed by
a running deployment.

```bash
curl --fail --silent --show-error \
  https://router.example.com/openapi.json \
  --output management.openapi.json
```

The version 1 resource base is `/management/v1`. JSON requests and responses
use `application/vnd.vllm-semantic-router.management.v1+json`.

## Build a custom console

A custom console should authenticate its operator to the Router, select one
authorized namespace, and call the Management listener directly. Do not use an
inference API key for Management operations; inference and control-plane
credentials are deliberately separate.

For namespace-owned resources, send the namespace selected from `GET
/management/v1/me`:

```http
VLLM-SR-Namespace: 4ff7db8d-f659-4a31-96d7-d63d36754ff2
```

For a mutation:

1. Generate a unique `Idempotency-Key` when the operation declares
   `x-router-idempotency: required`.
2. Read the current resource and retain its `ETag`.
3. Send that value in `If-Match` when the operation declares
   `x-router-revision: if_match_and_returns_etag`.
4. Treat `202 Accepted` as an operation resource, then poll its generated
   operation path until it reaches a terminal state.
5. Never persist one-time secrets from create, rotate, reveal, invitation, or
   delegated-session responses in application logs.

The generated operation metadata exposes these requirements without copying a
second route table into the console. Request and response schemas, permissions,
pagination, and secret behavior remain in OpenAPI extensions beside each
operation.

## Build an Agent client

Agent Profiles, Skills, Tools, connections, sessions, turns, events, artifacts,
and publication plans are ordinary Management resources under `/management/v1`.
The session is durable Router state; a browser or custom console does not run the
Agent loop itself.

Create a session with an authorized request-facing Model or Entrypoint, then create
turns with a fresh idempotency key. Resume its event stream with `Last-Event-ID`.
If retained events are no longer available, the Router returns the checkpoint from
which the client can reload the bounded history. Cancellation and publication are
separate authenticated operations.

A model-callable tool may prepare a publication plan, but it cannot commit one.
The human client submits the reviewed plan digest and current ETag to the publication
commit endpoint. The Router reauthorizes the actor and verifies every pinned
resource revision before changing the active snapshot.

## Keep generated clients in sync

After changing a Management registry operation or schema, regenerate and check
both artifacts:

```bash
make management-api-contract-generate
make management-api-contract-check
```

The check is a byte-for-byte drift gate. A route, method, media type, header, or
schema change cannot land while the committed OpenAPI and Dashboard operation
catalog still describe the old contract.

See [Router listener API](./apiserver) for listener modes, TLS, health, and the
live schema endpoint. See [Access and usage](../tutorials/global/access-and-usage)
for the User, Team, API key, policy, quota, and accounting workflow, and
[Build a Mixture in Playground](../tutorials/global/playground-builder) for the
Agent Builder experience.
