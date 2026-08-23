---
title: API Keys, Access, and Usage
description: Give users and teams controlled access to Mixture-of-Model entrypoints with reusable grants, exact quotas, and attributable usage.
---

# API Keys, Access, and Usage

## Overview

Semantic Router enforces inference access at the Router. The Dashboard is an
optional client for the same Management API, so API keys keep working when the
Dashboard is unavailable and behave identically from an application, the
Playground, Docker, or Kubernetes.

## What Problem Does It Solve?

It gives every model consumer an attributable credential while keeping model
visibility, quotas, and usage consistent across Router replicas.

## Understand the objects

| Object | What it controls |
| --- | --- |
| User | One model-service consumer and their personal defaults. |
| Team | Shared membership, access, and quota defaults. |
| API key | The credential presented to inference APIs and the unit of attribution. |
| Access policy | The Entrypoints and direct Models the subject may discover and invoke. |
| Rate-limit policy | Reusable request, token, cost, or concurrency rules. The Dashboard labels this a Budget. |

An API key has exactly one owner: a User or a Team. A User can also use a Team
context when membership and policy allow it. Key-level policy overrides User
defaults, and User defaults override Team defaults. Removing an override makes
the inherited policy effective again; it does not copy the inherited object.

## When to Use

Use this flow whenever a shared Router serves more than one User, Team, or
application and each caller needs explicit model access or a usage limit.

## Configuration

Create the identity boundary first, then attach access, budget, and credential
resources in that order.

## 1. Create the Team boundary

Open **Access → Identity → Teams** and create the Team. Choose its default
Access policy and Budget during creation, then invite members with an explicit
Dashboard role and Team role.

Dashboard roles control the management product. Team roles control what a
member may do inside that Team. Neither role is an inference credential, and a
Dashboard cookie is never accepted by the public model API.

## 2. Define model access

Open **Access → Policies → Access Groups** and select the published Entrypoints
the subject may use. Grant direct physical Models only to operators who need to
test them. The same grant controls both `/v1/models` discovery and invocation,
so a hidden Model cannot be reached by guessing its name.

Access is additive across applicable grants. Deny barriers for disabled keys,
Users, Teams, memberships, or restrictive policy changes take effect across
Router replicas before the Management API reports success.

## 3. Define a Budget

Open **Access → Policies → Budgets** and add one or more rules. Rules can limit:

- requests;
- input, output, total, or served tokens;
- cost in an explicit currency; and
- concurrent requests.

Choose an algorithm and window that match the product promise. For example, a
sliding cost rule can allow USD 5 every eight hours. Daily and monthly calendar
rules use an explicit timezone. Rules are exact decimal values; currency and
token arithmetic never use floating-point counters.

Token and cost rules settle from the provider's authoritative response usage.
If a request begins while capacity remains and its actual result crosses the
limit, that response completes and the next admission is blocked. Combine an
actual-usage rule with concurrency and generation caps when a strict bound on
in-flight overshoot is required.

## 4. Create an API key

Open **Access → Credentials → API Keys**, choose either **User** or **Team** as
the owner, and select explicit overrides only when this key should differ from
its owner. **Quota** offers three choices: inherit the owner policy, select an
existing Budget, or define custom rules. Custom rules are created atomically as
an ordinary reusable Budget and attached to the new key; they do not introduce a
second kind of limit. The generated secret is shown through the one-time delivery dialog.
Store it in a secret manager; ordinary list responses never contain it.

An administrator can later change the key's Budget allocation or remove the
override so it inherits a larger User or Team Budget. Policy changes preserve
the key identity and its attributed usage history.

Use the key against the public listener:

```bash
curl -sS http://localhost:8899/v1/models \
  -H "Authorization: Bearer $VLLM_SR_API_KEY"
```

```bash
curl -sS http://localhost:8899/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_SR_API_KEY" \
  -H 'content-type: application/json' \
  -d '{
    "model": "acme/assistant",
    "messages": [{"role": "user", "content": "Explain this change."}]
  }'
```

## 5. Verify usage and remaining quota

Open **Access → Usage** for the authorized global, Team, User, or API-key view.
The API-key detail page uses the same exact usage source and shows request,
token, and cost totals alongside each live quota meter. Live quota includes the
current remaining value and reset boundary; analytics expose their own
freshness time because durable rollups may lag the global counter briefly.

Playground requests use a short-lived delegated inference credential tied to
the selected logical key. They pass through the same authentication, grants,
quota, usage ledger, and request-log path as an external application.

## Operate usage storage

Usage facts are stored in aligned UTC-month PostgreSQL partitions. The Router
creates the current and future partitions automatically on every replica. Raw
usage and audit history are kept indefinitely unless an operator explicitly
sets raw usage retention:

```yaml
global:
  services:
    access:
      usage_storage:
        create_ahead_months: 2
        maintenance_interval: 5m
        raw_retention: 2160h # optional: 90 days
```

Monthly partitioning is fixed, so there is no interval knob. An empty
`raw_retention` is the default and deletes nothing. With retention enabled, the
Router removes only complete months whose minute, hour, and day rollups have
finished and which have no inference replay, unresolved usage fence, or pending
reconciliation reference. Aggregates and the settlement digest tombstone
remain; raw request detail for a retired month returns not found. A matching
late stream delivery is acknowledged without counting it again. Each
maintenance pass examines at most 32 old candidates and retires at most one
month; subsequent passes continue automatically.

`/management/v1/runtime-diagnostics` reports active and retired months plus
bounded dirty-rollup queue depths, and marks counts that reach the diagnostic
cap. A maintenance error degrades Router readiness,
while already running usage workers continue ingesting into safe partitions.
Request/response payload capture is disabled by default and must use its own
short, bounded retention when enabled; it is never required for quota, usage,
or audit correctness.

The v0.4 partition layout is a fresh-schema contract. To replace a preview
schema, fence new admission, drain usage-stream pending entries and unknown
reconciliation, export retained usage and audit data, rebuild with
`access-migrate`, import the validated export, and verify settlement/rollup
watermarks before removing the fence. The Router does not carry a runtime
dual-schema compatibility branch.

## Operate safely

- Disable a key to stop new use without deleting its history.
- Rotate a compromised secret with a bounded overlap, then remove the old
  credential.
- Delete only after dependent sessions and retention requirements are resolved.
- Treat incomplete provider usage as an explicit fenced state; never convert it
  silently to zero.
- Keep PostgreSQL and Valkey private. API keys and counters do not belong in
  Router YAML, gateway routes, ConfigMaps, or one Kubernetes resource per key.

## Automate the same lifecycle

An independent console can use the versioned Router Management API to create
Users, Teams, memberships, policies, bindings, and API keys, and to query
usage. Management service accounts and mTLS identities are separate workload
credentials; inference API keys cannot call Management endpoints.

See [Models, Entrypoints, and Serving](models-entrypoints-serving) to publish
the model products referenced by an Access policy.
