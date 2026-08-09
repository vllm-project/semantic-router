# Security Hardening Guide

This document describes the production-default Envoy configuration shipped with
the semantic router, and the two config surfaces operators most often mistake
for access control: identity-based routing signals and request guardrails.

## Overview

The semantic router sits between external clients and upstream LLM backends.
It processes requests through Envoy's ext_proc filter and makes routing
decisions based on signal classification. This guide covers:

1. **Production-default Envoy config** — defense-in-depth header stripping at
   the proxy layer.
2. **Identity-based routing signals** — `role_bindings` as an input to routing
   decisions, not as model authorization.
3. **Request guardrails** — the local limiter as a per-process traffic
   guardrail, not a distributed quota system.

## Architecture

```
 Client → Envoy (header stripping) → ext_proc → upstream LLM
                                       │
                             config.yaml (routing signals,
                                          decisions, guardrails)
```

## 1. Production-Default Envoy Config

The Envoy template strips sensitive internal headers from client requests
at the proxy layer (defense-in-depth). Even if ext_proc validation is
bypassed, these headers never reach the router:

```yaml
request_headers_to_remove:
  - "x-vsr-looper-request"
  - "x-vsr-looper-secret"
  - "x-vsr-looper-decision"
  - "x-vsr-looper-iteration"
  - "x-authz-user-id"
  - "x-authz-user-groups"
```

This is configured in both:

- `deploy/local/envoy.yaml` (local development)
- `src/vllm-sr/cli/templates/envoy.template.yaml` (production template)

## 2. Identity-Based Routing Signals

`routing.signals.role_bindings` maps request subjects to a role name, so a
decision can branch on who is calling:

```yaml
routing:
  signals:
    role_bindings:
      - name: premium_user
        description: Requests from paid end users.
        role: premium_user
        subjects:
          - kind: Group
            name: premium-tier
```

A decision can then reference that role as a positive routing rule, the same
way it references a domain or complexity signal.

This is a routing input, not deny-by-default authorization. A role binding
decides which decision a request can match; it does not prevent a caller from
reaching a model through some other decision, and it is only as trustworthy as
the identity headers Envoy passes through. Treat model access control as a
concern of the layer that authenticates the caller.

## 3. Request Guardrails

`global.services.ratelimit` configures limiter providers and rules:

```yaml
global:
  services:
    ratelimit:
      fail_open: false
      providers:
        - type: redis
          address: redis:6379
          domain: api
          rules:
            - name: premium-per-minute
              match:
                group: premium-tier
                model: qwen3-32b
              requests_per_unit: 120
              tokens_per_unit: 300000
              unit: minute
```

The `local-limiter` provider keeps its counters in the router process, so each
replica enforces its own limits and they reset on restart. It is a guardrail
against runaway traffic on a single instance, not a tenant quota or budget
ledger. Use a shared provider such as `redis` when a limit has to hold across
replicas, and set `fail_open: false` when exceeding a limit must reject rather
than pass the request through.

## Retired: dashboard security policy page

The dashboard used to ship a Security Policy page at `/security`, backed by
`GET/PUT /api/security/policy` and `POST /api/security/policy/preview`. It
bundled role bindings, generated `rbac-*` decisions, and local-limiter rules
behind one global form, kept its state in dashboard process memory, and
replaced whole config blocks on save. Those paths now return `410 Gone`, and
the `security.manage` permission has been removed.

Configuration the page previously wrote stays active. Nothing is deleted
automatically, because the generator recorded no ownership metadata and
name-based cleanup could remove operator-authored entries. To clean it up:

1. Read the active config and its version history from the Config page, or
   `GET /api/router/config/yaml` and `GET /api/router/config/versions`.
2. Identify likely generated entries: decisions named `rbac-*`, the role
   bindings they reference, and `local-limiter` rules.
3. Preview a complete diff of the config you intend to activate with
   `POST /api/router/config/deploy/preview`.
4. Activate the reviewed snapshot through the normal deploy flow.
5. If routing behavior changes unexpectedly, roll back with
   `POST /api/router/config/rollback`.

## Deployment Checklist

For production multi-user deployments:

- [ ] Verify Envoy config includes `request_headers_to_remove` for internal headers
- [ ] Use a shared rate-limit provider when limits must hold across replicas
- [ ] Set `ratelimit.fail_open: false` for strict enforcement
- [ ] Confirm the layer in front of the router authenticates callers and owns
      model access control
- [ ] Ensure the looper endpoint is only accessible from the router container
      (network-level isolation)
