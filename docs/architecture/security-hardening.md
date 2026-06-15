# Security Hardening Guide

This document describes the RBAC-to-router integration and production-default
Envoy configuration shipped with the semantic router to close trust-boundary
gaps between the dashboard and the routing plane.

## Overview

The semantic router sits between external clients and upstream LLM backends.
It processes requests through Envoy's ext_proc filter and makes routing
decisions based on signal classification. This guide covers:

1. **Production-default Envoy config** — defense-in-depth header stripping at
   the proxy layer.
2. **Dashboard RBAC integration** — translating dashboard role/permission
   changes into router `role_bindings` and `ratelimit` rules.
3. **New dashboard permissions** — fine-grained RBAC permissions for
   feedback, replay, and security policy management.

## Architecture

```
 Client → Envoy (header stripping) → ext_proc → upstream LLM
                                       │
                  Dashboard ──────────►│ (config fragment sync)
                    │                  │
                    ├── Security Policy API
                    ├── Role-to-Model mappings
                    └── Rate-limit tier mappings
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

## 2. Dashboard RBAC Integration

### Security Policy API

The dashboard provides a Security Policy management page (`/security`),
accessible from the **Manager** dropdown in the top navigation bar. It
allows administrators to:

1. **Map RBAC roles to models** — Define which user groups or individual users
   can access which LLM models. Each mapping generates a router authz signal
   rule and a corresponding decision entry.
2. **Set rate-limit tiers** — Configure per-role/group rate limits that are
   translated into router `ratelimit.providers[].rules`.

### Auto-Apply on Save

When a security policy is saved via `PUT /api/security/policy`, the dashboard
automatically applies the generated fragment to the router's `config.yaml` and
triggers a hot-reload. The full pipeline is:

```
Dashboard UI → PUT /api/security/policy
  → GenerateRouterFragment()
  → toCanonicalYAML()          (maps to routing.signals.role_bindings,
                                 routing.decisions, and
                                 global.services.ratelimit)
  → mergeDeployPayload()       (deep-merges into existing config.yaml)
  → writeConfigAtomically()
  → applyWrittenConfig()       (Envoy restart + fsnotify router hot-reload)
```

The merge uses **replace semantics** for both `routing.decisions` and
`global.services.ratelimit`: the entire block is replaced by the security
policy fragment. Other global config fields (observability, authz, etc.) are
preserved. `routing.signals.role_bindings` are merged alongside other signals.

The API response includes an `"applied": true` field when the fragment was
successfully written and propagated to the runtime.

A `SecurityPolicyConfig` contains:

- `role_mappings[]` — each entry maps subjects (Users/Groups) to a router role
  and a set of allowed models
- `rate_tiers[]` — each entry maps a user/group to RPM and TPM limits

The `GenerateRouterFragment()` function converts this into:

- `role_bindings[]` — router-config-compatible subject-to-role bindings
- `decisions[]` — decision entries with authz rules referencing the mapped role
- `ratelimit.providers[].rules[]` — local-limiter rules with per-unit limits

### Example

Given this security policy:

```json
{
  "role_mappings": [
    {
      "name": "premium",
      "subjects": [{"kind": "Group", "name": "paying-customers"}],
      "role": "premium_tier",
      "model_refs": ["gpt-4", "claude-3"],
      "priority": 10
    }
  ],
  "rate_tiers": [
    {
      "name": "premium-rate",
      "group": "paying-customers",
      "rpm": 1000,
      "tpm": 100000
    }
  ]
}
```

The generated router config fragment is:

```json
{
  "role_bindings": [
    {
      "subjects": [{"kind": "Group", "name": "paying-customers"}],
      "role": "premium_tier"
    }
  ],
  "decisions": [
    {
      "name": "rbac-premium",
      "priority": 10,
      "rules": {"type": "authz", "name": "premium_tier"},
      "modelRefs": [{"model": "gpt-4"}, {"model": "claude-3"}]
    }
  ],
  "ratelimit": {
    "providers": [
      {
        "type": "local-limiter",
        "rules": [
          {
            "name": "premium-rate",
            "match": {"group": "paying-customers"},
            "requests_per_unit": 1000,
            "tokens_per_unit": 100000,
            "unit": "minute"
          }
        ]
      }
    ]
  }
}
```

### API Endpoints

| Endpoint | Method | Permission | Description |
|---|---|---|---|
| `/api/security/policy` | `GET` | `config.read` | Get current security policy |
| `/api/security/policy` | `PUT` | `security.manage` | Update policy, generate fragment, and auto-apply to router |
| `/api/security/policy/preview` | `POST` | `security.manage` | Preview fragment without saving |

### Validation Rules

The security policy is validated before processing:

- Role mapping `name` must be non-empty and unique
- Role mapping `role` must be non-empty
- At least one subject (User or Group) per mapping
- Subject `kind` must be `"User"` or `"Group"`
- Subject `name` must be non-empty
- Rate tier `name` must be non-empty
- At least one of `rpm` or `tpm` must be set per tier

## 3. Dashboard RBAC Permissions

Three new permissions were added to the dashboard RBAC system:

| Permission | Description | Admin | Write | Read |
|---|---|---|---|---|
| `feedback.submit` | Submit model selection feedback | Yes | Yes | Yes |
| `replay.read` | Access router replay API records | Yes | Yes | Yes |
| `security.manage` | Manage security policy (write) | Yes | No | No |

The `security.manage` permission is required for `PUT` and `POST` requests
to `/api/security/*` endpoints. `GET` requests only require `config.read`.

## Deployment Checklist

For production multi-user deployments:

- [ ] Verify Envoy config includes `request_headers_to_remove` for internal headers
- [ ] Configure rate-limit tiers via the Security Policy page or API
- [ ] Set `ratelimit.fail_open: false` for strict enforcement
- [ ] Configure role-to-model mappings to restrict model access by group
- [ ] Review dashboard user roles — only admins should have `security.manage`
- [ ] Ensure the looper endpoint is only accessible from the router container
      (network-level isolation)
