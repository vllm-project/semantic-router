# Authz Signal

## Overview

`authz` turns identity and policy bindings into reusable routing inputs. It maps to `config/signal/authz/` and is declared under `routing.signals.role_bindings`.

This family is heuristic: it matches request identity against explicit roles and subjects instead of classifier output.

## Key Advantages

- Routes premium, internal, or tenant-scoped traffic without extra model inference.
- Keeps access policy visible inside `routing.decisions`.
- Reuses the same identity rule across multiple routes.
- Makes RBAC-driven routing auditable in YAML.

## What Problem Does It Solve?

Without an `authz` signal, routing decisions cannot see user tier or role membership directly. That pushes access-sensitive routing into scattered middleware and makes policy harder to review.

`authz` solves that by exposing role membership as a named signal that decisions can compose with domain, safety, or plugin logic.

## When to Use

Use `authz` when:

- admin traffic must route differently from end-user traffic
- premium tiers unlock stronger models or plugins
- tenant or group membership changes route eligibility
- route policy should stay in the same graph as the rest of routing logic

## Configuration

Source fragment family: `config/signal/authz/`

```yaml
routing:
  signals:
    role_bindings:
      - name: admin
        description: Requests from platform administrators.
        role: admin
        subjects:
          - kind: Group
            name: platform-admins
      - name: premium_user
        description: Requests from paid end users.
        role: premium_user
        subjects:
          - kind: Group
            name: premium-tier
```

Use `role_bindings` when the signal should fire from authenticated identity and policy metadata instead of prompt content.
