# Authz Signal

## Overview

`authz` turns trusted routing claims into reusable routing inputs under
`routing.signals.role_bindings`.

This family is heuristic: it matches request identity against explicit roles and subjects instead of classifier output.

## Key Advantages

- Routes premium, internal, or tenant-scoped traffic without extra model inference.
- Keeps access policy visible inside `routing.decisions`.
- Reuses the same identity rule across multiple routes.
- Makes identity-aware routing auditable in the Recipe.

## What Problem Does It Solve?

Without an `authz` signal, routing decisions cannot use a verified user, Team,
or routing claim as an input. This signal keeps those routing choices explicit
without turning them into access permissions.

`authz` solves that by exposing role membership as a named signal that decisions can compose with domain, safety, or plugin logic.

## When to Use

Use `authz` when:

- admin traffic must route differently from end-user traffic
- premium tiers unlock stronger models or plugins
- tenant or group membership changes route eligibility
- route policy should stay in the same graph as the rest of routing logic

## Configuration

```yaml
routing:
  signals:
    role_bindings:
      - name: admin
        description: Requests from one Team.
        role: admin
        subjects:
          - kind: Team
            name: team_platform
      - name: premium_user
        description: Requests from paid end users.
        role: premium_user
        subjects:
          - kind: Group
            name: premium-tier
```

Use `role_bindings` when the signal should fire from authenticated identity and policy metadata instead of prompt content.

## Dependencies and Limitations

Identity comes only from the Router-authenticated `TenantContext`; the signal
does not authenticate a request or grant model access. `User` and `Team`
subjects match their corresponding IDs. `Group` subjects match compiled string
claim values or the names of true boolean claims. Without Router-native access
and an authenticated `TenantContext`, a Recipe that requires these bindings
fails closed.
See a complete example:
[`config/fragments/signal/authz/rbac.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/authz/rbac.yaml).
