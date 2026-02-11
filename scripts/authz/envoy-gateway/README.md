# Envoy Gateway Auth Backend

Simulates [Envoy Gateway](https://gateway.envoyproxy.io/) SecurityPolicy with JWT
authentication and `claim_to_headers` to inject user identity from JWT claims.

## Identity Flow

```
JWT (sub, groups claims)
    → Envoy Gateway validates JWT via JWKS
    → claim_to_headers injects:
        x-jwt-sub:    <JWT sub claim>
        x-jwt-groups: <JWT groups claim>
    → Router reads custom: authz.identity.user_id_header = "x-jwt-sub"
                           authz.identity.user_groups_header = "x-jwt-groups"
```

## Files

| File | Purpose |
|---|---|
| `config.yaml` | Router config — custom identity headers, RBAC bindings, vLLM endpoints |
| `envoy.yaml` | Envoy config — NO ext_authz (simulates EG post-JWT-validation) |
| `profile.yaml` | Authz provider profile — EG JWT credential chain |
| `test.sh` | Live integration test — 7 tests with simulated JWT headers |

## Prerequisites

- Envoy on port 8802 (using `envoy.yaml`)
- Router on port 50053 with `config.yaml`
- vLLM 14B on port 8000, vLLM 7B on port 8001

## Simulation Note

Since this test doesn't run an actual Envoy Gateway, the client sends
`x-jwt-sub` and `x-jwt-groups` headers directly — this is exactly what
EG's `claim_to_headers` would inject after JWT validation.

## Run

```bash
bash scripts/authz/envoy-gateway/test.sh
```
