# Standalone Envoy JWT RBAC Integration

JWT-based user authentication and RBAC model routing — **no Kubernetes required**.

## Architecture

```
Client (Authorization: Bearer <JWT>)
  → Envoy :8802
    1. jwt_authn  — validate RSA256 signature against local JWKS
    2. claim_to_headers — extract sub → x-jwt-sub, groups → x-jwt-groups
    3. ext_proc   — semantic router evaluates RBAC role bindings, selects model
    4. ORIGINAL_DST — route to vLLM 14B or 7B based on decision
```

Single Envoy process. No Kubernetes, no Envoy Gateway, no Kind cluster.

## Quick Start

```bash
# 1. Start vLLM backends (14B on :8000, 7B on :8001)
# 2. Start semantic router on :50053 with EG identity config
#    (config/authz/envoy-jwt/config.yaml)

# 3. Setup: generate keys, tokens, start Envoy
bash scripts/authz/envoy-jwt/setup.sh

# 4. Test
bash scripts/authz/envoy-jwt/test.sh
```

## Files

| File | Purpose |
|------|---------|
| `setup.sh` | Generate keys/tokens, start Envoy container |
| `test.sh` | Run all 8 integration tests with real JWTs |
| `demo-asciinema.sh` | Record terminal demo |
| `generate-jwt-keys.py` | Generate RSA-2048 key pair + JWKS |
| `generate-jwt-tokens.py` | Mint signed JWTs for test users |
| `jwt-artifacts/` | Generated keys, JWKS, tokens (gitignored) |

## Config Files (in `config/authz/envoy-jwt/`)

| File | Purpose |
|------|---------|
| `envoy.yaml` | Envoy config: jwt_authn + ext_proc + ORIGINAL_DST |
| `config.yaml` | Semantic router config: custom identity headers, role bindings, decisions |
| — | No Kubernetes dependencies |

## Test Users

| User | JWT Groups | Role | Model |
|------|-----------|------|-------|
| alice | platform-admins | admin | 14B + reasoning |
| bob | premium-tier | premium_user | 14B (complex) or 7B (simple) |
| carol | free-tier | free_user | 7B |
| dave | premium-tier,platform-admins | admin (highest priority) | 14B |
| unknown | (none) | (no match) | 7B (default) |
| expired | premium-tier [EXPIRED] | (rejected) | 401 |

## How JWT Authentication Works

1. Client sends `Authorization: Bearer <JWT>` header
2. Envoy's `jwt_authn` filter:
   - Decodes the JWT header to find the `kid` (key ID)
   - Looks up the matching key in the local JWKS file
   - Validates the RSA256 signature
   - Checks `iss` (issuer) and `aud` (audience) claims
   - Checks `exp` (expiration) — rejects expired tokens
   - Extracts `sub` → `x-jwt-sub` header, `groups` → `x-jwt-groups` header
3. Semantic router reads `x-jwt-sub` and `x-jwt-groups` from request headers
4. RBAC classifier matches against `role_bindings` to emit role signals
5. Decision engine selects model based on role + other signals (keywords, context)
