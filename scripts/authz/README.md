# Authz Provider Tests

Each subdirectory contains the test scripts for a specific auth backend.

## Providers

| Directory | Auth Backend | Identity Headers | Test |
|---|---|---|---|
| `authorino/` | Authorino (ext_authz, K8s Secrets) | `x-authz-user-id`, `x-authz-user-groups` (defaults) | `authorino/test.sh` |
| `envoy-jwt/` | Envoy jwt_authn (RSA256, local JWKS) | `x-jwt-sub`, `x-jwt-groups` (custom) | `envoy-jwt/test.sh` |

## Running

```bash
# Authorino (requires Kind cluster + Authorino + Envoy on 8801 + Router on 50051)
bash scripts/authz/authorino/test.sh

# JWT (standalone Envoy with jwt_authn — no Kubernetes required)
# First run setup to generate keys, tokens, and start Envoy:
bash scripts/authz/envoy-jwt/setup.sh
# Then run tests:
bash scripts/authz/envoy-jwt/test.sh
```

## File Layout

```
scripts/authz/
├── README.md
├── authorino/
│   ├── config.yaml → ../../config/authz/authorino/config.yaml
│   ├── envoy.yaml  → ../../config/authz/authorino/envoy.yaml
│   ├── profile.yaml
│   ├── test.sh                        # live integration test (8 tests)
│   └── k8s/ → ../../config/authz/authorino/k8s/
└── envoy-jwt/
    ├── config.yaml → ../../config/authz/envoy-jwt/config.yaml
    ├── envoy.yaml  → ../../config/authz/envoy-jwt/envoy.yaml
    ├── setup.sh                       # generate keys/tokens, start Envoy container
    ├── test.sh                        # live integration test (8 tests, real JWTs)
    ├── demo-asciinema.sh              # terminal demo recording script
    ├── generate-jwt-keys.py           # RSA key pair + JWKS generation
    ├── generate-jwt-tokens.py         # JWT minting per test user
    └── jwt-artifacts/                 # generated keys, tokens (gitignored)
```
