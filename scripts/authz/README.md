# Authz Provider Configs and Tests

Each subdirectory contains the complete config and test for a specific auth backend.

## Providers

| Directory | Auth Backend | Identity Headers | Test |
|---|---|---|---|
| `authorino/` | Authorino (ext_authz, K8s Secrets) | `x-authz-user-id`, `x-authz-user-groups` (defaults) | `authorino/test.sh` |
| `envoy-gateway/` | Envoy Gateway (JWT, claim_to_headers) | `x-jwt-sub`, `x-jwt-groups` (custom) | `envoy-gateway/test.sh` |

## Running

```bash
# Authorino (requires Kind cluster + Authorino + Envoy on 8801 + Router on 50051)
bash scripts/authz/authorino/test.sh

# Envoy Gateway simulation (requires Envoy on 8802 + Router on 50053)
bash scripts/authz/envoy-gateway/test.sh
```

## File Layout

```
scripts/authz/
├── README.md                          # this file
├── authorino/
│   ├── config.yaml                    # router config (default identity headers)
│   ├── envoy.yaml                     # Envoy config (ext_authz + ext_proc)
│   ├── profile.yaml                   # authz provider profile (credential chain)
│   ├── test.sh                        # live integration test (8 tests)
│   └── k8s/                           # Kubernetes manifests
│       ├── authconfig.yaml            # Authorino AuthConfig CRD
│       ├── k8s-deploy.yaml            # Authorino standalone deployment
│       ├── secrets-byot.yaml          # BYOT user secret
│       ├── secrets-per-user.yaml      # per-user secret
│       └── secrets-shared.yaml        # shared secrets
└── envoy-gateway/
    ├── config.yaml                    # router config (custom identity headers)
    ├── envoy.yaml                     # Envoy config (NO ext_authz, ext_proc only)
    ├── profile.yaml                   # authz provider profile (EG JWT)
    └── test.sh                        # live integration test (7 tests)
```
