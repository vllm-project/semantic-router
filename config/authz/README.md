# Authz Provider Configurations

Each subdirectory contains the complete set of configs for a specific auth backend.

## Providers

| Directory | Auth Backend | Identity Headers |
|---|---|---|
| `authorino/` | Authorino (ext_authz, K8s Secrets) | `x-authz-user-id`, `x-authz-user-groups` (defaults) |
| `envoy-gateway/` | Envoy Gateway (JWT, claim_to_headers) | `x-jwt-sub`, `x-jwt-groups` (custom via `authz.identity`) |

## File Layout

```
config/authz/
├── README.md
├── authorino/
│   ├── config.yaml        # router config — default identity headers, RBAC bindings, decisions
│   ├── envoy.yaml         # Envoy — ext_authz (Authorino) + ext_proc (Router)
│   ├── profile.yaml       # authz credential chain profile
│   └── k8s/               # Kubernetes manifests
│       ├── authconfig.yaml
│       ├── k8s-deploy.yaml
│       ├── secrets-byot.yaml
│       ├── secrets-per-user.yaml
│       └── secrets-shared.yaml
└── envoy-gateway/
    ├── config.yaml        # router config — custom identity headers (x-jwt-sub/x-jwt-groups)
    ├── envoy.yaml         # Envoy — NO ext_authz (EG handles JWT externally)
    └── profile.yaml       # authz credential chain profile (EG JWT)
```

## Usage

Start the router with the provider-specific config:

```bash
# Authorino
go run cmd/main.go --config ../../config/authz/authorino/config.yaml --port 50051

# Envoy Gateway
go run cmd/main.go --config ../../config/authz/envoy-gateway/config.yaml --port 50053
```

## Adding a New Provider

1. Create `config/authz/<provider-name>/`
2. Add `config.yaml` with the appropriate `authz.identity` headers
3. Add `envoy.yaml` for the Envoy/gateway configuration
4. Add `profile.yaml` for the credential chain
5. Add a test script in `scripts/authz/<provider-name>/test.sh`
