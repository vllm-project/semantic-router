# Authorino Auth Backend

Uses [Authorino](https://github.com/Kuadrant/authorino) as an Envoy ext_authz service
to authenticate users via K8s Secrets and inject identity + credential headers.

## Identity Flow

```
K8s Secret (api_key, metadata.name, authz-groups annotation)
    → Authorino validates Bearer token
    → Injects headers:
        x-authz-user-id:     <Secret metadata.name>
        x-authz-user-groups: <Secret annotation authz-groups>
        x-user-openai-key:   <Secret annotation openai-key>
    → Router reads defaults: authz.identity.user_id_header = "x-authz-user-id"
```

## Files

| File | Purpose |
|---|---|
| `config.yaml` | Router config — RBAC role bindings, decisions, vLLM endpoints |
| `envoy.yaml` | Envoy config — ext_authz (Authorino) + ext_proc (Router) |
| `profile.yaml` | Authz provider profile — credential chain config |
| `test.sh` | Live integration test — 8 tests with real K8s tokens |
| `k8s/` | Kubernetes manifests for Authorino + user secrets |

## Prerequisites

- Kind cluster with Authorino pod running
- kubectl port-forward to Authorino
- Envoy on port 8801
- Router on port 50051 with `config.yaml`
- vLLM 14B on port 8000, vLLM 7B on port 8001

## Run

```bash
bash scripts/authz/authorino/test.sh
```
