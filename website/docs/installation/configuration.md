---
sidebar_position: 4
---

# Configuration

Semantic Router v0.3 uses one canonical YAML contract across local CLI, dashboard, Helm, and the operator:

```yaml
version:
listeners:
providers:
routing:
global:
```

The detailed background is in [Unified Config Contract v0.3](../proposals/unified-config-contract-v0-3). This page is the practical guide for using the contract.

## Canonical contract

- `version`: schema version. Use `v0.3`.
- `listeners`: router listener ports and timeouts.
- `providers`: deployment bindings and provider defaults.
- `routing`: routing semantics.
- `global`: sparse runtime overrides. If you omit a field here, the router's built-in default is used.

## Ownership by section

- `routing` is the DSL-owned surface.
  - `routing.modelCards`
  - `routing.signals`
  - `routing.decisions`
- `providers` owns deployment and default-selection metadata.
  - `default_model`
  - `reasoning_families`
  - `default_reasoning_effort`
  - `models`
- `global` owns router-wide runtime overrides.
  - Examples: `observability`, `semantic_cache`, `prompt_guard`, `tools`, `model_selection`

## Canonical example

```yaml
version: v0.3

listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
    timeout: 300s

providers:
  default_model: qwen3-8b
  reasoning_families:
    qwen3:
      type: effort
      parameter: reasoning.effort
  default_reasoning_effort: medium
  models:
    - name: qwen3-8b
      provider_model_id: qwen3-8b
      backend_refs:
        - name: primary
          endpoint: host.docker.internal:8000
          protocol: http
          weight: 100
          api_key_env: OPENAI_API_KEY

routing:
  modelCards:
    - name: qwen3-8b
      modality: text
      capabilities: [chat, reasoning]
      reasoning_family_ref: qwen3

  signals:
    keywords:
      - name: math_terms
        operator: OR
        keywords: ["algebra", "calculus"]

  decisions:
    - name: math_route
      description: Route math requests
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: keyword
            name: math_terms
      modelRefs:
        - model: qwen3-8b
          use_reasoning: true

global:
  observability:
    metrics:
      enabled: true
```

## Repository config assets

The repository now separates the runnable starter config from reusable routing fragments:

- `config/config.yaml`: canonical runnable starter config
- `config/signal/`: reusable `routing.signals` fragments
- `config/decision/`: reusable `routing.decisions` rule-shape fragments
- `config/algorithm/`: reusable `decision.algorithm` snippets
- `config/plugin/`: reusable route-plugin snippets

`config/decision/` is organized by boolean case shape: `single/`, `and/`, `or/`, `not/`, and `composite/`.
`config/algorithm/` is organized by routing policy family: `looper/` and `selection/`.
The repository enforces this fragment catalog in `go test ./pkg/config/...`, so routing-surface changes must update the `config/` tree in the same change.

Latest tutorials follow the same taxonomy:

- `tutorials/signal/` for `config/signal/`
- `tutorials/decision/` for `config/decision/`
- `tutorials/algorithm/` for `config/algorithm/`
- `tutorials/plugin/` for `config/plugin/`
- `tutorials/global/` for sparse router-wide overrides under `global:`

Repo-owned runtime and harness assets now live outside `config/`:

- `examples/runtime/semantic-cache/`
- `examples/runtime/response-api/`
- `examples/runtime/tools/`
- `e2e/config/`
- `deploy/local/envoy.yaml`

Test-only ONNX binding assets now live under `e2e/config/onnx-binding/`.

Those directories are support assets, not the main user-facing config contract. For hand-authored config, start from `config/config.yaml` or the fragment directories above. In this repository, the starter config points `global.tools.tools_db_path` at `examples/runtime/tools/tools_db.json` for local development.

## How to use it

### Python CLI

Use the canonical YAML directly.

```bash
vllm-sr serve --config config.yaml
```

To migrate an older config first:

```bash
vllm-sr config migrate --config old-config.yaml
vllm-sr validate config.yaml
```

`vllm-sr init` was removed in v0.3. The steady-state file is `config.yaml`.
Inside this repository, the default starter file is [`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml).

### Router local / YAML-first

For local Docker or direct router development, hand-author `config.yaml` in canonical form and validate it before serving:

```bash
vllm-sr validate config.yaml
vllm-sr serve --config config.yaml
```

If you only need to override a few runtime defaults, write those under `global:` and leave the rest unset.

### Dashboard / onboarding

Use the dashboard when you want to import or edit the full canonical YAML directly.

- onboarding remote import accepts a complete `version/listeners/providers/routing/global` file
- the config page edits the same canonical contract
- the DSL editor can import the same YAML, but it only decompiles `routing`

### Helm

Helm values now mirror the same canonical contract under `config`.

```yaml
config:
  version: v0.3
  providers:
    default_model: qwen3-8b
    models:
      - name: qwen3-8b
        provider_model_id: qwen3-8b
        backend_refs:
          - name: primary
            endpoint: semantic-router-vllm.default.svc.cluster.local:8000
            protocol: http
  routing:
    modelCards:
      - name: qwen3-8b
```

Then install or upgrade normally:

```bash
helm upgrade --install semantic-router deploy/helm/semantic-router -f values.yaml
```

### Operator

The operator keeps the same logical contract, but it wraps it inside the CRD:

- `spec.config.providers`
- `spec.config.routing`
- `spec.config.global`

`spec.vllmEndpoints` is still the Kubernetes-native backend discovery adapter. The controller projects that data into canonical `providers.models[].backend_refs[]` and `routing.modelCards` entries when it renders the router config.

See [Kubernetes Operator](./k8s/operator).

### DSL

DSL only owns the `routing` surface.

- Author `MODEL`, `SIGNAL`, and `ROUTE`
- Compile to a routing fragment
- Keep `providers` and `global` in YAML

The DSL compiler emits:

```yaml
routing:
  modelCards:
  signals:
  decisions:
```

It does not emit `listeners`, `providers`, or `global`.

## Import and migration

### Onboarding remote import

The setup wizard can import a full canonical YAML file from a URL and apply the complete config, including `providers`, `routing`, and `global`.

### DSL import

The DSL editor can import:

- a full router config YAML
- a routing-only YAML fragment

In both cases, only the `routing` section is decompiled into DSL.

### Migrate old configs

Use the CLI migration command for older flat or mixed configs:

```bash
vllm-sr config migrate --config old-config.yaml
```

This migrates legacy shapes such as:

- top-level `signals` and `decisions`
- `providers.models[].endpoints`
- inline `access_key`

into canonical `providers/routing/global`.

## Quick guides by environment

### Python CLI

1. Write `config.yaml` in canonical form.
2. Run `vllm-sr validate config.yaml`.
3. Run `vllm-sr serve --config config.yaml`.

### Router local

1. Keep deployment bindings in `providers.models[].backend_refs[]`.
2. Keep routing semantics in `routing.modelCards/signals/decisions`.
3. Put only runtime overrides you actually need under `global:`.

### Helm

1. Put the same canonical config under `values.yaml -> config`.
2. Use `helm upgrade --install ... -f values.yaml`.
3. Treat Helm as a deployment wrapper, not a second config schema.

### Operator

1. Put portable config under `spec.config`.
2. Use `spec.vllmEndpoints` only when you want Kubernetes-native backend discovery.
3. Expect the operator to render canonical router config from that adapter layer.

### DSL

1. Use DSL for `routing.modelCards`, `routing.signals`, and `routing.decisions`.
2. Importing a full YAML file still works, but only `routing` is decompiled into DSL.
3. Keep endpoints, API keys, listeners, and `global` in YAML.
4. Reusable routing fragments now live under `config/signal/`, `config/decision/`, `config/algorithm/`, and `config/plugin/`.
