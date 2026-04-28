# Unified Config Contract v0.3

Issue: [#1505](https://github.com/vllm-project/semantic-router/issues/1505)

---

## Before

Before v0.3, the repo had several partially overlapping config contracts:

- router runtime consumed a flat Go config
- Python CLI used its own nested YAML plus merge/default logic
- dashboard and onboarding imported YAML but still assumed legacy top-level `signals` and `decisions`
- Helm and operator each translated config differently
- DSL mixed routing semantics with legacy `BACKEND` and `GLOBAL` expectations

This caused three persistent problems:

1. The same concept had to be edited in multiple schema layers.
2. Endpoint, API key, and model semantics were mixed together.
3. Runtime defaults depended on external template files such as `router-defaults.yaml`, which made defaults harder to reason about and replace.

## Problems with the old model

### CLI and router drifted

The Python CLI and Go router did not share one schema owner. A user could build config through the CLI, the dashboard, or Kubernetes and still hit structural mismatches.

### Model semantics and deployment bindings were entangled

Logical models were carrying:

- semantic routing identity
- endpoint binding
- API key
- provider model ID

That made reuse hard. If several logical models pointed at the same backend, config still repeated backend details.

### DSL scope was too broad

DSL was useful for routing semantics, but legacy `BACKEND` and `GLOBAL` blocks made it look like the right place to author deployment and runtime state too. That was not sustainable across local, dashboard, and Kubernetes workflows.

## v0.3 contract

v0.3 defines one canonical config:

```yaml
version:
listeners:
providers:
routing:
global:
```

### What each section means

- `providers`: deployment bindings and provider defaults
- `routing`: semantic routing graph
- `global`: sparse router-wide runtime overrides

### DSL boundary

DSL now owns only:

- `routing.modelCards`
- `routing.signals`
- `routing.projections` for signal coordination and derived routing outputs
- `routing.decisions`, including bounded candidate-iteration metadata used by DSL `FOR ... IN` authoring

It no longer owns endpoints, API keys, listeners, or router-global runtime settings.

### Deployment binding split

Model semantics and deployment bindings are now separated explicitly:

- `routing.modelCards` carries semantic catalog data such as size, context window, description, and capabilities
- `routing.modelCards[].loras` carries the canonical LoRA adapter catalog for each logical model
- `providers.defaults` carries provider-wide defaults such as `default_model`, `reasoning_families`, and `default_reasoning_effort`
- `providers.models` carries per-model access bindings directly
- each `providers.models[].backend_refs[]` item carries its own transport and auth fields such as `endpoint`, `base_url`, `protocol`, `auth_header`, `auth_prefix`, `api_key`, and `api_key_env`
- `routing.decisions[].modelRefs[].lora_name` resolves against the matching `routing.modelCards[].loras` entry, so `lora_name` is now part of the supported routing contract instead of a runtime-only escape hatch
- `routing.decisions[].candidateIterations` is bounded to `decision.candidates` or explicit model lists and remains declarative metadata for the selection layer, not a second policy interpreter

## Global defaults

Router-global defaults are now owned by the router itself, not by a second user-maintained defaults file.

- the router provides typed built-in defaults
- `global:` only overrides what you need to change
- `global.router` groups router-engine control knobs, including `config_source`
- `global.services` groups shared APIs and runtime services
- `global.services.router_replay.enabled` provides the router-wide replay default, while route-local `router_replay.enabled: false` is the explicit opt-out
- `global.stores` groups storage-backed services
- `global.integrations` groups helper runtime integrations
- `global.model_catalog` groups router-owned model assets under `embeddings`, `system`, `external`, `classifiers`, `kbs`, and `modules`, including embedding fallback knobs such as `embedding_config.top_k`, shared prototype-aware scoring controls such as `prototype_scoring`, and built-in knowledge-base source paths such as `knowledge_bases/privacy/`
- `global.model_catalog.modules` is the home for router-owned module settings such as `prompt_compression`, `prompt_guard`, `classifier`, `complexity`, `hallucination_mitigation`, `feedback_detector`, and `modality_detector`
- omitted fields keep the built-in default

This makes local, dashboard, Helm, and operator behavior converge on the same baseline.

## How the surfaces unify

### Onboarding import

Remote onboarding import can fetch and apply a full canonical YAML file. That keeps the original intent of "one remote YAML can set up the whole router".

### DSL import

DSL import still accepts a full router config YAML, but it decompiles only the `routing` section into DSL. Static deployment and global runtime settings stay in YAML.

The router parser itself now accepts only canonical v0.3 YAML for steady-state runtime config. Legacy mixed layouts must go through explicit migration first.

The remaining in-process CRD reconciliation path now also re-enters the same canonical parser through `global.router.config_source: kubernetes`, instead of maintaining a separate steady-state runtime layout.

### Repository config assets

The repo no longer ships large full-example trees under `config/intelligent-routing/` and similar directories. Instead:

- `config/config.yaml` is the exhaustive canonical reference config
- `config/signal/`, `config/decision/`, `config/algorithm/`, and `config/plugin/` hold reusable routing fragments
- `config/decision/` is organized by boolean rule shape (`single`, `and`, `or`, `not`, `composite`)
- `config/algorithm/` is organized by routing policy family (`looper`, `selection`)
- latest `docs/tutorials/` source tree mirrors `signal/decision/algorithm/plugin/global`, and the older tutorial trees were removed from the active docs surface
- runtime support examples such as `deploy/examples/runtime/semantic-cache/`, `deploy/examples/runtime/response-api/`, and `deploy/examples/runtime/tools/` stay separate because they are not part of the user-facing config contract
- harness-only manifests live under `e2e/config/`
- `go test ./pkg/config/...` and `make agent-lint` enforce that `config/config.yaml` stays exhaustive and aligned with the public config contract

### Helm and operator

Helm values and operator config now align to the same canonical concepts instead of inventing separate steady-state router schemas.

## Migration path

Older configs can be migrated with:

```bash
vllm-sr config migrate --config old-config.yaml
```

That command rewrites legacy config into canonical `providers/routing/global`.

It covers both mixed nested layouts and older flat runtime layouts such as top-level `keyword_rules`, `model_config`, `vllm_endpoints`, and `provider_profiles`.

`vllm-sr init` was removed as part of the cleanup. Canonical `config.yaml` is now the only steady-state file users are expected to author.

## Result

The repo now has one public config story:

- full router configuration lives in canonical YAML
- DSL is the routing-semantic view of that config
- deployment bindings live in `providers.defaults` and `providers.models[]`
- runtime overrides live in `global.router`, `global.services`, `global.stores`, `global.integrations`, and `global.model_catalog`, with model-backed modules under `global.model_catalog.modules`
- structure-signal `density` uses one built-in multilingual normalization path instead of exposing per-rule `normalize_by` choices
- `global.router.config_source` is the canonical switch between file-backed config and Kubernetes CRD-backed reconciliation
- built-in defaults live in the router
- repo-owned sample assets are organized by `signal/decision/algorithm/plugin` fragments instead of parallel full-config examples

That removes the old CLI/router/dashboard/Helm/operator drift and gives every environment one shared contract.
