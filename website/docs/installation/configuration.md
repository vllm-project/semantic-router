---
sidebar_position: 4
description: Practical guide to the v0.3 canonical YAML configuration contract across CLI, dashboard, Helm, and operator deployments.
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
  - `routing.modelCards[].loras`
  - `routing.signals`
  - `routing.projections` for partitions plus derived routing outputs
  - `routing.decisions`
- `providers` owns deployment and default-selection metadata.
  - `defaults`
  - `models`
  - `providers.defaults` holds `default_model`, `reasoning_families`, and `default_reasoning_effort`
  - `providers.models[*]` holds `provider_model_id`, `backend_refs`, `pricing`, `api_format`, and `external_model_ids`
- `global` owns router-wide runtime overrides.
  - `global.router` groups router-engine control knobs such as config-source selection, route-cache, and model-selection defaults
  - `global.router.config_source` selects whether runtime config comes from the canonical YAML file (`file`) or from in-process Kubernetes CRD reconciliation (`kubernetes`)
  - `global.router.model_selection.model_switch_gate` enables optional shadow or enforce-mode auditing for session-aware stay-vs-switch decisions after a selector chooses a candidate model
  - `global.services` groups shared APIs and control-plane services such as `response_api`, `router_replay`, `observability`, `authz`, and `ratelimit`
  - `global.services.router_replay.enabled` acts as the default replay switch for every decision; route-local `router_replay.enabled: false` is the explicit opt-out
  - `global.stores` groups shared storage-backed services such as `semantic_cache`, `memory`, and `vector_store`
- `global.integrations` groups helper runtime integrations such as `tools` and `looper`
- `global.model_catalog` groups router-owned model assets such as embeddings, system models, external models, reusable classifiers, and model-backed modules
- `global.model_catalog.embeddings.semantic.embedding_config.top_k` limits how many ranked embedding rules are emitted for routing after scoring; the built-in default is `1`
- `prototype_scoring` is the shared prototype-aware scoring block for embedding-backed signal families; use it under `global.model_catalog.embeddings.semantic.embedding_config`, `global.model_catalog.modules.classifier.preference`, `global.model_catalog.kbs[]`, and `global.model_catalog.modules.complexity` when you want exemplar banks compressed into representative prototypes
- built-in knowledge bases keep canonical source paths like `knowledge_bases/privacy/`; local runtime seeds missing KBs into `.vllm-sr/knowledge_bases/<dir>/` once and then reads the shared runtime KB store from there
- `global.model_catalog.classifiers[]` is the reusable registry for startup-loaded classifier packages such as taxonomy classifiers
- `global.model_catalog.modules` groups capability modules such as `prompt_guard`, `classifier`, `complexity`, and `hallucination_mitigation`

## Canonical example

```yaml
version: v0.3

listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
    timeout: 300s

providers:
  defaults:
    default_model: qwen3-8b
    reasoning_families:
      qwen3:
        type: chat_template_kwargs
        parameter: enable_thinking
    default_reasoning_effort: medium
  models:
    - name: qwen3-8b
      reasoning_family: qwen3
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
      loras:
        - name: math-adapter
          description: Adapter used for symbolic math and proof-style prompts.

  signals:
    keywords:
      - name: math_terms
        operator: OR
        keywords: ["algebra", "calculus"]
    structure:
      - name: many_questions
        feature:
          type: count
          source:
            type: regex
            pattern: '[?？]'
        predicate:
          gte: 3
    embeddings:
      - name: technical_support
        threshold: 0.75
        candidates: ["installation guide", "troubleshooting steps"]
      - name: account_management
        threshold: 0.72
        candidates: ["billing information", "subscription management"]

  projections:
    partitions:
      - name: support_intents
        semantics: exclusive
        temperature: 0.3
        members: [technical_support, account_management]
        default: technical_support
    scores:
      - name: request_difficulty
        method: weighted_sum
        inputs:
          - type: embedding
            name: technical_support
            weight: 0.18
            value_source: confidence
          - type: context
            name: long_context
            weight: 0.18
          - type: structure
            name: many_questions
            weight: 0.12
    mappings:
      - name: request_band
        source: request_difficulty
        method: threshold_bands
        outputs:
          - name: support_fast
            lt: 0.25
          - name: support_escalated
            gte: 0.25

  decisions:
    - name: support_route
      description: Route support requests that need an escalated answer
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: embedding
            name: technical_support
          - type: projection
            name: support_escalated
      modelRefs:
        - model: qwen3-8b
          use_reasoning: true
          lora_name: math-adapter

global:
  router:
    config_source: file
  services:
    observability:
      metrics:
        enabled: true
    router_replay:
      enabled: true
```

For `routing.signals.structure`, `feature.type: density` now uses built-in multilingual text-unit normalization. The router counts each CJK character as one unit, counts contiguous runs of other letters and digits as one unit, and ignores punctuation, so the same density rule shape behaves consistently across English, Chinese, and mixed-script prompts without a separate `normalize_by` field.

## Repository config assets

The repository now separates the exhaustive canonical reference config from reusable routing fragments:

- `config/config.yaml`: exhaustive canonical reference config
- `config/signal/`: reusable `routing.signals` fragments
- `config/decision/`: reusable `routing.decisions` rule-shape fragments
- `config/algorithm/`: reusable `decision.algorithm` snippets
- `config/plugin/`: reusable route-plugin snippets

`config/decision/` is organized by boolean case shape: `single/`, `and/`, `or/`, `not/`, and `composite/`.
`config/algorithm/` is organized by routing policy family: `looper/` and `selection/`.
`config/plugin/` is organized one plugin or reusable bundle per directory.
The repository enforces this fragment catalog in `go test ./pkg/config/...`, so routing-surface changes must update the `config/` tree in the same change.

Latest tutorials follow the same taxonomy:

- `tutorials/signal/overview` plus `tutorials/signal/heuristic/` and `tutorials/signal/learned/` for `config/signal/`
- `tutorials/decision/` for `config/decision/`
- `tutorials/algorithm/` for `config/algorithm/`, with one page per algorithm
- `tutorials/plugin/` for `config/plugin/`, with one page per plugin
- `tutorials/global/` for sparse router-wide overrides under `global:`

Repo-owned runtime and harness assets now live outside `config/`:

- `deploy/examples/runtime/semantic-cache/`
- `deploy/examples/runtime/response-api/`
- `deploy/examples/runtime/tools/`
- `e2e/config/`
- `deploy/local/envoy.yaml`

Test-only ONNX binding assets now live under `e2e/config/onnx-binding/`.

Those directories are support assets, not the main user-facing config contract. For hand-authored config, start from `config/config.yaml` or the fragment directories above. In this repository, the exhaustive reference config points `global.integrations.tools.tools_db_path` at `deploy/examples/runtime/tools/tools_db.json` for local development.

`config/config.yaml` is not just a sample anymore. The repository enforces it as the exhaustive public-contract reference:

- `go test ./pkg/config/...` checks that it stays aligned to the canonical schema and routing surface catalog
- `make agent-lint` runs the same reference-config contract check at lint level, so config/schema drift is blocked before merge
- maintained `deploy/` and `e2e/` router config assets are checked against the same canonical contract, so repo-owned examples and harness profiles cannot drift back to legacy steady-state fields

## Projection Workflow

Use `routing.projections` when the raw signal catalog is not enough on its own:

1. `routing.signals` defines reusable detectors.
2. `routing.projections.partitions` resolves one winner inside an exclusive domain or embedding family.
3. `routing.projections.scores` combines learned and heuristic signals into a weighted score.
4. `routing.projections.mappings` turns that score into named routing bands.
5. `routing.decisions[*].rules.conditions[*]` can reference those bands with `type: projection`.

The dashboard mirrors the same contract:

- `Config -> Projections` edits partitions, scores, and mappings
- `Config -> Decisions` can reference mapping outputs with condition type `projection`
- `DSL -> Visual` manages `PROJECTION partition`, `PROJECTION score`, and `PROJECTION mapping` entities directly

For a focused tutorial, read [Projections](../tutorials/projection/overview). For a maintained end-to-end example, use:

- [`deploy/recipes/balance.yaml`](https://github.com/vllm-project/semantic-router/blob/main/deploy/recipes/balance.yaml)
- [`deploy/recipes/balance.dsl`](https://github.com/vllm-project/semantic-router/blob/main/deploy/recipes/balance.dsl)

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
Inside this repository, the default exhaustive reference file is [`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml).

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
- decision model refs can carry `lora_name`, and those names resolve against `routing.modelCards[].loras`

### Helm

Helm values now mirror the same canonical contract under `config`.

```yaml
config:
  version: v0.3
  providers:
    defaults:
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

`spec.vllmEndpoints` is still the Kubernetes-native backend discovery adapter. The controller projects that data into canonical `providers.models[].backend_refs[]` and `routing.modelCards` entries, including any declared `loras`, when it renders the router config.

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

- top-level `signals`, flat `keyword_rules`/`categories`/other signal blocks, and `decisions`
- top-level `model_config`
- top-level `vllm_endpoints` and `provider_profiles`
- `providers.models[].endpoints`
- inline `access_key`

into canonical `providers/routing/global`.

### Import OpenClaw model providers

Use the CLI import command when you already have an `openclaw.json` with supported OpenAI-compatible provider endpoints and want VSR to take over model routing while rewriting OpenClaw to the first VSR listener:

```bash
vllm-sr config import --from openclaw --source openclaw.json --target config.yaml
```

When `--source` is omitted, the importer checks `OPENCLAW_CONFIG_PATH`, `./openclaw.json`, and `~/.openclaw/openclaw.json` in that order.

## Quick guides by environment

### Python CLI

1. Write `config.yaml` in canonical form.
2. Run `vllm-sr validate config.yaml`.
3. Run `vllm-sr serve --config config.yaml`.

### Router local

1. Keep provider-wide defaults in `providers.defaults` and deployment bindings in `providers.models[].backend_refs[]`.
2. Keep routing semantics in `routing.modelCards/signals/decisions`.
3. Put only runtime overrides you actually need under `global.router/services/stores/integrations/model_catalog`, and keep model-backed module settings under `global.model_catalog.modules`.
4. Use `global.router.config_source: kubernetes` only when the in-process `IntelligentPool` / `IntelligentRoute` controller is the active source of truth. Leave it as `file` for normal local, CLI, dashboard, Helm, and operator-authored canonical YAML.

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
