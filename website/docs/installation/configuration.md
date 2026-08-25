---
title: Configuration
description: Understand the canonical v0.3 YAML document and where Models, Recipes, Entrypoints, services, and secrets belong.
---

# Configuration

Semantic Router uses one canonical YAML document across the CLI, Dashboard,
Helm, and Operator. The top-level structure is:

```yaml
version:
listeners:
providers:
routing:
recipes:
entrypoints:
global:
```

Every runnable manifest keeps physical Model connections under `providers`,
connection-free Model metadata under `routing.modelCards`, reusable routing logic
under `recipes`, and public virtual Model names under `entrypoints`. Add `global`
only for shared services, stores, integrations, or runtime behavior that differs
from the built-in defaults.

## What belongs where

| Section | Owns |
| --- | --- |
| `version` | Canonical schema version. Use `v0.3`. |
| `listeners` | Public Router listeners and timeouts. |
| `providers.models` | Physical Model connections, invocation control, reasoning family, and pricing. |
| `routing.modelCards` | Connection-free Model descriptions and capabilities used while designing Recipes. |
| `recipes[].routing` | Model-free routing documents containing signals, projections, decisions, strategy, algorithms, and route plugins. |
| `entrypoints` | Public virtual Model names, Recipe references, and complete Decision-to-Model assignments. |
| `global` | Shared billing, Router services, stores, integrations, observability, learning, and router-owned model assets. |

Keep these boundaries clear:

- signals detect facts;
- projections combine evidence;
- decisions define eligibility and route policy;
- algorithms choose or coordinate candidate models;
- plugins add behavior at route-specific hook points; and
- Entrypoints bind each Recipe decision to one or more Models.

Model rates belong in each `providers.models[].pricing` block. Their common denomination
belongs in one place:

```yaml
global:
  billing:
    currency: USD
```

This block is optional when no Model is priced. A priced manifest must set one
uppercase ISO-4217 currency so fallback, multi-model execution, usage, and cost
quotas share an unambiguous unit. When a Management store initializes an empty
Namespace, this value becomes its immutable billing currency.

Each physical Model can keep invocation behavior together under `control`:

```yaml
providers:
  models:
    - name: remote/frontier
      control:
        retry:
          count: 2
          on: [unavailable, timeout]
        timeout:
          request: 60s
          stream: 10m
```

`retry.count` is the number of additional attempts after the initial call and
must be between 0 and 5. `retry.on` accepts Router evidence classes
`unavailable`, `overloaded`, and `timeout`; when a positive count omits the list,
the default is `[unavailable]`. A retry starts only when the Router has proved
that the failed attempt produced no client-visible output. Request and stream
timeouts bound the whole physical dispatch, including its retries.

When one Model has multiple `backend_refs`, each backend's existing `weight`
is the physical traffic-distribution input. The public `control` block does not
expose load-balancing, health-check, or outlier-ejection fields until the Router
can enforce those policies end to end.

Reasoning assignments use the same connection/metadata boundary. A physical Model
selects its wire adapter with `providers.models[].reasoning_family`; the matching
`routing.modelCards[]` entry declares supported values with
`reasoning: {type: reasoning_effort, efforts: [medium, high]}`. The types must agree,
and an Entrypoint cannot request an undeclared effort.

Pricing uses quoted decimal strings so accounting never passes through binary
floating point:

```yaml
pricing:
  input_cost_per_million_tokens: "0.10"
  output_cost_per_million_tokens: "0.40"
  cache_read_cost_per_million_tokens: "0.02"
  cache_write_cost_per_million_tokens: "0.12"
```

A file-backed Model accepts the existing `api_key` or `api_key_env` credential
source on each backend reference. Configure exactly one. Prefer `api_key_env` for
shared or committed manifests:

```yaml
providers:
  models:
    - name: remote/model
      provider_model_id: provider/model
      api_format: openai
      backend_refs:
        - provider: openai-compatible
          base_url: https://models.example.com/v1
          api_key_env: MODEL_API_KEY
routing:
  modelCards:
    - name: remote/model
      description: General-purpose remote model.
      capabilities: [chat, tools]
```

The control plane resolves the provider integration, compiles the connection,
and pins the resulting immutable routing snapshot. The Router injects the selected
credential only after it selects that backend. A literal `api_key` remains valid for
existing file-based workflows, but it makes every copy or export of that authoring
file secret-bearing. Dynamic Model resources use versioned
ProviderCredential resources published through the Management API instead.

The selected Provider Integration owns its default origin, API style, discovery,
path, safe headers, and credential adapter. Private and compatible providers accept
an explicit base URL; fixed public APIs need only a credential and provider model.

The [Routing Pipeline](../overview/signal-driven-decisions) explains the design.
Capability pages under **Capabilities** document each signal, projection,
decision, algorithm, plugin, and global block.

## Capability catalog

Use this catalog to choose a reusable building block, then open its guide for
configuration details. The inventory comes from `config/fragments/`; each
one-line goal comes from the matching guide's **Overview**. The documentation
build regenerates this block and fails if the checked-in catalog has drifted.

<!-- BEGIN GENERATED CONFIGURATION CATALOG -->
<!-- Generated by website/scripts/generate-configuration-catalog.mjs. Do not edit this block by hand. -->

### Signals

| Family and type | Use it to | Reusable fragment | Guide |
| --- | --- | --- | --- |
| `authz` — heuristic signal | `authz` turns trusted routing claims into reusable routing inputs under `routing.signals.role_bindings`. | [`config/fragments/signal/authz/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/authz/) | [Guide](../tutorials/signal/heuristic/authz) |
| `classifier` — learned signal | `classifier` exposes reusable label scores from a local native sequence classifier or a configured external LLM. | [`config/fragments/signal/classifier/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/classifier/) | [Guide](../tutorials/signal/learned/classifier) |
| `complexity` — learned signal | `complexity` estimates whether a request is `easy`, `medium`, or `hard` by comparing it with configured example sets. | [`config/fragments/signal/complexity/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/complexity/) | [Guide](../tutorials/signal/learned/complexity) |
| `context` — heuristic signal | `context` detects requests that need a larger effective context window. | [`config/fragments/signal/context/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/context/) | [Guide](../tutorials/signal/heuristic/context) |
| `conversation` — heuristic signal | `conversation` routes on the structure of a chat, such as message count, developer instructions, available tools, or an active tool loop. | [`config/fragments/signal/conversation/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/conversation/) | [Guide](../tutorials/signal/heuristic/conversation) |
| `domain` — learned signal | `domain` classifies the request topic family. | [`config/fragments/signal/domain/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/domain/) | [Guide](../tutorials/signal/learned/domain) |
| `embedding` — learned signal | `embedding` matches requests by semantic similarity to representative examples. | [`config/fragments/signal/embedding/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/embedding/) | [Guide](../tutorials/signal/learned/embedding) |
| `event` — heuristic signal | `event` routes structured event-like requests by event type, severity, urgency, or domain-specific action code. | [`config/fragments/signal/event/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/event/) | [Guide](../tutorials/signal/heuristic/event) |
| `fact-check` — learned signal | `fact-check` decides whether a prompt should be treated as evidence-sensitive traffic. | [`config/fragments/signal/fact-check/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/fact-check/) | [Guide](../tutorials/signal/learned/fact-check) |
| `jailbreak` — learned signal | `jailbreak` detects prompt-injection and jailbreak attempts before the Router commits to a route. | [`config/fragments/signal/jailbreak/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/jailbreak/) | [Guide](../tutorials/signal/learned/jailbreak) |
| `kb` — learned signal | `kb` binds routing signals to the output of a named knowledge base instance. | [`config/fragments/signal/kb/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/kb/) | [Guide](../tutorials/signal/learned/kb) |
| `keyword` — heuristic signal | `keyword` matches explicit words and phrases in the request. | [`config/fragments/signal/keyword/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/keyword/) | [Guide](../tutorials/signal/heuristic/keyword) |
| `language` — heuristic signal | `language` detects the request language and exposes it as a routing signal. | [`config/fragments/signal/language/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/language/) | [Guide](../tutorials/signal/heuristic/language) |
| `metadata` — heuristic signal | `metadata` matches bounded string values supplied by the caller in request metadata. | [`config/fragments/signal/metadata/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/metadata/) | [Guide](../tutorials/signal/heuristic/metadata) |
| `modality` — learned signal | `modality` detects whether a request should stay in text generation, switch into image generation, or support both. | [`config/fragments/signal/modality/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/modality/) | [Guide](../tutorials/signal/learned/modality) |
| `pii` — learned signal | `pii` detects sensitive personal data in requests. | [`config/fragments/signal/pii/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/pii/) | [Guide](../tutorials/signal/learned/pii) |
| `preference` — learned signal | `preference` infers response-style preferences from examples and classifier settings. | [`config/fragments/signal/preference/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/preference/) | [Guide](../tutorials/signal/learned/preference) |
| `reask` — learned signal | `reask` detects when the current user turn semantically repeats recent user turns in the same conversation. | [`config/fragments/signal/reask/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/reask/) | [Guide](../tutorials/signal/learned/reask) |
| `structure` — heuristic signal | `structure` detects request-shape facts such as many explicit questions, ordered workflow markers, or dense constraint phrasing. | [`config/fragments/signal/structure/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/structure/) | [Guide](../tutorials/signal/heuristic/structure) |
| `user-feedback` — learned signal | `user-feedback` detects correction, dissatisfaction, or escalation feedback from the conversation. | [`config/fragments/signal/user-feedback/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/user-feedback/) | [Guide](../tutorials/signal/learned/user-feedback) |

### Selection algorithms

| Family and type | Use it to | Reusable fragment | Guide |
| --- | --- | --- | --- |
| `automix` — selection algorithm | `automix` is an experimental selector that ranks candidate models by configured quality and cost plus internal verification and escalation estimates. | [`config/fragments/algorithm/selection/automix.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/automix.yaml) | [Guide](../tutorials/algorithm/selection/automix) |
| `hybrid` — selection algorithm | `hybrid` combines Elo ratings, Router-DC description similarity, AutoMix's one-model value estimate, and cost into one weighted candidate score. | [`config/fragments/algorithm/selection/hybrid.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/hybrid.yaml) | [Guide](../tutorials/algorithm/selection/hybrid) |
| `kmeans` — selection algorithm | `kmeans` sends a request to the model assigned to its nearest learned cluster. | [`config/fragments/algorithm/selection/kmeans.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/kmeans.yaml) | [Guide](../tutorials/algorithm/selection/kmeans) |
| `knn` — selection algorithm | `knn` chooses a candidate from the models that performed well on the most similar recorded requests. | [`config/fragments/algorithm/selection/knn.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/knn.yaml) | [Guide](../tutorials/algorithm/selection/knn) |
| `latency-aware` — selection algorithm | `latency_aware` ranks eligible candidates using observed TTFT and TPOT percentiles and selects the lowest relative-latency score. | [`config/fragments/algorithm/selection/latency-aware.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/latency-aware.yaml) | [Guide](../tutorials/algorithm/selection/latency-aware) |
| `mlp` — selection algorithm | `mlp` runs a trained neural classifier on CPU to map a request to a candidate model. | [`config/fragments/algorithm/selection/mlp.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/mlp.yaml) | [Guide](../tutorials/algorithm/selection/mlp) |
| `multi-factor` — selection algorithm | `multi_factor` ranks candidates by a configurable combination of quality, latency, cost, and load, then rejects any candidate that violates a hard limit. | [`config/fragments/algorithm/selection/multi-factor.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/multi-factor.yaml) | [Guide](../tutorials/algorithm/selection/multi-factor) |
| `prompt` — selection algorithm | `prompt` selects exactly one Model from the matched decision's Entrypoint assignment. | [`config/fragments/algorithm/selection/prompt.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/prompt.yaml) | [Guide](../tutorials/algorithm/selection/prompt) |
| `router-dc` — selection algorithm | `router_dc` embeds the request and each model description, then selects the candidate with the strongest semantic similarity. | [`config/fragments/algorithm/selection/router-dc.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/router-dc.yaml) | [Guide](../tutorials/algorithm/selection/router-dc) |
| `static` — selection algorithm | `static` provides deterministic model choice without metrics or learned state. | [`config/fragments/algorithm/selection/static.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/static.yaml) | [Guide](../tutorials/algorithm/selection/static) |
| `svm` — selection algorithm | `svm` uses a trained linear or RBF support-vector classifier to map request features to a candidate model. | [`config/fragments/algorithm/selection/svm.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/svm.yaml) | [Guide](../tutorials/algorithm/selection/svm) |

### Looper algorithms

| Family and type | Use it to | Reusable fragment | Guide |
| --- | --- | --- | --- |
| `confidence` — looper algorithm | `confidence` tries candidate models in order and stops when response confidence reaches a configured threshold. | [`config/fragments/algorithm/looper/confidence.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/looper/confidence.yaml) | [Guide](../tutorials/algorithm/looper/confidence) |
| `fusion` — looper algorithm | `fusion` asks several models to analyze a request and a judge model to synthesize one final answer. | [`config/fragments/algorithm/looper/fusion.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/looper/fusion.yaml) | [Guide](../tutorials/algorithm/looper/fusion) |
| `ratings` — looper algorithm | `ratings` calls every candidate model and returns one OpenAI-compatible choice per successful model. `max_concurrent` limits parallel work; it does not limit the total number of candidates executed. | [`config/fragments/algorithm/looper/ratings.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/looper/ratings.yaml) | [Guide](../tutorials/algorithm/looper/ratings) |
| `remom` — looper algorithm | `remom` runs several candidate models across bounded rounds and synthesizes their responses into one answer. | [`config/fragments/algorithm/looper/remom.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/looper/remom.yaml) | [Guide](../tutorials/algorithm/looper/remom) |
| `workflows` — looper algorithm | `workflows` runs a bounded, multi-step Router Flow behind one OpenAI-compatible model name. | [`config/fragments/algorithm/looper/workflows.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/looper/workflows.yaml) | [Guide](../tutorials/algorithm/looper/workflows) |

### Plugins and bundles

| Family and type | Use it to | Reusable fragment | Guide |
| --- | --- | --- | --- |
| `content-safety` — plugin bundle | Content Safety combines supported route-local safety plugins into one reusable policy. | [`config/fragments/plugin/content-safety/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/content-safety/) | [Guide](../tutorials/plugin/content-safety) |
| `context-compression` — route plugin | `context_compression` is a route-local request plugin that reduces large tool/function outputs before the selected provider receives the request. | [`config/fragments/plugin/context-compression/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/context-compression/) | [Guide](../tutorials/plugin/context-compression) |
| `fast-response` — route plugin | `fast_response` is a route-local plugin that returns a deterministic fallback message immediately. | [`config/fragments/plugin/fast-response/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/fast-response/) | [Guide](../tutorials/plugin/fast-response) |
| `hallucination` — route plugin | `hallucination` is a route-local plugin for fact-checking and response-quality screening after the decision already matched. | [`config/fragments/plugin/hallucination/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/hallucination/) | [Guide](../tutorials/plugin/hallucination) |
| `header-mutation` — route plugin | `header_mutation` is a route-local plugin for adding, updating, or deleting downstream headers. | [`config/fragments/plugin/header-mutation/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/header-mutation/) | [Guide](../tutorials/plugin/header-mutation) |
| `image-gen` — route plugin | `image_gen` is a route-local plugin for handing a matched route off to an image-generation backend. | [`config/fragments/plugin/image-gen/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/image-gen/) | [Guide](../tutorials/plugin/image-gen) |
| `memory` — route plugin | `memory` is a route-local plugin for retrieving and storing conversation memory. | [`config/fragments/plugin/memory/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/memory/) | [Guide](../tutorials/plugin/memory) |
| `rag` — route plugin | `rag` retrieves external context for a matched route before generation. | [`config/fragments/plugin/rag/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/rag/) | [Guide](../tutorials/plugin/rag) |
| `request-params` — route plugin | `request_params` is a route-local plugin that validates and trims OpenAI Chat Completions request bodies before they are forwarded to backends. | [`config/fragments/plugin/request-params/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/request-params/) | [Guide](../tutorials/plugin/request-params) |
| `response-cache` — route plugin | `response_cache` is the route-local plugin for reusing exact or semantically compatible prior responses. | [`config/fragments/plugin/response-cache/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/response-cache/) | [Guide](../tutorials/plugin/response-cache) |
| `response-jailbreak` — route plugin | `response_jailbreak` is a route-local plugin for screening the model response before it is returned. | [`config/fragments/plugin/response-jailbreak/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/response-jailbreak/) | [Guide](../tutorials/plugin/response-jailbreak) |
| `router-replay` — route plugin | `router_replay` is a route-local plugin for overriding replay/debug capture on one route. | [`config/fragments/plugin/router-replay/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/router-replay/) | [Guide](../tutorials/plugin/router-replay) |
| `system-prompt` — route plugin | `system_prompt` is a route-local plugin for inserting or modifying the system prompt on matched traffic. | [`config/fragments/plugin/system-prompt/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/system-prompt/) | [Guide](../tutorials/plugin/system-prompt) |
| `tool-selection` — route plugin | `tool_selection` is a decision plugin that controls how tools are chosen for a matched route. | [`config/fragments/plugin/tool-selection/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/tool-selection/) | [Guide](../tutorials/plugin/tool-selection) |
| `tools` — route plugin | `tools` is a route-local plugin for tool filtering and semantic tool selection. | [`config/fragments/plugin/tools/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/tools/) | [Guide](../tutorials/plugin/tools) |

<!-- END GENERATED CONFIGURATION CATALOG -->

## Minimal example

```yaml
version: v0.3

listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
    timeout: 300s

providers:
  models:
    - name: local/general
      provider_model_id: my-served-model
      backend_refs:
        - provider: vllm
          base_url: http://host.docker.internal:8000/v1
      control:
        retry:
          count: 1
          on: [unavailable]
        timeout:
          request: 60s
          stream: 10m
      pricing:
        input_cost_per_million_tokens: "0.10"
        output_cost_per_million_tokens: "0.40"

routing:
  modelCards:
    - name: local/general
      description: General chat model.
      capabilities: [chat, tools]
      modality: text

recipes:
  - name: explain
    routing:
      strategy: priority
      signals:
        keywords:
          - name: needs_explanation
            operator: OR
            keywords: ["explain", "walk me through"]
      decisions:
        - name: explanatory_answer
          description: Prefer an explanatory answer when the request asks for one.
          priority: 100
          rules:
            operator: AND
            conditions: [{type: keyword, name: needs_explanation}]

entrypoints:
  - model_names: [vllm-sr/explain, explain]
    recipe: explain
    assignments:
      explanatory_answer:
        models: [{model: local/general}]

global:
  billing:
    currency: USD
  services:
    observability:
      metrics:
        enabled: true
```

Requests enter a Recipe only through an explicit Entrypoint alias. A concrete
Model name selects that Model directly; there is no implicit default Recipe or
hidden automatic alias.

## Validate and serve

```bash
vllm-sr validate --config config.yaml
vllm-sr serve
# Or select another immutable bootstrap manifest.
vllm-sr serve --config /path/to/config.yaml
```

Validation catches schema errors, unresolved references, incompatible recipe
boundaries, invalid provider bindings, and unsupported plugin or algorithm
settings before the Router starts. The ordinary local command reads
`config.yaml` from the current workspace; `--config` selects another immutable
v0.3 bootstrap manifest. It is not a Model, Recipe, or routing-policy operand.

## Environment references and secrets

Prefer keeping credential values outside the YAML file. Reference an environment
variable from the physical backend that needs it:

```yaml
providers:
  models:
    - name: remote/general
      provider_model_id: provider/general
      backend_refs:
        - provider: openai-compatible
          base_url: https://models.example.com/v1
          api_key_env: MODEL_API_KEY
```

Supported string substitutions are:

- `${VAR}` and `$VAR`;
- `${VAR:-default}` when `VAR` is unset or empty;
- `${VAR-default}` when `VAR` is unset; and
- `$$` for a literal `$`.

The local Docker target forwards variables referenced by the selected bootstrap
config. Kubernetes deployments place sensitive environment values in Secrets
rather than ConfigMaps or Helm values. See
[Security Hardening](security-hardening).

The existing literal `backend_refs[].api_key` input remains accepted for file-backed
deployments. It is mutually exclusive with `api_key_env`; protect the manifest as a
credential whenever the literal form is used.

## Secure the Management listener

A remotely exposed Management API requires Router-terminated TLS. Reference PEM
material through absolute secret-file paths or environment-variable names; do
not place certificate or private-key literals in Router YAML.

```yaml
version: v0.3
global:
  stores:
    management:
      postgres:
        dsn_env: VLLM_SR_POSTGRES_DSN
  services:
    management_api:
      enabled: true
      bind_address: 0.0.0.0
      port: 8080
      tls:
        certificate_file: /run/secrets/management-server.pem
        private_key_file: /run/secrets/management-server-key.pem
        # Optional. When present, every client must present a certificate
        # issued by this CA.
        client_ca_bundle_file: /run/secrets/management-client-ca.pem
      auth:
        mode: router
```

Each TLS value accepts exactly one `_file` or `_env` reference. An environment
source contains the PEM payload, not another file path. At startup the Router
parses the certificate chain, verifies that the private key matches, requires a
DNS or IP subject alternative name, and enforces a validity margin. Missing or
invalid material prevents the listener from binding. Remote Management connections use
TLS 1.3 or newer; plaintext connections are rejected. A loopback-only listener
may use the documented local development policy.

Mount private-key files with owner-only permissions (`0400` or `0600`); a
group- or world-readable key fails startup. Rotate mounted
material through an atomic Secret replacement. The Router reloads each
listener context on a bounded interval. A failed replacement retains the last
valid context but makes readiness fail until valid material is installed;
replicas also leave readiness before the active certificate expires.

### Dynamic access resources

YAML configures only the Management and access services, their stores, and their
secret references. Users, Teams, inference API keys, grants, rate-limit policies,
bindings, counters, usage, and audit are versioned Management API resources in
PostgreSQL and the runtime store; none of those resources is accepted in Router
YAML.

### Bind Agent calls to the public inference front door

Router replicas run Agent workers without another Agent container. Give
them a stable address for the deployment's ordinary public inference listener:

```yaml
global:
  services:
    agent:
      public_inference_endpoint: https://inference.example.com/v1/chat/completions
```

The URL must use HTTP or HTTPS and end exactly at `/v1/chat/completions`. It is
operator-owned bootstrap configuration: it never comes from an Agent profile,
Tool Source, Dashboard setting, or published routing snapshot. Do not point it
at a physical model backend. Agent turns use short-lived delegated API keys at
this endpoint, preserving Model visibility, quota enforcement, request logs,
and actual usage settlement.

### Usage storage lifecycle

Router-native access uses fixed UTC-month PostgreSQL partitions for request,
dispatch, and attempt facts. The safe default retains raw usage indefinitely:

```yaml
global:
  services:
    access:
      usage_storage:
        create_ahead_months: 2
        maintenance_interval: 5m
        # raw_retention: 2160h # optional explicit 90-day policy
```

`create_ahead_months` is between 1 and 24, and `maintenance_interval` is between
one minute and 24 hours. `raw_retention` is empty unless an operator explicitly
chooses a duration. Retention never removes settlement digest tombstones,
rollups, audit history, replay-referenced facts, or facts needed by unresolved
usage reconciliation. See
[API Keys, Access, and Usage](../tutorials/global/access-and-usage#operate-usage-storage)
for the lifecycle and preview-schema migration procedure.

### Provider integrations

Provider definitions are control-plane integrations, not inference-time product
branches. The application composes an immutable Integration Registry, and the
Management API exposes its safe catalog to clients. A Model create or import
request selects a Provider and submits only schema-approved connection values; the
control plane compiles them into the provider-neutral backend stored in the Model
revision.

The data plane receives only immutable compiled snapshots with canonical
origins, non-secret connection values, ProviderCredential references, and
stable protocol adapters. Product names, logos, forms, discovery rules, and
compiler plugins stay in the control plane. File loading runs the same compiler
over the readable source manifest. See the
[Provider catalog proposal](../proposals/router-native-access-control-provider-catalog)
for the extension and rollout contract.

## Entrypoints and recipes

An Entrypoint maps one or more public model aliases to a Recipe and its complete
decision assignments. A Recipe owns
its signal, projection, decision, algorithm, plugin, cache, replay, learning,
and routing state. Providers, stores, and router-owned classifier assets may be
shared without allowing policy state to cross recipe boundaries.

In the common schema, `entrypoints[].model_names` lists public names,
`entrypoints[].recipe` selects a Recipe by name, and `recipes[].routing` contains
that Recipe's policy. The compiler resolves readable names to immutable snapshot
identities; generated identities never appear in authoring YAML.

Each Entrypoint assigns Models to every Decision name without copying the Recipe:

```yaml
entrypoints:
  - model_names: [company/assistant, assistant]
    recipe: assistant
    assignments:
      quick:
        models:
          - model: local/fast
      complex:
        models:
          - model: remote/frontier
            reasoning: {enabled: true, effort: high}
```

`assignments` is keyed by Decision name. Each value is an assignment set
with a non-empty `models` list. `priority`, `weight`, `lora`, and typed
`reasoning` controls are optional assignment values. A single-dispatch decision
can also define a closed priority `fallback` policy. URLs, credentials,
invocation control, and pricing stay on the Model.

Validation rejects an unknown Recipe or Model name, missing decision assignment,
unknown Decision name, invalid LoRA/reasoning control, ambiguous rule, or
cross-namespace reference. A claimed alias with no matching rule fails
closed; it never falls through to a concrete Model.

Each Recipe should contain an unconditional lowest-priority decision. Every
Entrypoint rule must assign that decision explicitly; there is no hidden
default Model. The virtual Entrypoint name never reaches a backend.

See
[Models, Entrypoints, and Serving](../tutorials/global/models-entrypoints-serving)
for built-in Recipes, CLI serving, connections, assignments, and
packaging. See
[Virtual Models](../tutorials/global/entrypoints-and-recipes)
for the complete schema.

## Configuration workflows

The canonical document can be authored or applied through several interfaces:

- local CLI and YAML;
- Dashboard setup and visual routing tools;
- Helm or `vllm-sr serve --target k8s`;
- the Kubernetes Operator; and
- the routing DSL.

[Configuration Workflows](configuration-workflows) explains which interface
owns which part of the document and how to avoid competing sources of truth.

## Reference sources

- [`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml)
  is the exhaustive canonical example.
- [`config/fragments/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments)
  contains reusable signal, decision, algorithm, and plugin fragments.
- [Providers and routing tutorials](../tutorials/global/overview) describe
  shared runtime configuration.
- [Upgrade and rollback](upgrade-rollback) explains the strict offline conversion
  of fields removed or renamed by the current v0.3 contract.

Avoid copying the exhaustive example as an application config. Start with the
smallest document that describes the deployment, then add only the capabilities
and services it uses.
