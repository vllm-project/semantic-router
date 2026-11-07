---
title: Configuration
description: Understand the canonical v0.3 YAML document and where routing, providers, recipes, services, and secrets belong.
---

# Configuration

Semantic Router uses one canonical YAML document across the CLI, Dashboard,
Helm, and Operator. The top-level structure is:

```yaml
version:
listeners:
providers:
evaluation:
routing:
entrypoints:
recipes:
global:
```

Most deployments begin with `version`, `listeners`, `providers`, and one
top-level `routing` profile. Add `entrypoints` and `recipes` when one deployment
needs several isolated policies. Add `global` settings only for shared services
or runtime behavior that differs from the built-in defaults.

## What belongs where

| Section | Owns |
| --- | --- |
| `version` | Canonical schema version. Use `v0.3`. |
| `listeners` | Public Router listeners: address, port, the default time for a whole call including its streamed response (`timeout`, 300s), which also bounds a stalled stream, so `0s` lifts both limits (to lift only the call limit for one model, set that model's `total_timeout: 0s`), optional client API keys, an optional Chat model allowlist (`models`; empty accepts all), separate native System One grants (`systemone.models`; omission publishes none) and the longest they wait for a model (`systemone.max_deadline_ms`, 10 minutes by default), optional one-way TLS (`tls.cert_file`, `tls.key_file`), and the identity sources a listener trusts (`identity.trust_headers`, `identity.trusted_peers`; by default none), which the Router honors in standalone mode. |
| `providers` | Logical provider models, physical backend endpoints, pricing, capabilities, and defaults. |
| `evaluation` | Optional operator-owned benchmark definitions, versioned index DAGs, and model-linked records. |
| `routing` | The default recipe: model cards, signals, projections, decisions, strategy, algorithms, and route plugins. |
| `entrypoints` | Public virtual model aliases mapped to the default or a named recipe. |
| `recipes` | Additional isolated routing profiles that share providers and global infrastructure. |
| `global` | Router services, stores, integrations, observability, learning, and the models the Router asks (`global.model_catalog`; see the [model catalog reference](model-catalog.md)). |

Keep these boundaries clear:

- signals detect facts;
- projections combine evidence;
- decisions define eligibility and route policy;
- algorithms choose or coordinate candidate models;
- plugins add behavior at route-specific hook points; and
- providers bind logical model names to inference endpoints.

Provider pricing belongs beside each concrete model under
`providers.models[].pricing`. It accepts an optional uppercase three-letter
`currency` plus non-negative `prompt_per_1m`, `completion_per_1m`,
`cached_input_per_1m`, and `cache_write_per_1m` rates. Routing model cards do not
repeat deployment prices or credentials.

Evaluation measurements belong in `evaluation.records[]` and reference a
canonical Model Card identity through `model`. Built-in benchmark IDs work
directly; define new benchmark semantics and indices beside the records under
`evaluation`. See [Custom evaluations](../benchmarking/custom-evaluations).

Use [Protocol Compatibility](protocol-compatibility) to choose the model's
backend `api_format`. Then see
[Backend Target Compatibility](backend-target-compatibility) before moving its
bindings between Docker, Helm, the Operator, and Dashboard workflows. The
target matrix distinguishes canonical pass-through from Kubernetes discovery
and records which URL, path, weight, and provider fields each surface
preserves.

Router-wide debugging surfaces stay closed by default.
`global.services.observability.profiling` serves Go `pprof` endpoints, and only
when it is explicitly enabled; it then binds `127.0.0.1:6060` so profiles never
reach a routable interface without an explicit `bind` change. The switch is read
once at startup, so changing it requires a Router restart. See
[API and Observability](../tutorials/global/api-and-observability).

The models the Router asks, the ones that answer signals, embed text and
rerank documents, are deployments under `global.model_catalog.deployments`. The
decision model answers every question-style signal by default, in the
[model runtime](model-runtime/overview.md). A binding
(`global.model_catalog.bindings.<signal>`, or a recipe's
`routing.model_bindings.<signal>`) or a rule's own `deployment` names another
deployment, such as a remote `classify` service for domain, PII or complexity.
[Choose a model for a signal](choose-a-model-for-a-signal.md) explains the rule,
and the [model catalog reference](model-catalog.md) lists every deployment
field. The earlier `variant`, `use_modernbert` and `use_mmbert_32k` selectors
are gone; `vllm-sr config migrate` removes them.

PII rules default to the legacy prompt and optional history scope. To inspect
textual tool results, opt in per rule with `source: tool_result`; the router
uses the neutral protocol representation, so the same rule applies to the
supported chat, responses, and messages wire formats. Tool-result scope does
not implicitly include the prompt or conversation history, and unsupported
source values fail configuration validation.

A complexity rule names a remote scorer with its `deployment`: a `classify`
deployment that returns one score, which each rule turns into a verdict
through its own `hard_above`/`easy_below` boundaries (or
`hard_below`/`easy_above` for a score that falls as difficulty rises). One
score drives every complexity rule, so the rules of a routing profile name the
same scorer. `threshold` remains the symmetric shorthand for the local signed
margin. A remote score reports no confidence, so decisions gated on those
rules rank on the engine's structural default; the Router warns at startup.
The remote call is visible through `llm_remote_connector_*` and
`llm_complexity_*` metrics, and a scorer failure is recorded on every
complexity rule's signal errors rather than dropped.

A remote PII classifier is a `classify` deployment that `pii` is bound to. It
returns entity spans as code-point offsets into the exact request string it
was sent, and every label it returns must exist in the configured
`pii_mapping_path`; a response the span contract rejects is a failure rather
than a clean "no PII" result.
`global.model_catalog.modules.classifier.pii.on_error` selects what such a
failure, or a provider-declared `truncated_at`, does to the rule that consumed
it: `allow` (the default) treats the content as not matching, `block` matches
it as `classification_error`. Spans returned before a declared truncation
still count under both policies.

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
| `authz` — heuristic signal | `authz` turns identity and policy bindings into reusable routing inputs under `routing.signals.role_bindings`. | [`config/fragments/signal/authz/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/authz/) | [Guide](../tutorials/signal/heuristic/authz) |
| `complexity` — learned signal | `complexity` estimates whether a request is `easy`, `medium`, or `hard` by comparing it with configured example sets. | [`config/fragments/signal/complexity/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/complexity/) | [Guide](../tutorials/signal/learned/complexity) |
| `context` — heuristic signal | `context` detects requests that need a larger effective context window. | [`config/fragments/signal/context/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/context/) | [Guide](../tutorials/signal/heuristic/context) |
| `conversation` — heuristic signal | `conversation` routes on chat structure and protocol facts, such as message count, developer instructions, available tools, explicit tool-use constraints, or an active tool loop. | [`config/fragments/signal/conversation/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/conversation/) | [Guide](../tutorials/signal/heuristic/conversation) |
| `domain` — learned signal | `domain` classifies the request topic family. | [`config/fragments/signal/domain/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/domain/) | [Guide](../tutorials/signal/learned/domain) |
| `fact-check` — learned signal | `fact-check` decides whether a prompt should be treated as evidence-sensitive traffic. | [`config/fragments/signal/fact-check/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/fact-check/) | [Guide](../tutorials/signal/learned/fact-check) |
| `hallucination` — learned signal | `hallucination` checks the model's answer against the grounding context the request carried, such as tool results or retrieved documents, and reports the claims that context does not support. | [`config/fragments/signal/hallucination/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/hallucination/) | [Guide](../tutorials/signal/learned/hallucination) |
| `input-modality` — heuristic signal | `input_modality` deterministically matches which kinds of input — `text`, `image`, `audio`, or `video` — are present in the parsed request. | [`config/fragments/signal/input-modality/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/input-modality/) | [Guide](../tutorials/signal/heuristic/input-modality) |
| `jailbreak` — learned signal | `jailbreak` detects prompt-injection and jailbreak attempts before the Router commits to a route. | [`config/fragments/signal/jailbreak/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/jailbreak/) | [Guide](../tutorials/signal/learned/jailbreak) |
| `keyword` — heuristic signal | `keyword` matches explicit words and phrases in the request. | [`config/fragments/signal/keyword/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/keyword/) | [Guide](../tutorials/signal/heuristic/keyword) |
| `language` — heuristic signal | `language` detects the request language and exposes it as a routing signal. | [`config/fragments/signal/language/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/language/) | [Guide](../tutorials/signal/heuristic/language) |
| `metadata` — heuristic signal | `metadata` matches bounded string values supplied by the caller in request metadata. | [`config/fragments/signal/metadata/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/metadata/) | [Guide](../tutorials/signal/heuristic/metadata) |
| `modality` — learned signal | `modality` detects whether a request should stay in text generation, switch into image generation, or support both. | [`config/fragments/signal/modality/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/modality/) | [Guide](../tutorials/signal/learned/modality) |
| `pii` — learned signal | `pii` detects sensitive personal data in requests. | [`config/fragments/signal/pii/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/pii/) | [Guide](../tutorials/signal/learned/pii) |
| `refusal` — learned signal | `refusal` asks the decision model whether the model's answer declines the request. | [`config/fragments/signal/refusal/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/refusal/) | [Guide](../tutorials/signal/learned/refusal) |
| `safety` — learned signal | The `safety` signal predicts content risks. | [`config/fragments/signal/safety/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/safety/) | [Guide](../tutorials/signal/learned/safety) |
| `systemone` — learned signal | `systemone` asks a System One model a typed question about the request and turns the answer into a routing fact. | [`config/fragments/signal/systemone/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/systemone/) | [Guide](../tutorials/signal/learned/systemone) |
| `turn-relation` — learned signal | `turn_relation` asks the decision model how the current user turn relates to the conversation before it: it `continue`s the current topic, `reask`s an earlier turn, `shift`s to an unrelated one, or `return`s to an earlier topic. | [`config/fragments/signal/turn-relation/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/turn-relation/) | [Guide](../tutorials/signal/learned/turn-relation) |
| `user-feedback` — learned signal | `user-feedback` detects correction, dissatisfaction, or escalation feedback from the conversation. | [`config/fragments/signal/user-feedback/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/user-feedback/) | [Guide](../tutorials/signal/learned/user-feedback) |

### Selectors

| Family and type | Use it to | Reusable fragment | Guide |
| --- | --- | --- | --- |
| `latency-aware` — selector | To choose the fastest eligible candidate, weight Multi Factor on latency alone. | [`config/fragments/algorithm/selection/latency-aware.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/latency-aware.yaml) | [Guide](../tutorials/algorithm/selection/latency-aware) |
| `multi-factor` — selector | `multi_factor` chooses one candidate from quality, latency, cost, and load. | [`config/fragments/algorithm/selection/multi-factor.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/multi-factor.yaml) | [Guide](../tutorials/algorithm/selection/multi-factor) |
| `random` — selector | `random` picks one model uniformly from a decision's eligible candidates. | [`config/fragments/algorithm/selection/random.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/random.yaml) | [Guide](../tutorials/algorithm/selection/random) |
| `static` — selector | `static` provides deterministic model choice without metrics or learned state. | [`config/fragments/algorithm/selection/static.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/static.yaml) | [Guide](../tutorials/algorithm/selection/static) |
| `systemone` — selector | `systemone` asks a System One model which of a routing decision's `modelRefs` should answer the request. | [`config/fragments/algorithm/selection/systemone.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/systemone.yaml) | [Guide](../tutorials/algorithm/selection/systemone) |

### Executors

| Family and type | Use it to | Reusable fragment | Guide |
| --- | --- | --- | --- |
| `escalate` — executor | `escalate` calls a decision's candidates one at a time, in order, and returns the first answer it accepts. | [`config/fragments/algorithm/looper/escalate.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/looper/escalate.yaml) | [Guide](../tutorials/algorithm/looper/escalate) |
| `fusion` — executor | `fusion` asks several models to answer a request and a judge model to synthesize one final answer. | [`config/fragments/algorithm/looper/fusion.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/looper/fusion.yaml) | [Guide](../tutorials/algorithm/looper/fusion) |
| `ratings` — executor | `ratings` calls every candidate model and returns one OpenAI-compatible choice per successful model. `max_concurrent` limits parallel work; it does not limit the total number of candidates executed. | [`config/fragments/algorithm/looper/ratings.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/looper/ratings.yaml) | [Guide](../tutorials/algorithm/looper/ratings) |
| `remom` — executor | `remom` runs several candidate models across bounded rounds and synthesizes their responses into one answer. | [`config/fragments/algorithm/looper/remom.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/looper/remom.yaml) | [Guide](../tutorials/algorithm/looper/remom) |
| `workflows` — executor | `workflows` runs a bounded, multi-step Router Flow behind one OpenAI-compatible model name. | [`config/fragments/algorithm/looper/workflows.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/looper/workflows.yaml) | [Guide](../tutorials/algorithm/looper/workflows) |

### Plugins and bundles

| Family and type | Use it to | Reusable fragment | Guide |
| --- | --- | --- | --- |
| `content-safety` — plugin bundle | Content Safety combines supported route-local safety plugins into one reusable policy. | [`config/fragments/plugin/content-safety/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/content-safety/) | [Guide](../tutorials/plugin/content-safety) |
| `context` — route plugin | `context` is the route-local plugin that assembles the request's context. | [`config/fragments/plugin/context/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/context/) | [Guide](../tutorials/plugin/context) |
| `fast-response` — route plugin | `fast_response` is a route-local plugin that returns a deterministic fallback message immediately. | [`config/fragments/plugin/fast-response/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/fast-response/) | [Guide](../tutorials/plugin/fast-response) |
| `header-mutation` — route plugin | `header_mutation` is a route-local plugin for adding, updating, or deleting downstream headers. | [`config/fragments/plugin/header-mutation/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/header-mutation/) | [Guide](../tutorials/plugin/header-mutation) |
| `masking` — route plugin | `masking` is a route-local plugin that detects personal data (emails, phone numbers, names, and other configured entity types) in the provider-bound request and replaces each occurrence with a deterministic placeholder, such as `[EMAIL_ADDRESS_0]`, immediately before the request is encoded and sent to the model provider. | [`config/fragments/plugin/masking/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/masking/) | [Guide](../tutorials/plugin/masking) |
| `prompt-cache` — route plugin | `prompt_cache` is a route-local plugin that inserts Anthropic prompt-cache breakpoints on a matched route. | [`config/fragments/plugin/prompt-cache/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/prompt-cache/) | [Guide](../tutorials/plugin/prompt-cache) |
| `request-params` — route plugin | `request_params` is a route-local plugin that validates and trims OpenAI Chat Completions request bodies before they are forwarded to backends. | [`config/fragments/plugin/request-params/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/request-params/) | [Guide](../tutorials/plugin/request-params) |
| `response-cache` — route plugin | `response_cache` is the route-local plugin for reusing exact or semantically compatible prior responses. | [`config/fragments/plugin/response-cache/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/response-cache/) | [Guide](../tutorials/plugin/response-cache) |
| `response-guard` — route plugin | `response_guard` is a route-local plugin that blocks or annotates the model's answer on the signals it is checked with. | [`config/fragments/plugin/response-guard/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/response-guard/) | [Guide](../tutorials/plugin/response-guard) |
| `router-replay` — route plugin | Use Router Replay to inspect requests in Dashboard Insights: the selected route, model, token usage, response, and tool trajectory. | [`config/fragments/plugin/router-replay/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/router-replay/) | [Guide](../tutorials/plugin/router-replay) |
| `system-prompt` — route plugin | `system_prompt` is a route-local plugin for inserting or modifying the system prompt on matched traffic. | [`config/fragments/plugin/system-prompt/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/system-prompt/) | [Guide](../tutorials/plugin/system-prompt) |
| `tools` — route plugin | `tools` is the route-local plugin for the request's tools. | [`config/fragments/plugin/tools/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/tools/) | [Guide](../tutorials/plugin/tools) |

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
  defaults:
    model: local/general
  models:
    - name: local/general
      provider_model_id: my-served-model
      backend_refs:
        - name: primary
          endpoint: host.docker.internal:8000
          protocol: http
          provider: vllm

routing:
  strategy: priority
  modelCards:
    - name: local/general
      modality: text
      capabilities: [chat]
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
        conditions:
          - type: keyword
            name: needs_explanation
      modelRefs:
        - model: local/general

global:
  services:
    observability:
      metrics:
        enabled: true
```

### Model configuration

Models can inherit identity and reasoning from the built-in catalog or define a
private model locally. This catalog-backed example lets the selected Provider
mapping supply the native model ID, protocol, reasoning transport, and request
path:

```yaml
providers:
  defaults:
    model: production
    reasoning_effort: medium
  models:
    - name: production
      catalog: openai/gpt-5.6-sol
      backend_refs:
        - provider: openai
          auth: {api_key_env: OPENAI_API_KEY}
```

The `name` remains the local Router alias. A private or newly released model
omits `catalog` and can optionally define a Model Card and reasoning contract
under that alias. Start with [Configure models](model-configuration), then use
[Model configuration patterns](model-configuration-patterns) to compare the
catalog, custom, reasoning, Provider, and replica combinations. The
[Model and provider Day-0 guide](../community/model-provider-day-0-support.md)
is for contributors adding reusable support to the repository catalog.

Classifier backend failures remain `Unknown` while the complete boolean tree
is evaluated. Set `rules.on_unknown` to `no_match`, `match`, or `fail_request`
to resolve an undetermined terminal result. Omitting it preserves the existing
classifier-family error behavior.

Requests using an automatic model alias enter the default `routing` profile.
A concrete provider model name is a direct pass-through request and bypasses
recipe signals, decisions, route plugins, cache, learning, and session routing.

## Validate and serve

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
```

Validation catches schema errors, unresolved references, incompatible recipe
boundaries, invalid provider bindings, and unsupported plugin or algorithm
settings before the Router starts.

For portable model-free Recipes, set
`routing.decisions[].algorithm.minimum_candidates` to the smallest pool that
preserves the decision's intended behavior. Empty built-in assets remain
valid, while a published Entrypoint is rejected if its concrete assignments do
not meet the declared cardinality.

## Environment references and secrets

Keep credentials outside the YAML file:

```yaml
api_key: ${MODEL_API_KEY}
```

Supported string substitutions are:

- `${VAR}` and `$VAR`;
- `${VAR:-default}` when `VAR` is unset or empty;
- `${VAR-default}` when `VAR` is unset; and
- `$$` for a literal `$`.

For a custom Recipe, authorize required host variables explicitly with
`--recipe-env NAME`. Kubernetes deployments place sensitive environment values
in Secrets rather than ConfigMaps or Helm values. See
[Security Hardening](security-hardening).

## Entrypoints and recipes

Without an explicit entrypoint for `recipe: default`, the top-level routing
recipe is published as `vllm-sr/auto`. To replace that name, declare its
`model_names` in `entrypoints` with `recipe: default`. Named recipes require
their own entrypoints. `MoM` has no implicit special meaning.

An entrypoint maps one or more public model aliases to a recipe. A recipe owns
its signal, projection, decision, algorithm, plugin, cache, learning,
and routing state. Providers, stores, and router-owned classifier assets may be
shared without allowing policy state to cross recipe boundaries.

Set `limits.max_response_bytes` on a remote classifier deployment, and
`max_response_bytes` on the MCP classifier module, to cap one upstream
classifier response.

In the schema, `entrypoints[].model_names` lists the public aliases,
`entrypoints[].recipe` selects a named recipe, and `recipes[].routing` contains
that recipe's policy.

`global.router.strategy` and `global.router.fallback` provide shared defaults.
Top-level `routing` configures the default recipe only; each named
`recipes[].routing` resolves its own strategy and fallback independently from
`global.router`. Named recipes never inherit the default recipe's routing
settings. A decision's fallback overrides its own recipe's effective fallback.
Sparse overrides such as `fallback: {enabled: false}` preserve the remaining
shared defaults in both default and named recipes. The runtime strategy default
is `priority`.

If no decision matches, the recipe uses `providers.defaults.model`.
The virtual entrypoint name never reaches a backend.

See
[Models, Entrypoints, and Serving](../tutorials/global/models-entrypoints-serving)
for built-in virtual models, CLI serving, backend binding, forking, packaging,
and migration. See
[Virtual Models](../tutorials/global/entrypoints-and-recipes)
for the complete schema.

### Recipe candidate requirements

Set optional candidate requirements inside the default or a named recipe's
`routing` block:

```yaml
candidate_requirements:
  capabilities: declared
  context: known_limits
```

`capabilities: declared` requires the assigned model to declare support for the
request's task, including tools and image input, as well as a compatible provider
protocol. `context: known_limits` checks estimated input demand plus the effective
output reserve against declared model limits, and refuses a request only when it
does not fit even without the input estimate's reserves for structured data and
media. The request must supply an output
bound, or its decision must configure `request_params.default_max_tokens` as a
positive integer or `auto` (see below).
That default applies only when the caller omits the bound; `max_tokens_limit` then
caps it as usual. A model's maximum output capacity is not a request default.
Missing required model facts or an effective output bound make a candidate ineligible. Input accounting remains estimated, especially for
multimodal content; this is not an exact provider token-capacity guarantee. Omit a
field to retain that dimension's existing compatibility behavior.

For example, a decision can supply the bound through its existing plugin:

```yaml
plugins:
  - type: request_params
    configuration:
      default_max_tokens: 4096
      max_tokens_limit: 8192
```

To use each model's available output capacity when the caller omits a limit:

```yaml
algorithm:
  type: multi_factor
  multi_factor:
    expected_output_tokens: 4096
plugins:
  - type: request_params
    configuration:
      default_max_tokens: auto
```

`auto` requires an explicit `vllm` provider, one backend per model, the OpenAI
Chat format, and declared context and output limits. Enable the backend's
`/v1/chat/completions/render` API with `vllm serve --enable-scale-out` on a vLLM
version that supports it. The Router preprocesses each candidate's complete
provider request without generating an answer, then repeats that check after
final request changes. It uses the actual templated input length to allocate
`min(max_output_tokens, context_window - input_tokens)`. A smaller backend limit
than declared is a configuration error; align the deployment and model card.
Explicit caller limits keep their existing behavior and need no render call.
`max_tokens_limit` still caps the resulting output budget.

`expected_output_tokens` is a positive cost forecast, **not an output limit**.
It is required for `multi_factor` with `auto`, and is bounded by the caller's
limit or each candidate's available capacity. Different maximum context windows
do not by themselves make a model's forecast more expensive. Without this field,
non-automatic requests retain their existing cost calculation.

A request that fits a candidate is not truncated based on a text-length estimate.
If every compatible candidate reports input overflow, the Router applies the
configured context compression policy once and verifies the result with the
backend. Compression can be conservative; final capacity checking is exact.
Without a suitable compression policy, overflow remains an error. An unavailable
or invalid render API fails clearly and never triggers compression. Preview
reports `execution_required` because it does not have the complete provider
request. Automatic budgets currently support text and tool requests, excluding
multimodal input, provider truncation, LoRA, Looper, and shadow dispatch.

For multi-factor selection, `latency_metric: ttft` compares time to first token;
`tpot` compares time per output token. Omission preserves the existing TPOT-then-TTFT
fallback. Pair the metric with explicit quality evidence and a lexicographic
objective when quality is a floor rather than a score to trade away.

Discover the current contract with
`vllm-sr config schema --section routing.candidate_requirements`.
DSL `ROUTING` blocks support the same object. Kubernetes CRD emission preserves
these requirements for the default routing profile; named recipes and entrypoints
require canonical YAML and are rejected by CRD emission rather than discarded.

### Replay capture defaults and decision overrides

`global.services.router_replay` owns shared storage, retention, enablement and
capture defaults. A decision's `router_replay` plugin overrides only explicitly
configured capture fields. Omitted fields inherit; `enabled: false` disables
capture for that decision, while `enabled: true` can opt in when the global
default is disabled. There is no `routing.data_policy` or recipe-level Replay
configuration.

Set `capture_personal_data: false` globally or in the decision plugin to retain
routing evidence while omitting content when PII is detected or its status is
unknown. Missing PII detectors suppress content conservatively rather than
preventing startup. Client tool results stay redacted unless
`capture_tool_results: true`, which requires `capture_personal_data: true`;
startup refuses the combination with `capture_personal_data: false`. Rejected
requests without a selected decision use global capture defaults. These
controls do not govern other stores or backend retention.
See [Router Replay](../tutorials/plugin/router-replay) for the complete fields and
examples, or inspect `vllm-sr config schema --section global.services.router_replay`.

## Configuration workflows

The canonical document can be authored or applied through several interfaces:

- local CLI and YAML;
- Dashboard setup and visual routing tools;
- Helm or `vllm-sr serve --target k8s`;
- the Kubernetes Operator; and
- the routing DSL.

[Configuration Workflows](configuration-workflows) explains which interface
owns which part of the document and how to avoid competing sources of truth.
[Configuration Contract](configuration-contract) describes the generated
machine-readable schema, Router discovery and validation APIs, and the safe
authoring loop for tools and agents. [Configuration Management](configuration-management)
explains how a change activates on a running Router, how a rejected change is
reported, and how to list and roll back versions.

## Reference sources

- [`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml)
  is the exhaustive canonical example.
- [`config/fragments/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments)
  contains reusable signal, decision, algorithm, and plugin fragments.
- [Providers and routing tutorials](../tutorials/global/overview) describe
  shared runtime configuration.
- [Unified Config Contract v0.3](../proposals/unified-config-contract-v0-3)
  records the design behind the current contract.
- [Configuration Contract](configuration-contract) is the live discovery and
  validation contract for the current Router build.

Avoid copying the exhaustive example as an application config. Start with the
smallest document that describes the deployment, then add only the capabilities
and services it uses.
