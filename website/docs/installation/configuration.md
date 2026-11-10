---
title: Configuration
description: Understand the canonical v0.5 YAML document and where providers, recipes, entrypoints, services, and secrets belong.
---

# Configuration

Semantic Router uses one canonical YAML document across the CLI, Dashboard,
Helm, and Operator. The top-level structure is:

```yaml
version:
listeners:
providers:
evaluation:
recipes:
entrypoints:
global:
```

- **Start small.** Most deployments begin with `version`, `listeners`,
  `providers` and one recipe.
- **One recipe needs no entrypoints.** It serves `vllm-sr/auto`. Several
  recipes need `entrypoints`, with exactly one chat entrypoint marked
  `default: true`.
- **Add `global` only when needed:** for shared services, default plugins, or
  runtime behavior that differs from the built-in defaults.

## What belongs where

| Section | Owns |
| --- | --- |
| `version` | Canonical schema version. Use `v0.5`. |
| `listeners` | Public Router listeners: address, port, timeouts, client API keys, model allowlists, TLS and trusted identity. See [Listeners](#listeners). |
| `providers` | Logical provider models: their backend endpoints, pricing, defaults, and the facts routing reads about each model (`capabilities`, `modalities`, context window, LoRA adapters). |
| `evaluation` | Optional operator-owned benchmark definitions, versioned index DAGs, and model-linked records. |
| `recipes` | Isolated routing profiles: each recipe's signals, projections, decisions, strategy, fallback, candidate requirements, model bindings, and default plugins. |
| `entrypoints` | Public virtual model names mapped to a recipe; with several recipes, exactly one chat entrypoint is the default. |
| `global` | Router services, stores, integrations, observability, learning, the default plugins every request runs (`global.plugins`), and the models the Router asks (`global.model_catalog`). |

Keep these boundaries clear:

- signals detect facts;
- projections combine evidence;
- decisions define eligibility and route policy;
- algorithms choose or coordinate candidate models;
- plugins add behavior at route-specific hook points; and
- providers bind logical model names to inference endpoints.

The [Routing Pipeline](../overview/signal-driven-decisions) explains the design,
and the pages under **Capabilities** document each signal, projection,
decision, algorithm, plugin and global block.

### Listeners

| Field | Meaning | Default |
| --- | --- | --- |
| `timeout` | The whole call, including its streamed response; it also bounds a stalled stream. `0s` lifts both limits. To lift only one model's call limit, set that model's `total_timeout: 0s`. | `300s` |
| `api_keys` | Client API keys the listener requires. | none |
| `models` | The chat models the listener accepts. | empty: all |
| `systemone.models` | The native System One models the listener publishes, separate from `models`. Standalone mode serves them; `--gateway extproc` refuses the field. | none |
| `systemone.max_deadline_ms` | The longest a System One request waits for a model. | 10 minutes |
| `tls.cert_file`, `tls.key_file` | One-way TLS. | off |
| `identity.trust_headers`, `identity.trusted_peers` | The identity sources the listener trusts, in standalone mode. | none |

### Providers and model facts

- **Facts.** The facts routing reads about a model sit on its
  `providers.models[]` entry.
- **Pricing.** `providers.models[].pricing` takes an optional uppercase
  three-letter `currency`, plus non-negative `prompt_per_1m`,
  `completion_per_1m`, `cached_input_per_1m` and `cache_write_per_1m` rates.
- **Backend format.** [Protocol Compatibility](protocol-compatibility) helps
  you choose a backend's `api_format`. Check
  [Backend Target Compatibility](backend-target-compatibility) before moving
  bindings between Docker, Helm, the Operator and the Dashboard.

### The Router's own models

The decision model that answers signals and the embedding model that recall
uses are deployments under `global.model_catalog.deployments`. They run in the
[model runtime](../model-runtime/get-started/overview.md), or point to a remote API.

- **One default.** The decision model answers every model-judged signal unless
  something names another deployment.
- **Overrides, most specific first:** a rule's own `deployment`, then
  `recipes[].model_bindings.<type>`, then `global.model_catalog.bindings.<type>`.
- **Limits.** A remote deployment's `limits.max_request_bytes` and
  `limits.max_response_bytes` cap one exchange with it.

[Choose a model for a signal](choose-a-model-for-a-signal.md) explains the
rule, and the [model catalog reference](model-catalog.md) lists every
deployment field.

### Evaluation records

Measurements belong in `evaluation.records[]`, and reference a canonical
Model Card identity through `model`. Built-in benchmark IDs work directly;
define new benchmarks and indices beside the records. See
[Custom evaluations](../benchmarking/custom-evaluations).

:::note Debug surfaces stay closed
`global.services.observability.profiling` serves Go `pprof` endpoints only
when enabled, on `127.0.0.1:6060` unless you change `bind`. The Router reads
it once at startup, so a change needs a restart. See
[API and Observability](../tutorials/global/api-and-observability).
:::

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
| `authz` — heuristic signal | `authz` turns identity and policy bindings into reusable routing inputs under `recipes[].signals.role_bindings`. | [`config/fragments/signal/authz/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/authz/) | [Guide](../tutorials/signal/heuristic/authz) |
| `complexity` — application signal | `complexity` rates how difficult a request is to answer well: easy, medium or hard. | [`config/fragments/signal/complexity/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/complexity/) | [Guide](../tutorials/signal/application/complexity) |
| `context` — heuristic signal | `context` detects requests that need a larger effective context window. | [`config/fragments/signal/context/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/context/) | [Guide](../tutorials/signal/heuristic/context) |
| `conversation` — heuristic signal | `conversation` routes on chat structure and protocol facts, such as message count, developer instructions, available tools, explicit tool-use constraints, or an active tool loop. | [`config/fragments/signal/conversation/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/conversation/) | [Guide](../tutorials/signal/heuristic/conversation) |
| `domain` — application signal | `domains` routes on the request's subject area, one of 14 that the decision model chooses. | [`config/fragments/signal/domain/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/domain/) | [Guide](../tutorials/signal/application/domain) |
| `fact-check` — application signal | `fact_check` detects requests whose answer relies on factual claims that should be verified. | [`config/fragments/signal/fact-check/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/fact-check/) | [Guide](../tutorials/signal/application/fact-check) |
| `hallucination` — application signal | `hallucination` finds the claims in the model's answer that its grounding context does not support. | [`config/fragments/signal/hallucination/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/hallucination/) | [Guide](../tutorials/signal/application/hallucination) |
| `input-modality` — heuristic signal | `input_modality` deterministically matches which kinds of input — `text`, `image`, `audio`, or `video` — are present in the parsed request. | [`config/fragments/signal/input-modality/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/input-modality/) | [Guide](../tutorials/signal/heuristic/input-modality) |
| `jailbreak` — application signal | `jailbreak` detects prompt-injection and jailbreak attempts, in the request before routing or in the model's answer. | [`config/fragments/signal/jailbreak/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/jailbreak/) | [Guide](../tutorials/signal/application/jailbreak) |
| `keyword` — heuristic signal | `keyword` matches explicit words and phrases in the request. | [`config/fragments/signal/keyword/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/keyword/) | [Guide](../tutorials/signal/heuristic/keyword) |
| `language` — heuristic signal | `language` detects the request language and exposes it as a routing signal. | [`config/fragments/signal/language/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/language/) | [Guide](../tutorials/signal/heuristic/language) |
| `metadata` — heuristic signal | `metadata` matches bounded string values supplied by the caller in request metadata. | [`config/fragments/signal/metadata/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/metadata/) | [Guide](../tutorials/signal/heuristic/metadata) |
| `modality` — application signal | `modality` reads which output the request asks for: text, an image, text with an image, speech, audio or video. | [`config/fragments/signal/modality/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/modality/) | [Guide](../tutorials/signal/application/modality) |
| `pii` — application signal | `pii` locates personal data in the request, with the exact characters of every span. | [`config/fragments/signal/pii/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/pii/) | [Guide](../tutorials/signal/application/pii) |
| `refusal` — application signal | `refusal` asks the decision model whether the model's answer declines the request. | [`config/fragments/signal/refusal/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/refusal/) | [Guide](../tutorials/signal/application/refusal) |
| `safety` — application signal | `safety` detects harmful content in the request or the model's answer, optionally in named harm categories. | [`config/fragments/signal/safety/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/safety/) | [Guide](../tutorials/signal/application/safety) |
| `systemone` — primitive signal | `systemone` asks the decision model a typed question you write, and turns the answer into a routing fact. | [`config/fragments/signal/systemone/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/systemone/) | [Guide](../tutorials/signal/systemone) |
| `turn-relation` — application signal | `turn_relation` asks the decision model how the current user turn relates to the conversation before it. | [`config/fragments/signal/turn-relation/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/turn-relation/) | [Guide](../tutorials/signal/application/turn-relation) |
| `user-feedback` — application signal | `user_feedbacks` reads what the current user turn says about the previous answer. | [`config/fragments/signal/user-feedback/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/user-feedback/) | [Guide](../tutorials/signal/application/user-feedback) |

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
| `context` — route plugin | `context` is the plugin that assembles the request's context. | [`config/fragments/plugin/context/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/context/) | [Guide](../tutorials/plugin/context) |
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
version: v0.5
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
      capabilities: [chat]
recipes:
  - name: default
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
omits `catalog`, and can define its own Model Card and reasoning contract
under that alias.

- [Configure models](model-configuration) is the starting point.
- [Model configuration patterns](model-configuration-patterns) compares the
  catalog, custom, reasoning, provider and replica combinations.
- The [Model and provider Day-0 guide](../community/model-provider-day-0-support.md)
  is for contributors adding support to the repository catalog.

:::tip When a signal fails
By default, a failed or late signal's condition does not match. Set
`rules.on_unknown` on a decision to decide otherwise; see
[Decisions](../tutorials/decision/overview).
:::

## Validate and serve

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
```

Validation catches schema errors, unresolved references, incompatible recipe
boundaries, invalid provider bindings, and unsupported plugin or algorithm
settings before the Router starts.

:::tip Portable recipes
For a recipe shared without models, set
`recipes[].decisions[].algorithm.minimum_candidates` to the smallest pool the
decision needs. The empty recipe stays valid; publishing an entrypoint fails
if its assigned models fall short.
:::

## Environment references and secrets

Keep credentials outside the YAML file:

```yaml
api_key: ${MODEL_API_KEY}
```

| Form | Expands to |
| --- | --- |
| `${VAR}`, `$VAR` | The value of `VAR` |
| `${VAR:-default}` | `default` when `VAR` is unset or empty |
| `${VAR-default}` | `default` when `VAR` is unset |
| `$$` | A literal `$` |

- A custom recipe must authorize the host variables it needs with
  `--recipe-env NAME`.
- On Kubernetes, put sensitive values in Secrets, not ConfigMaps or Helm
  values. See [Security Hardening](security-hardening).

## Entrypoints and recipes

- **One recipe:** no entrypoints are needed. The recipe is the default, and is
  published as `vllm-sr/auto`.
- **Several recipes:** `entrypoints` maps their public model names, and exactly
  one chat entrypoint is marked `default: true`. `vllm-sr/auto` is published
  only when an entrypoint names it.

Names have no implicit meaning: a recipe called `MoM` is just a name.

```yaml
recipes:
  - name: balanced
    decisions: []
  - name: private
    decisions: []
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: balanced
    default: true
  - model_names: [acme/private]
    recipe: private
```

| Field | Meaning |
| --- | --- |
| `entrypoints[].model_names` | The public model aliases. |
| `entrypoints[].recipe` | The recipe those aliases enter. |
| `entrypoints[].default` | Marks the default chat entrypoint. |
| `recipes[]` | One recipe's policy: its signals, projections, decisions, algorithms, plugins, cache, learning and routing state. |

Providers, stores and the model catalog's deployments can be shared without
letting policy state cross recipe boundaries. A concrete provider model name
skips recipes: it bypasses signals, decisions, plugins, cache, learning and
session routing.

## Defaults and overrides

Defaults live in `global`; a recipe overrides them, and a decision overrides
its recipe:

| Setting | Default in | Overridden by |
| --- | --- | --- |
| Strategy and fallback | `global.router.strategy`, `global.router.fallback` | Each recipe's `strategy` and `fallback`; a decision's fallback |
| Plugins | `global.plugins` | A recipe's `plugins`, then a decision's `plugins` |
| Deployments that answer signals | `global.model_catalog.bindings` | A recipe's `model_bindings`, then a rule's `deployment` |

- **No inheritance between recipes.** Each recipe resolves its own settings
  against `global`.
- **Sparse overrides.** `fallback: {enabled: false}` keeps the other fallback
  defaults. The runtime strategy default is `priority`.
- **Plugins merge by type, field by field.** Mappings merge key by key; a list
  or scalar replaces the value below it.
- **Unmatched requests.** A request that matches no decision runs its recipe's
  layered plugins; a request for a concrete model runs `global.plugins` but
  skips what needs a recipe: the response cache, the context plugin's
  retrieval, `response_guard` and masking.
- **Model facts.** `capabilities`, `modalities`, `context_window_size`,
  `max_output_tokens` and `loras` sit on the model's `providers.models[]`
  entry.

If no decision matches, the recipe uses `providers.defaults.model`. The
virtual entrypoint name never reaches a backend.

- [Models, Entrypoints, and Serving](../tutorials/global/models-entrypoints-serving)
  covers built-in virtual models, CLI serving, backend binding, forking,
  packaging and migration.
- [Virtual Models](../tutorials/global/entrypoints-and-recipes) has the
  complete schema.

### Recipe candidate requirements

Set optional candidate requirements on a recipe:

```yaml
candidate_requirements:
  capabilities: declared
  context: known_limits
```

| Requirement | A candidate is eligible when |
| --- | --- |
| `capabilities: declared` | It declares support for the request's task, including tools and image input, and a compatible provider protocol. |
| `context: known_limits` | The estimated input plus the output reserve fits its declared limits. A request is refused only when it doesn't fit even without the estimate's reserves for structured data and media. |

Omit a field to keep that dimension's default compatibility behavior.

- **An output bound is required.** The request must set one, or its decision
  must set `request_params.default_max_tokens` to a positive integer or `auto`.
  That default applies only when the caller omits a bound, and
  `max_tokens_limit` still caps it.
- **Missing facts make a candidate ineligible**, and so does a missing output
  bound. A model's maximum output is never used as a request default.
- **Input is estimated**, especially for multimodal content. This is not an
  exact provider token-capacity guarantee.

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

With `auto`, the Router asks each candidate's backend to render the complete
provider request, without generating an answer, and repeats the check after
the final request changes. It then allocates
`min(max_output_tokens, context_window - input_tokens)` from the real input
length.

:::warning `auto` needs a render-capable vLLM backend

- an explicit `vllm` provider, one backend per model, the OpenAI Chat format,
  and declared context and output limits;
- the backend's `/v1/chat/completions/render` API, enabled with
  `vllm serve --enable-scale-out` on a vLLM version that supports it;
- text and tool requests only: not multimodal input, provider truncation, LoRA,
  Looper or shadow dispatch.

A backend limit smaller than the declared one is a configuration error. An
unavailable or invalid render API fails clearly.
:::

- **Caller limits win.** An explicit caller limit keeps its behavior and needs
  no render call; `max_tokens_limit` still caps the budget.
- **`expected_output_tokens` is a cost forecast, not a limit.** `multi_factor`
  needs it with `auto`, bounded by the caller's limit or each candidate's
  capacity. Without it, other requests keep their usual cost calculation.
- **Overflow.** A request that fits is never truncated by a length estimate.
  If every compatible candidate overflows, the Router applies the configured
  context compression once and checks the result with the backend; without a
  compression policy, overflow stays an error.
- **Preview.** It reports `execution_required`, because it doesn't have the
  complete provider request.

For `multi_factor`, `latency_metric: ttft` compares time to first token and
`tpot` time per output token; see [Multi Factor](../tutorials/algorithm/selection/multi-factor).

Discover the current contract with
`vllm-sr config schema --section recipes.candidate_requirements`. DSL
`ROUTING` blocks support the same object, and the Kubernetes CRD keeps it for
every recipe.

### Replay capture defaults and decision overrides

| Where | What it sets |
| --- | --- |
| `global.services.router_replay` | Shared storage and retention. |
| A `router_replay` plugin in `global.plugins` | Whether capture is on, and the capture defaults. |
| A recipe's or decision's `router_replay` plugin | Only the fields it sets. `enabled: false` turns capture off there; `enabled: true` opts in when the default is off. |

- **Personal data.** `capture_personal_data: false` keeps routing evidence but
  omits content when PII is detected or its status is unknown. Without a PII
  detector, content is suppressed rather than blocking startup.
- **Tool results** stay redacted unless `capture_tool_results: true`, which
  needs `capture_personal_data: true`.
- **Rejected requests** without a selected decision use their recipe's
  capture defaults.

These controls don't govern other stores or a backend's retention. See
[Router Replay](../tutorials/plugin/router-replay) for every field, or run
`vllm-sr config schema --section global.services.router_replay`.

## Configuration workflows

The canonical document can be authored or applied through the local CLI and
YAML, the Dashboard, Helm or `vllm-sr serve --target k8s`, the Kubernetes
Operator, or the routing DSL.

| Page | Covers |
| --- | --- |
| [Configuration Workflows](configuration-workflows) | Which interface owns which part of the document, and how to avoid competing sources of truth. |
| [Configuration Contract](configuration-contract) | The generated schema, the Router's discovery and validation APIs, and the safe authoring loop for tools and agents. |
| [Configuration Management](configuration-management) | How a change activates on a running Router, how a rejection is reported, and how to roll back. |

## Reference sources

- [`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml)
  is the exhaustive canonical example.
- [`config/fragments/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments)
  contains reusable signal, decision, algorithm, and plugin fragments.
- [Providers and routing tutorials](./global-settings) describe
  shared runtime configuration.
- [Unified Config Contract v0.3](../proposals/unified-config-contract-v0-3)
  records the design the v0.5 structure builds on.
- [Configuration Contract](configuration-contract) is the live discovery and
  validation contract for the current Router build.

Avoid copying the exhaustive example as an application config. Start with the
smallest document that describes the deployment, then add only the capabilities
and services it uses.
