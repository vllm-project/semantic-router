---
title: Model and Provider Day-0 Support
description: Add a built-in model or provider once and generate every runtime and product view from the shared catalog.
---

# Model and Provider Day-0 Support

Built-in support is a validated resource graph, not a name added to several
independent lists. The repository catalog under `config/catalog/` generates the
runtime registry, built-in distribution, Dashboard Model Hub and Add Model cards, and the
public [Models page](/models).

## Choose the smallest change

| Change | Catalog resources | Code adapter |
| --- | --- | --- |
| New model on an existing compatible provider | Model Card; one entry in that provider's `models[]`; reasoning/evaluation records when known | No |
| New OpenAI- or Anthropic-compatible provider | One provider file with its built-in `models[]` mappings | No |
| New wire protocol or genuinely different semantics | Protocol and provider definitions, conformance fixtures | Yes, at the protocol/auth/transport seam |
| Self-hosted model used by one operator | None required; write a normal custom model binding and optional Model Card | No |

A provider card is not a claim that every model is built in. A built-in model
has a validated Model Card, and a hosted model is selectable only when a
provider's `models[]` maps it to the provider-native ID. Every active physical
Model Card must therefore have at least one provider mapping. Virtual recipes are different: their
recommended pools may name operator-defined custom models and are not foreign
keys into the built-in catalog.

## Curated catalog admission

The built-in physical catalog is curated by mainstream model creator, not by a
global top-model count. For an existing creator, a Day-0 contribution normally
adds or rotates the catalog toward roughly its latest three generations or
representative product lines. Adding a new creator changes the reviewed
baseline in `manifest.yaml.inventory.physical`; it should explain why the company
belongs in the mainstream set and which current lines form a useful operator
surface. The generator enforces creator membership and minimum depth, while the
human review decides recency and relevance.

Provider support is independent. A new Provider ID may have no built-in model
mapping and still be valuable for handwritten custom models in Add Model. Do
not add an obscure Model Card merely to make a provider card look populated.

## Add a model

1. Add or update a focused family file under
   `config/catalog/resources/models/single/`. Record intrinsic facts only:
   canonical ID, publisher and presentation, distribution source/license,
   revision, limits, modalities, capabilities, protocols, lifecycle, and
   reasoning-family reference. Recipe-backed logical models belong in
   `models/virtual/` instead.
2. Add a mapping under `config/catalog/resources/providers/<provider>.yaml`
   `models[]` when vLLM-SR should know a provider-native model ID, protocol
   restriction, price, or other provider-specific fact. Do not create a second
   provider/model inventory.
3. Reuse a reasoning family. Add a new family only when the request projection
   itself is new; do not duplicate a built-in family in user configuration.
4. Add benchmark records under `evaluations/single/` only when the result is
   attributable to a primary model source and redistributable. Keep each
   benchmark version, exact subject, and raw metric explicit. Virtual-model
   recipe runs use the identical schema under `evaluations/virtual/`. Record
   `measured_at` when the run date is known; otherwise record `observed_at` as
   the date the published value was reviewed, without presenting it as a run
   date.
5. Add conformance fixtures for capabilities or protocol behavior claimed by
   the card/provider mapping.

Model IDs are namespaced, for example `organization/model`. Benchmark and
index IDs use full semantic versions. A changed dataset, grader, prompt
protocol, or aggregation rule requires a new benchmark version.

The default intelligence index uses MMLU-Pro, GPQA Diamond, Humanity's Last
Exam, SWE-bench Verified, and Terminal-Bench 2.1 at equal weight. It emits a
headline score only at 60% coverage. A new model may ship with less evidence;
the Hub then shows the available components and `Not yet measured` rather than
inventing a value. Two available values for one model and versioned metric are
rejected. Vendor-published results retain their exact model variant, reasoning
mode, tool mode, and harness metadata and are labeled claimed; only a frozen
vLLM-SR run with its artifact can be labeled reproduced.

`reasoning_effort` names a runtime control, not an evidence source. If a score
does not disclose the runtime effort, use `unspecified`; keep whether the result
was vendor-published or independently measured in `evidence.provenance` and
`evidence.verification`.

Generation emits exactly those five benchmark slots for every model and every
selectable reasoning effort. A slot without trustworthy evidence is explicit
`missing`, never zero. When a source reports several efforts, author a separate
evaluation record for each effort; results from `high`, `xhigh`, or `max` are
not copied across rows. Extra benchmarks remain visible as evidence without
silently entering this version of the default index.

## Add a provider

1. Give the provider one stable lowercase Provider ID in a focused file under
   `config/catalog/resources/providers/`.
2. Declare only protocols the provider actually transports and the exact
   fully-qualified `supported_operations` subset it implements. Every declared
   protocol must include `#create`; add `#list_models` only when that provider
   surface really exposes compatible discovery. Use `native` for a first-class
   semantic integration, `compatible` for a compatibility API, and `runtime`
   for private serving runtimes such as vLLM or SGLang.
3. Add the default base URL, auth strategy, header/prefix, non-secret default
   request headers, path overrides, API-version behavior, and an existing
   `reasoning_transport` only when the provider changes reasoning-field
   placement. Secrets and credential-bearing default headers are rejected by
   generation.
   Reuse `output_config_effort` when the provider carries the selected level as
   `output_config.effort`. Use `activation_parameter` only when activation is a
   separate boolean control from the model's effort ladder.
   For mixed-thinking Chat APIs, select `top_level_effort_template_switch`
   when the switch belongs in local `chat_template_kwargs`, or
   `top_level_effort_boolean_switch` when both fields are top-level provider
   extensions. Responses keeps the standard `reasoning.effort` shape.
4. Add display name, category, logo source or monogram fallback, and
   conformance status. A missing image must never block configuration.
5. Add code only when generic protocol, URL, and auth handling cannot represent
   the provider. Keep that adapter beside its conformance tests; do not add a
   provider case to a generic config helper.

Provider API operations are defined in
`config/catalog/resources/protocols.yaml`. A provider selects a subset with
`supported_operations` and may override a selected operation path; it cannot
invent or override an undeclared operation. Runtime dispatch, Dashboard model discovery, and connection
verification resolve paths, auth, and non-secret headers from these same
definitions. Provider-specific reasoning placement reuses one of the catalog's
validated transport modes; endpoint hostnames are never used as provider
identity.

### Reasoning wire contract

Operators use one protocol-neutral decision surface:
`use_reasoning`, optional `reasoning_mode`, and optional `reasoning_effort`.
The built-in model family limits the selectable values, and a provider-model
mapping may narrow them further. The selected protocol and catalog transport
then produce exactly one provider-native shape:

| Provider/runtime surface | Final request control |
| --- | --- |
| Local vLLM/SGLang template control | switch and/or effort inside `chat_template_kwargs` |
| Local mixed effort plus template switch | top-level effort plus `chat_template_kwargs.<activation_parameter>` |
| Local template effort flags | activation boolean plus one mutually exclusive effort flag; default/full may be omission |
| OpenAI-compatible Chat effort | top-level `reasoning_effort` |
| OpenAI Responses effort | `reasoning.effort` |
| Provider boolean switch | top-level `<parameter>: true\|false` |
| Gateway-normalized reasoning | `reasoning.effort` or `reasoning.enabled` |
| Thinking-object API | `thinking.type: enabled\|disabled\|adaptive` |
| Thinking object plus effort | `thinking.type` plus top-level `reasoning_effort` |
| DeepSeek Chat | `thinking.type` plus top-level `reasoning_effort` when enabled |
| Anthropic Messages | `thinking.type` plus `output_config.effort` when enabled |

DeepSeek Responses remains a standard Responses request and therefore uses
`reasoning.effort`, rather than the Chat-specific two-field shape. Provider
adaptation removes competing `reasoning`, `thinking`, top-level effort,
`output_config.effort`, and local template controls before writing the selected
shape. A Day-0 contribution that changes reasoning semantics must add a final
encoded-request fixture for every claimed protocol, including enabled,
disabled, adaptive, and effort variants that the model actually exposes.

For a custom model whose template uses boolean effort flags, keep the public
decision on `reasoning_effort` and map only the wire names in its inline
`reasoning.effort_flags`. One active level may be intentionally unmapped to
mean “all effort flags omitted”; every other active level must have a unique
flag. Built-in models already carry this mapping and require no user YAML.

Dashboard discovery applies a stricter network boundary than ordinary runtime
dispatch. A cloud Model API must use the exact built-in origin. A self-hosted
runtime may intentionally use a private or loopback address, but link-local and
metadata endpoints are rejected. Redirects and environment proxies are disabled,
and resolved addresses are checked again by the dialer before credentials are
sent.

## Worked example: GPT-6 Astra

GPT-6 Astra is an existing-provider Day-0 change, so its implementation stays
inside the shared catalog and the existing OpenAI protocol adapters. Its source
packet is the official [model reference](https://developers.openai.com/api/docs/models/gpt-6-astra),
[latest-model guide](https://developers.openai.com/api/docs/guides/latest-model),
and [launch evaluation report](https://openai.com/index/gpt-6-astra/):

| Contract | Source of truth |
| --- | --- |
| Identity, limits, modalities, capabilities, and presentation | `config/catalog/resources/models/single/openai.yaml` |
| OpenAI model ID, Chat/Responses availability, pricing, and API constraints | `config/catalog/resources/providers/openai.yaml` |
| Always-on `low`, `medium`, `high`, `xhigh`, and `max` reasoning | `config/catalog/resources/reasoning-families.yaml` |
| Exact vendor-published benchmark results | `config/catalog/resources/evaluations/single/openai.yaml` |
| Final Chat and Responses request shapes | `src/semantic-router/pkg/extproc/provider_request_catalog_contract_test.go` |
| Runnable aliases and mock-provider credentials | `e2e/profiles/response-api/values.yaml` |
| Default CI profile membership | `e2e/profiles/response-api/profile.go` |
| Black-box model-ID and reasoning projection | `e2e/testcases/model_catalog_astra_day0.go` |

The official model page does not name a default reasoning effort, so the
catalog does not invent one. It also does not expose a disabled mode: omitting
an effort lets the API choose its default, while `use_reasoning: false` is
rejected during configuration validation. Published launch scores are marked
`unspecified` because the source reports the maximum result across supported
efforts rather than attributing each score to one effort.

The OpenAI binding records that tools require Responses; requests beyond
272,000 input tokens use the published long-context multipliers; and
`temperature`, `top_p`, and `top_logprobs` are unsupported (`logprobs` is also
unsupported for Chat Completions, and Responses cannot include
`message.output_text.logprobs`). It also narrows Chat Completions to `low`,
`medium`, `high`, and `xhigh`, because `max` is Responses-only. These are
discoverable catalog constraints,
not silent request rewriting: the Router projects protocol and reasoning
fields exactly, while OpenAI remains authoritative for rejecting unsupported
request fields.

Use Responses for Astra tool calls. A minimal routed configuration needs no
handwritten Model Card or reasoning family:

```yaml
version: v0.3
providers:
  models:
    - name: astra
      catalog: openai/gpt-6-astra
      api_format: responses
      backend_refs:
        - name: primary
          provider: openai
          api_key_env: OPENAI_API_KEY

routing:
  decisions:
    - name: astra_default
      priority: 1
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: astra
          use_reasoning: true
          reasoning_effort: high
```

For a plain Chat Completions workload without tools, omit
`api_format: responses`; the same catalog binding emits top-level
`reasoning_effort` and rejects a Responses-only effort during startup.
Responses emits `reasoning.effort`. The black-box E2E also
proves that a Responses tool definition survives the Router-to-provider hop.
The generated Router, CLI, Dashboard, and website views all come from these
authored resources.

Run the worked example by name:

```bash
make e2e-test-specific \
  E2E_PROFILE=response-api \
  E2E_TESTS=model-catalog-astra-day0
```

The profile uses a fixture credential and local mock provider; it never calls
the external model API. The test sends Chat with `xhigh`, Responses with
Responses-only `max`, and a Responses `high` tool request. It then reads the
mock provider's captured request and checks the provider-native model ID,
protocol-specific reasoning field, absence of the competing protocol shape,
and tool preservation. Keep this final-hop assertion when copying the example
for another model whose protocol or reasoning contract differs.

## Built-in and custom user configuration

A built-in card is selected with one optional `catalog` reference. The model
`name` remains the request-facing alias, while `backend_refs[].provider` is the
catalog Provider ID:

```yaml
version: v0.3
providers:
  defaults:
    model: production
    reasoning_effort: medium
  models:
    - name: production
      catalog: organization/model
      backend_refs:
        - name: primary
          provider: provider-id
          api_key_env: PROVIDER_API_KEY
```

Built-in cards do not require `routing.modelCards`. An intentional override
uses the canonical catalog identity as its card name:

```yaml
routing:
  modelCards:
    - name: organization/model
      tags: [production, approved]
```

`api_format` only selects the upstream wire contract; never use it to infer the
Provider from the current registry contents. A physical model behind a
Router-owned listener must have an explicit `backend_refs[].provider` so a
future Provider addition cannot change the meaning of existing YAML.

When a provider binding declares an operator-defined `deployment_name`, its
catalog ID only records availability; the user must set
`providers.models[].provider_model_id` (or that provider's
`external_model_ids` entry) explicitly.

Treat multiple `backend_refs` on one alias as homogeneous replicas in one
Envoy pool. HTTP endpoint address, port, and weight may differ; HTTPS replicas
may vary by port and weight but keep one DNS hostname. Provider ID,
protocol/model mapping, credential source, auth and default headers, effective
request path, and DNS/TLS behavior must otherwise match. Split heterogeneous providers or
credentials into separate aliases so Router request shaping cannot diverge
from the endpoint Envoy selects. The config loader and CLI generator reject an
unsafe mixed pool with the differing semantic fields named and never include
credential values in the error.

Built-in reasoning comes from that card. A `providers.models[].reasoning`
block is accepted only when `catalog` is omitted for a custom model.

Custom cards may also declare optional `publisher`, `presentation`, and
`distribution` metadata. That lets a private or newly released model render as
a complete effective card without adding a repository resource; none of these
fields stores credentials.

For a private model, omit `catalog`. Its alias is its local card identity, and
both reasoning and evaluations remain optional:

```yaml
providers:
  defaults:
    model: private-reasoner
  models:
    - name: private-reasoner
      provider_model_id: private-awq
      reasoning:
        family: qwen3
      backend_refs:
        - name: lab
          provider: vllm
          endpoint: model-gateway.example:8000/v1

routing:
  modelCards:
    - name: private-reasoner
      context_window_size: 131072
      capabilities: [chat, tools, reasoning]
      evaluations:
        - benchmark: organization/private-eval@1.0.0
          benchmark_profile: published-standard
          reasoning_effort: high
          metrics:
            pass_rate: 0.82
```

User-authored evaluations intentionally have a small surface: `benchmark` and
`metrics`, plus optional `benchmark_profile`, `reasoning_effort`, `source`,
`measured_at`, and scalar `metadata`. Omit `benchmark_profile` to use a known
benchmark's default profile; set `reasoning_effort` when the measurement came
from a specific model effort. Catalog records retain richer evidence and
provenance internally. Custom benchmark identities remain namespaced and
versioned, while metric values must be finite.

## Generate and validate

```bash
make model-catalog-generate
make model-catalog-check
make impact ENV=cpu BASE_REF=upstream/main
make check BASE_REF=upstream/main
make verify PROFILE=response-api
```

Commit the authored resources, built-in distribution snapshot, Router embed,
and shared public snapshot together. CLI package assets are build-time staging
outputs, are ignored by Git, and must not be committed. Use the E2E profile
reported by `impact`
instead of `response-api` when the model or provider belongs to another
profile; use `make verify DOMAIN=<domain>` for an explicit integration domain.
Run `make ci-full BASE_REF=upstream/main` when a complete local PR baseline is
required. A complete Day-0 pull request demonstrates:

- stable identities and valid references;
- protocol and capability conformance for every support claim;
- generated Dashboard provider/model cards and logo fallback;
- generated website support and benchmark-comparison rows;
- no secrets or restricted benchmark data;
- absent score data rather than a fabricated zero or placeholder row.

Do not hand-edit generated JSON, Go, or built-in catalog snapshots. If a
generated view is wrong, fix the source resource or generator and regenerate
it.
