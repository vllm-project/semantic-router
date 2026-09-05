---
title: Model and Provider Day-0 Support
description: Add a built-in model or provider once and generate the Router, CLI, Dashboard, website, and validation views from the shared catalog.
---

# Model and Provider Day-0 Support

Built-in support is a validated resource graph, not a name added to several
independent lists. The repository catalog under `config/catalog/` generates the
runtime registry, CLI bundle, Dashboard Model Hub and Add Model cards, and the
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
   recipe runs use the identical schema under `evaluations/virtual/`.
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
          metrics:
            pass_rate: 0.82
```

User-authored evaluations intentionally have a small surface: `benchmark` and
`metrics`, plus optional `source`, `measured_at`, and scalar `metadata`. Catalog
records retain richer evidence and provenance internally. Custom benchmark
identities remain namespaced and versioned, while metric values must be finite.

## Generate and validate

```bash
make model-catalog-generate
make model-catalog-check
make agent-report ENV=cpu CHANGED_FILES="config/catalog/resources/models/organization.yaml"
```

Commit the authored resources and every generated projection together. Then
run the gates reported for the actual changed files. A complete Day-0 pull
request demonstrates:

- stable identities and valid references;
- protocol and capability conformance for every support claim;
- generated Dashboard provider/model cards and logo fallback;
- generated website support and leaderboard rows;
- no secrets or restricted benchmark data;
- explicit missing evaluation status rather than a fabricated score.

Do not hand-edit generated JSON, Go, or CLI catalog snapshots. If a generated
view is wrong, fix the source resource or generator and regenerate it.
