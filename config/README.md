# Configuration assets

Use this directory to find a complete configuration, copy a focused fragment,
or start from a maintained routing recipe.

| Need | Start here |
| --- | --- |
| See every supported field | `config/config.yaml`, the exhaustive canonical reference config |
| Add one routing capability | `config/fragments/` |
| Run a complete use case | `config/recipes/` |
| Serve a packaged virtual model | `config/recipes/built-in/` |
| Configure a storage or service backend | `config/runtime/` |
| Validate a managed asset | `config/schemas/` |
| Discover the compact machine-readable contract index | `vllm-sr config schema` or `GET /api/v1/config/schema` |

The website's [configuration guide](../website/docs/installation/configuration.md)
is the reader-facing reference. `config/config.yaml` is intentionally exhaustive;
trim it for a deployment instead of treating every optional service as required.

## Canonical shape

A router config uses these top-level sections:

```yaml
version: v0.5
listeners: []
providers: {}
evaluation: {}
entrypoints: []
recipes: [{name: default}]
global: {}
```

- `listeners` exposes inference and management endpoints.
- `providers.defaults` defines shared provider behavior;
  `providers.models[]` binds model names to concrete backends and owns their
  deployment pricing metadata. A built-in model may add an optional `catalog`
  identity, while `backend_refs[].provider` selects the stable runtime Provider
  ID. Custom vLLM/SGLang models continue to omit `catalog`. The facts routing
  reads about a model (`capabilities`, `modalities`, context window, LoRA
  adapters) sit on the same entry.
- `evaluation` optionally defines benchmarks, indices, and model-linked
  measurement records.
- `recipes` are isolated routing profiles. Each recipe has its own signals,
  projections, decisions, strategy, algorithms, and plugins while sharing
  providers and router-wide services. One recipe needs no entrypoints.
- `entrypoints` maps request-facing model names to recipes; with several
  recipes, exactly one chat entrypoint is marked `default: true`. See
  [`tutorials/global/entrypoints-and-recipes.md`](../website/docs/tutorials/global/entrypoints-and-recipes.md).
- `global` owns cross-cutting router settings, services, stores, integrations,
  router-managed model assets, and the default plugins (`global.plugins`) that
  recipes and decisions override field by field.

Validate a file before serving it:

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
```

[Choose a deployment](../website/docs/system-one/deploy/deployments.md) explains the
built-in decision model, the other sizes and remote System One APIs, and what
the Router checks at load.

`src/semantic-router/pkg/configschema/router-config-v0.5.schema.json` is the one
checked-in schema generated from the Go configuration types and routing
registries. Do not edit it directly. See the
[Configuration Contract](../website/docs/installation/configuration-contract.md)
for schema discovery, semantic validation, and the extension workflow.

## Choose the right asset

### Fragments

Fragments show one capability in its owning section. They are not complete
deployments and may rely on model or service definitions from a base config.

- `config/fragments/signal/`: request and response facts used by decisions.
  The guide of the `systemone` primitive is `tutorials/signal/systemone.md`;
  application and heuristic signal guides live under
  `tutorials/signal/application/` and `tutorials/signal/heuristic/`.
- `config/fragments/decision/`: `single`, `and`, `or`, `not`, and nested
  boolean rule shapes. Rules on model-judged signals may resolve a terminal
  `Unknown` result with `on_unknown: no_match|match|fail_request`.
- `config/fragments/algorithm/`: per-decision model selection and bounded
  multi-model execution policies.
- `config/fragments/plugin/`: route-local request or response processing such
  as caching, memory, RAG, tool policy, and safety handling.

The corresponding website sections are
[`tutorials/signal/`](../website/docs/tutorials/signal/),
[`tutorials/decision/`](../website/docs/tutorials/decision/),
[`tutorials/algorithm/`](../website/docs/tutorials/algorithm/),
[`tutorials/plugin/`](../website/docs/tutorials/plugin/), and
[`tutorials/global/`](../website/docs/tutorials/global/).

### Recipes and built-in models

`config/recipes/` contains complete, runnable examples. Read a recipe's Model
Card before using it; the card explains its intended use, backend roles, data
handling, evaluation scope, and limitations.

`config/recipes/built-in/` is the versioned source for virtual models bundled
with the distribution. Dashboard presents these under **Build →
Mixture-of-Models → Recipes** and lets operators assign connected Models
without editing provider credentials into a Recipe.

### Runtime examples

`config/runtime/` contains backend-specific support files for memory, response
cache, the Response API, tools, and vector stores. These files configure a
runtime dependency; they do not define routing behavior by themselves.

## Important boundaries

- Model backend credentials belong in environment references, not literal YAML
  values.
- Catalog-backed models materialize their built-in Model Card, reasoning family,
  provider protocol, path, and non-secret defaults automatically. A handwritten
  override uses the canonical `catalog` identity as `providers.models[].name`;
  a fully custom model uses its request alias as the card name.
- `api_format` selects the upstream wire format; it never selects a Provider.
  When this config declares a listener, every physical model must use
  `backend_refs` with an explicit Provider ID. Metadata-only external-gateway
  configs (`listeners: []`) and built-in virtual models may remain backendless.
  The local `vllm-sr serve` path manages an Envoy listener, so it rejects a
  backendless physical model even when the authored listener list is empty;
  deploy state-only metadata through the external-gateway integration instead.
- Multiple `backend_refs` on one alias are homogeneous replicas. HTTP targets
  may vary by host, port, and weight. HTTPS targets may vary by port and weight
  but must keep one DNS hostname. Provider ID, wire protocol, native model ID,
  credentials, headers, request path, and TLS semantics must also match; use
  separate aliases for heterogeneous providers.
- `providers.models` describes semantic capabilities; concrete URLs,
  credentials, and pricing belong in `providers.models`.
- Protocol controls such as `tool_choice` enter routing as conversation facts;
  projections combine those facts with text-derived intent before decisions
  apply policy.
- `recipes[].projections` derives named routing outputs from signals. Decisions
  consume those outputs instead of embedding free-form computation.
- Candidate iteration is bounded policy metadata, not a general scripting
  runtime.
- `recipes[].decisions[].algorithm.minimum_candidates` keeps a model-free Recipe
  portable while making its candidate-pool cardinality executable as soon as
  an Entrypoint assigns Models. Request-time context filtering cannot silently
  reduce the eligible pool below that contract.
- A PII rule reads the current user turn. `input.history: true` adds the
  earlier user turns, `input.tool_results: true` the request's textual
  tool results, and `input.request: false` leaves the current turn out.
  A tool result the Router cannot read as text, such as an image or an
  unsupported block, counts as unread input, which `on_unscanned` decides.
- The decision model answers the application signals (domain, PII, jailbreak,
  complexity and the others) as System One questions. A rule's `deployment`, a
  recipe's `model_bindings` or `global.model_catalog.bindings` names another
  deployment from `global.model_catalog.deployments`, such as a remote System
  One API.
- A complexity rule answers `easy`, `medium` or `hard`, and a decision
  condition names the level it routes on with `label`.
- A PII rule needs a deployment that answers span questions, or the
  configuration fails to load with `span_required`. When the rule's model gives
  no answer, its `on_unknown` decides: `no_match` (the default) or `match`.
  Text the model could not read counts as personal data.
- A remote deployment caps one upstream response with
  `limits.max_response_bytes`.
- Router Learning lives under `global.router.learning`; it is separate from a
  decision's request-time base algorithm.
- Router replay is disabled by default and can capture request or response
  bodies. Review its access controls and retention settings before enabling it.
- `global.router.skip_processing.enabled` should be enabled only when an
  authenticated upstream component owns the bypass header.

## Keep examples in sync

When a public config field or supported routing surface changes, update its Go
type or registry, then update its fragment, exhaustive reference, affected
recipes, and the matching website page together. Regenerate the
machine-readable contract before running the semantic and repository gates:

The focused semantic gate is `go test ./pkg/config/...`; `make check` applies
the complete changed-surface policy.

```bash
make config-schema-generate
make config-schema-check
go test ./pkg/config/...
make check
```

Complete routing scenarios belong in `config/recipes/`; backend support files
belong in `config/runtime/`; local Envoy and test-only manifests belong under
`deploy/` and `e2e/`, respectively.
