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

The website's [configuration guide](../website/docs/installation/configuration.md)
is the reader-facing reference. `config/config.yaml` is intentionally exhaustive;
trim it for a deployment instead of treating every optional service as required.

## Canonical shape

A router config uses these top-level sections:

```yaml
version: v0.3
listeners: []
providers: {}
routing: {}
entrypoints: []
recipes: []
global: {}
```

- `listeners` exposes inference and management endpoints.
- `providers.defaults` defines shared provider behavior;
  `providers.models[]` binds model names to concrete backends and owns their
  deployment pricing metadata.
- `routing` owns model cards, signals, projections, decisions, and the routing
  strategy for the default profile.
- `entrypoints` maps request-facing model names to isolated `recipes`. Each
  recipe has its own signals, decisions, algorithms, and plugins while sharing
  providers and router-wide services. See
  [`tutorials/global/entrypoints-and-recipes.md`](../website/docs/tutorials/global/entrypoints-and-recipes.md).
- `global` owns cross-cutting router settings, services, stores, integrations,
  and router-managed model assets.

Validate a file before serving it:

```bash
vllm-sr validate --config config.yaml
vllm-sr serve --config config.yaml
```

## Choose the right asset

### Fragments

Fragments show one capability in its owning section. They are not complete
deployments and may rely on model or service definitions from a base config.

- `config/fragments/signal/`: request and response facts used by decisions.
  Heuristic and learned signal guides live under
  `tutorials/signal/heuristic/` and `tutorials/signal/learned/`.
- `config/fragments/decision/`: `single`, `and`, `or`, `not`, and nested
  boolean rule shapes.
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
- `routing.modelCards` describes semantic capabilities; concrete URLs,
  credentials, and pricing belong in `providers.models`.
- `routing.projections` derives named routing outputs from signals. Decisions
  consume those outputs instead of embedding free-form computation.
- Candidate iteration is bounded policy metadata, not a general scripting
  runtime.
- Router Learning lives under `global.router.learning`; it is separate from a
  decision's request-time base algorithm.
- Router replay is disabled by default and can capture request or response
  bodies. Review its access controls and retention settings before enabling it.
- `global.router.skip_processing.enabled` should be enabled only when an
  authenticated upstream component owns the bypass header.
- Knowledge bases are declared under `global.model_catalog.kbs[]`; routing
  signals bind to those shared assets by name.

## Keep examples in sync

When a public config field or supported routing surface changes, update its
fragment, the exhaustive reference, affected recipes, and the matching website
page together. Run `go test ./pkg/config/...` from `src/semantic-router`, then
run `make agent-lint` from the repository root:

```bash
cd src/semantic-router
go test ./pkg/config/...

cd ../..
make agent-lint
```

Complete routing scenarios belong in `config/recipes/`; backend support files
belong in `config/runtime/`; local Envoy and test-only manifests belong under
`deploy/` and `e2e/`, respectively.
