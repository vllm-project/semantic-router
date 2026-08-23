# Configuration assets

Use this directory to find a complete configuration, copy a focused fragment,
or start from a maintained routing recipe.

| Need | Start here |
| --- | --- |
| See every supported field | `config/config.yaml`, the exhaustive canonical reference config |
| Add one routing capability | `config/fragments/` |
| Run a complete use case | `config/recipes/` |
| Start from a built-in Recipe | `config/recipes/built-in/` |
| Configure a storage or service backend | `config/runtime/` |
| Validate a managed asset | `config/schemas/` |

The website's [configuration guide](../website/docs/installation/configuration.md)
is the reader-facing reference. `config/config.yaml` is intentionally exhaustive;
trim it for a deployment instead of treating every optional service as required.

## Canonical shape

A router config uses these top-level sections:

```yaml
version: v0.4
listeners: []
models: []
recipes: []
entrypoints: []
global: {}
```

- `listeners` exposes inference and management endpoints.
- `models` contains each logical Model as a readable `name`, semantic `card`,
  one or more provider `connections`, and optional `runtime` and `pricing`.
  Each connection may select a catalog interface such as `chat`, `responses`,
  or `messages`; omitting `interface` selects that Provider's declared default.
- `recipes` contains reusable, model-free routing documents. Each document owns
  signals, projections, decisions, strategy, algorithms, and route plugins.
- `entrypoints` defines callable virtual models. The common form references one
  Recipe by name and assigns an ordered Model list to every Decision name;
  conditional rules use the same readable references. There is no detached
  pool or binding resource. Each effective rule runs in its own routing-state
  scope while sharing Models and router-wide services. See
  [`tutorials/global/entrypoints-and-recipes.md`](../website/docs/tutorials/global/entrypoints-and-recipes.md).
- `global` owns cross-cutting router settings, services, stores, integrations,
  and router-managed model assets.

Validate a file before serving it:

```bash
vllm-sr validate --config config.yaml
vllm-sr serve
```

`serve` starts the infrastructure and reads `config.yaml` only as deployment
bootstrap. Models, Recipes, decision assignments, and Entrypoints are created
through the Router Management API or the Dashboard; they are not selected by a
CLI operand.

## Choose the right asset

### Fragments

Fragments show one capability in its owning section. They are not complete
deployments and may rely on model or service definitions from a base config.

- `config/fragments/signal/`: request and response facts used by decisions.
  Heuristic and learned signal guides live under
  `tutorials/signal/heuristic/` and `tutorials/signal/learned/`.
- `config/fragments/decision/`: `single`, `and`, `or`, `not`, and nested
  boolean rule shapes.
- `config/fragments/algorithm/`: per-decision selection and bounded multi-model
  execution policies over the Models assigned by an Entrypoint.
- `config/fragments/plugin/`: route-local request or response processing such
  as caching, memory, RAG, tool policy, and safety handling.

The corresponding website sections are
[`tutorials/signal/`](../website/docs/tutorials/signal/),
[`tutorials/decision/`](../website/docs/tutorials/decision/),
[`tutorials/algorithm/`](../website/docs/tutorials/algorithm/),
[`tutorials/plugin/`](../website/docs/tutorials/plugin/), and
[`tutorials/global/`](../website/docs/tutorials/global/).

### Recipes and built-in policy

`config/recipes/` contains complete, runnable examples. Read a recipe's Model
Card before using it; the card explains its intended use, backend roles, data
handling, evaluation scope, and limitations.

`config/recipes/built-in/` is the versioned source for model-free Recipes
bundled with each release. Inspect a built-in Recipe in the Dashboard, then assign
configured Models and publish a Mixture of Models. Independent control planes
can perform the same lifecycle through the Router Management API.

### Runtime examples

`config/runtime/` contains backend-specific support files for memory, response
cache, the Response API, tools, and vector stores. These files configure a
runtime dependency; they do not define routing behavior by themselves.

## Important boundaries

- Standalone model backends use `credential_ref` to select a named
  `global.services.backend_credentials` entry backed by exactly one
  `secret_file` or `secret_env`. Literal backend keys and caller-supplied
  authorization headers are rejected. Managed mode uses published
  ProviderCredential resources instead of YAML secret references.
- Managed mode requires Router-terminated Management TLS. Configure exactly one
  file or environment source for both the server certificate and private key;
  a client CA source enables required, verified mTLS.
- Managed mode requires `global.services.agent.public_inference_endpoint` to
  name the ordinary public Router `/v1/chat/completions` endpoint. It must not
  point to the Dashboard or a physical model backend. Agent calls use delegated
  API keys so they pass through the same access, quota, logging, and usage path
  as every other inference request.
- Managed usage storage uses fixed UTC-month partitions. Under
  `global.services.access.usage_storage`, `create_ahead_months` and
  `maintenance_interval` tune bounded lifecycle work; `raw_retention` is empty
  by default and must be set explicitly before any raw month can be retired.
  Settlement tombstones and audit history are retained indefinitely.
- `models[]` is the only runtime Model definition. A backend contains compiled
  adapter fields; provider product metadata and authoring forms remain in the
  control plane.
- `recipes[].document.projections` derives named routing outputs from signals. Decisions
  consume those outputs instead of embedding free-form computation.
- Candidate iteration is bounded policy metadata, not a general scripting
  runtime.
- Router Learning lives under `global.router.learning`; it is separate from a
  decision's request-time base algorithm.
- Router replay is disabled by default and can capture request or response
  bodies. Review its access controls and retention settings before enabling it.
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
