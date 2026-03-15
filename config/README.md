# Config Assets

`config/` is now the user-facing config surface only.

- `config/config.yaml`: exhaustive canonical reference config
- `config/signal/`: reusable `routing.signals` fragments
- `config/decision/`: reusable `routing.decisions` rule-shape fragments
- `config/algorithm/`: reusable `decision.algorithm` snippets
- `config/plugin/`: reusable route plugin snippets

Inside canonical `config.yaml`:

- `providers.defaults` holds provider-wide defaults such as `default_model` and reasoning families
- `providers.models[]` holds concrete backend access details directly
- `routing.modelCards[]` holds semantic model metadata, including optional `loras[]` catalogs for decision-level `lora_name` references
- `global.router`, `global.services`, `global.stores`, `global.integrations`, and `global.model_catalog` expose router-wide overrides explicitly
- router-owned model-backed module config lives under `global.model_catalog.modules`

`config/decision/` is organized by boolean rule shape:

- `single/`: one signal condition
- `and/`: conjunction examples
- `or/`: disjunction examples
- `not/`: exclusion examples
- `composite/`: nested AND/OR/NOT cases

Decision fragments may reference `modelRefs[].lora_name`, but those adapter names must be declared in the base config's `routing.modelCards[].loras`.

`config/algorithm/` is organized by routing policy:

- `looper/`: multi-model execution policies such as `confidence`, `ratings`, and `remom`
- `selection/`: candidate-selection policies such as `elo`, `router_dc`, `automix`, and `latency_aware`

The repository enforces this fragment catalog, the exhaustive reference config, the maintained deploy/E2E config assets, and the core public config docs in Go tests. When a supported signal, decision algorithm, plugin surface, or canonical contract term changes, both `go test ./pkg/config/...` and `make agent-lint` will fail until `config/`, maintained `deploy/` / `e2e/` config assets, and the core config docs are updated to match.

Latest official tutorials mirror the same top-level taxonomy:

- `tutorials/signal/`
- `tutorials/decision/`
- `tutorials/algorithm/`
- `tutorials/plugin/`
- `tutorials/global/`

`config/` no longer carries runtime support files or test manifests.

- Runtime support examples moved to `examples/runtime/`
- Local Envoy moved to `deploy/local/envoy.yaml`
- Harness and smoke manifests moved to `e2e/config/`

The old full-example trees under `config/intelligent-routing/`, `config/memory-rag/`, `config/multi-modal/`, `config/observability/`, and `config/prompt-guard/` were retired in v0.3. Use `config/config.yaml` as the exhaustive contract reference, then copy or trim it into deployment-specific `config.yaml` files as needed.
