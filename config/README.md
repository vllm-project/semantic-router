# Config Assets

`config/` is now the user-facing config surface only.

- `config/config.yaml`: exhaustive canonical reference config
- `config/signal/`: reusable `routing.signals` fragments
- `config/decision/`: reusable `routing.decisions` rule-shape fragments
- `config/algorithm/`: reusable `decision.algorithm` snippets
- `config/plugin/`: reusable route plugin snippets

Inside canonical `config.yaml`:

- `providers.defaults` holds provider-wide defaults such as `default_model` and reasoning families
- `providers.models[]` holds concrete backend access details directly, including optional `backend_id` and `engine_kind` identity hints on each `backend_refs[]` entry
- `providers.models[].pricing` supports separate prompt, cached-input, cache-write, and completion rates; omitted `cache_write_per_1m` falls back to `prompt_per_1m`
- `routing.modelCards[]` holds semantic model metadata, including optional `loras[]` catalogs for decision-level `lora_name` references
- `routing.projections` carries cross-signal coordination and derived routing outputs
- `routing.projections.partitions` is the canonical runtime home for exclusive domain or embedding partitions; DSL authoring uses `PROJECTION partition`
- `routing.projections.scores` and `routing.projections.mappings` let maintained configs turn learned and heuristic signals into named routing bands that decisions can reference with `type: projection`
- `routing.decisions[].candidateIterations` carries bounded DSL `FOR ... IN` metadata for candidate-model authoring; it is declarative selection policy input, not a general scripting runtime
- `routing.decisions[].emits[]` carries typed side-effect directives from DSL `EMIT` blocks; the current supported kind is `retention`, where `drop: true` skips response-side semantic-cache writes and the remaining fields stay structured/auditable for follow-up runtime consumers such as turn-aware cache TTL, current-model affinity, prefix/KV-cache warmth, and session transition telemetry
- request-shape detectors such as `routing.signals.structure` stay in the signal layer as typed named facts; numeric thresholds live inside the detector config instead of turning decisions into a free-form expression language
- `routing.signals.embeddings[].query_modality` declares which modality of incoming request payload the embedding rule's query is computed from. Defaults to `"text"`; `"image"` and `"audio"` require `global.model_catalog.embeddings.semantic.embedding_config.model_type=multimodal` so the query and candidate embeddings land in the same shared space. See `website/docs/tutorials/signal/learned/embedding.md` for the worked multimodal example.
- structure `density` features now use built-in multilingual text-unit normalization; the contract no longer exposes a per-rule `normalize_by` switch
- the dashboard and DSL builder now expose the same projection surface directly; see `website/docs/tutorials/projection/overview.md` and the maintained `deploy/recipes/balance.{yaml,dsl}` pair for end-to-end usage
- top-level `entrypoints` maps request-facing virtual model names onto named routing recipes; the names never reach a backend, they only select which routing profile evaluates the request
- top-level `recipes` adds named routing profiles beside the `routing` block. The top-level `routing` profile is the `default` recipe; each `recipes[].routing` block carries the same profile shape (`signals`, `projections`, `decisions`) but never `modelCards` — the model catalog and the signal/projection name registry stay global across recipes. See `website/docs/tutorials/global/entrypoints-and-recipes.md`
- `global.router`, `global.services`, `global.stores`, `global.integrations`, and `global.model_catalog` expose router-wide overrides explicitly
- `global.router.learning.adaptation` adds online model-choice learning after the base decision algorithm. `global.router.learning.protection` protects agentic continuity, cache, tool loops, and handoff cost. Decisions can opt out with `routing.decisions[].adaptations.mode: bypass`, use component-level `adaptations.adaptation.mode` / `adaptations.protection.mode`, or override the adaptation search space with `adaptations.adaptation.candidate_set`. `decision.algorithm.type=session_aware|elo|rl_driven|gmtrouter|bandit|personalization` is no longer a supported public algorithm.
- `global.services.router_replay.enabled` is the router-wide replay default; when it is on, decisions inherit replay capture unless a route-local `router_replay` plugin sets `enabled: false`
- embedding fallback tuning such as `global.model_catalog.embeddings.semantic.embedding_config.top_k` lives under the router-owned model catalog, not under individual signal rules
- prototype-aware exemplar compression and label scoring live alongside their owning signal families: `global.model_catalog.embeddings.semantic.embedding_config.prototype_scoring`, `global.model_catalog.modules.classifier.preference.prototype_scoring`, `global.model_catalog.kbs[].prototype_scoring`, and `global.model_catalog.modules.complexity.prototype_scoring`
- reusable startup-loaded knowledge bases live under `global.model_catalog.kbs[]`, while `routing.signals.kb[]` binds label/group matches into normal routing signals
- built-in knowledge base defaults keep `source.path` aligned with the steady-state `knowledge_bases/<dir>/` contract; local/dev runtime seeds missing built-ins from `config/knowledge_bases/<dir>/` into `.vllm-sr/knowledge_bases/<dir>/` once, and router reads the shared runtime KB store from there
- `global.router.config_source` selects the router's steady-state config source; the exhaustive reference uses `file`, while Kubernetes CRD reconciliation uses `kubernetes`
- `global.router.auto_model_names` declares the request model aliases that enter full vLLM-SR automatic routing. Defaults include `vllm-sr/auto`, `auto`, and `MoM`; `auto_model_name` remains the legacy single-name compatibility field.
- `global.router.skip_processing.enabled` opts the router into honoring the `x-vsr-skip-processing` request header; defaults to `false` so an arbitrary upstream caller cannot bypass router policy by injecting the header. Enable only when an authenticated upstream filter (ext_authz, etc.) is responsible for setting or stripping the header (see [#1808](https://github.com/vllm-project/semantic-router/issues/1808))
- router-owned model-backed module config lives under `global.model_catalog.modules`
- `global.model_catalog.modules.prompt_compression.profile` selects built-in signal-compression scoring defaults (`default`, `coding`, `medical`, `security`, or `multi_turn`); `multi-turn` is accepted as an alias, unknown names fail validation, and explicit weight/preserve fields override the profile.

`config/decision/` is organized by boolean rule shape:

- `single/`: one signal condition
- `and/`: conjunction examples
- `or/`: disjunction examples
- `not/`: exclusion examples
- `composite/`: nested AND/OR/NOT cases

Decision fragments may reference `modelRefs[].lora_name`, but those adapter names must be declared in the base config's `routing.modelCards[].loras`.
Candidate iteration fragments must stay bounded to `decision.candidates` or an explicit model list and feed existing decision outputs such as `MODEL <iterator>`.

`config/algorithm/` is organized by routing policy:

- `looper/`: multi-model execution policies such as `confidence`, `ratings`, `remom`, and `fusion`
- `selection/`: request-time candidate-selection policies such as `router_dc`, `automix`, `hybrid`, `multi_factor`, and `latency_aware`

Each supported algorithm now has its own tutorial page under `website/docs/tutorials/algorithm/`.

`config/plugin/` is organized by route-local plugin or reusable plugin bundle:

- one directory per plugin or bundle, such as `semantic-cache/`, `rag/`, `memory/`, or `content-safety/`
- route-local tool policy examples live under `tools/`
- one fragment example per directory in the current catalog

Each supported plugin now has its own tutorial page under `website/docs/tutorials/plugin/`.

The repository enforces this fragment catalog, the exhaustive reference config, the maintained deploy/E2E config assets, and the core public config docs in Go tests. When a supported signal, decision algorithm, plugin surface, or canonical contract term changes, both `go test ./pkg/config/...` and `make agent-lint` will fail until `config/`, maintained `deploy/` / `e2e/` config assets, and the core config docs are updated to match.

Latest official tutorials mirror the same top-level taxonomy:

- `tutorials/signal/`
  - `tutorials/signal/heuristic/` for rule-based and lightweight detector signals
  - `tutorials/signal/learned/` for embedding- and classifier-driven signals
- `tutorials/decision/`
- `tutorials/algorithm/` with one page per algorithm
- `tutorials/learning/` for cross-request Router Learning adaptation and protection
- `tutorials/plugin/` with one page per plugin
- `tutorials/global/`

`config/` no longer carries runtime support files or test manifests.

- Runtime support examples moved to `deploy/examples/runtime/`
- Local Envoy moved to `deploy/local/envoy.yaml`
- Harness and smoke manifests moved to `e2e/config/`

The old full-example trees under `config/intelligent-routing/`, `config/memory-rag/`, `config/multi-modal/`, `config/observability/`, and `config/prompt-guard/` were retired in v0.3. Use `config/config.yaml` as the exhaustive contract reference, then copy or trim it into deployment-specific `config.yaml` files as needed.
