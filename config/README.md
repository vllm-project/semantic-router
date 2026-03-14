# Config Assets

`config/` is now the user-facing config surface only.

- `config/config.yaml`: canonical runnable starter config
- `config/signal/`: reusable `routing.signals` fragments
- `config/decision/`: reusable `routing.decisions` rule-shape fragments
- `config/algorithm/`: reusable `decision.algorithm` snippets
- `config/plugin/`: reusable route plugin snippets

`config/decision/` is organized by boolean rule shape:

- `single/`: one signal condition
- `and/`: conjunction examples
- `or/`: disjunction examples
- `not/`: exclusion examples
- `composite/`: nested AND/OR/NOT cases

`config/algorithm/` is organized by routing policy:

- `looper/`: multi-model execution policies such as `confidence`, `ratings`, and `remom`
- `selection/`: candidate-selection policies such as `elo`, `router_dc`, `automix`, and `latency_aware`

The repository enforces this fragment catalog and the core public config docs in Go tests. When a supported signal, decision algorithm, plugin surface, or canonical contract term changes, `go test ./pkg/config/...` will fail until both `config/` and the core config docs are updated to match.

`config/` no longer carries runtime support files or test manifests.

- Runtime support examples moved to `examples/runtime/`
- Local Envoy moved to `deploy/local/envoy.yaml`
- Harness and smoke manifests moved to `e2e/config/`

The old full-example trees under `config/intelligent-routing/`, `config/memory-rag/`, `config/multi-modal/`, `config/observability/`, and `config/prompt-guard/` were retired in v0.3. Build new configs from canonical `config/config.yaml` plus the fragment directories above.
