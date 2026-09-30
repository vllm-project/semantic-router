# Open Decision Runtime for vLLM Semantic Router

This proposal makes vLLM Semantic Router (vllm-sr) a runtime for decision models, with an engine mode, the Open Decision API and router integration. All three share a tiered runtime that embeds the model contract rather than the inference engine.

| | |
|---|---|
| Status | Draft for review, 2026-09-30 |
| Scope | `vllm-sr` CLI, Go router, a new decision runtime (Go and Rust), the `vllm-sr-plugins` package |
| Markers | **Decided**: by the user. **Recommended**: coordinator recommendation awaiting confirmation. **OD-n**: open decision, all listed in §16.2. **TBD**: awaiting prototype results. |

## 1. Summary

vllm-sr can reach a decision model today only as an HTTP classifier with labels fixed in a mapping file. The Decision 1.0 launch post promised native integration and an **Open Decision API**. This proposal defines the runtime behind them, the **Open Decision Runtime (ODR)**:

1. **One runtime, two entry points.** `vllm-sr serve <hf-model>` starts a standalone engine that serves `/v1/decisions`, as `vllm serve` does. The router runs the same code through a new `decision` signal type and selector.
2. **Embed the contract, not the engine.** The router links `RouterModelRuntime`, one Go interface over one cgo boundary into a Rust core. The core owns the manifest, renderer, calibration and answer assembly, and the evaluation harness shares it. vLLM runs out of process, as a supervised sidecar on a Unix domain socket (UDS).
3. **Tiers placed at run time** from a hardware probe, the manifest and parity receipts:
   - encoders run in-process on ONNX Runtime (ORT), candle or llama.cpp;
   - label-token deciders run on vLLM's stock generate runner;
   - scoring-head models such as Decision 2.0 run on vLLM's pooling runner with our plugin;
   - MoE and ≥27B models run on a shared tier.
4. **Open to third-party models** through reviewed adapters. On a public decision-model leaderboard, 16 of the top 20 open entrants read label-token logits in one pass, which stock vLLM can serve given our renderer and calibration.
5. **No vLLM fork:** one pinned plugin package, plus upstream contributions.

The MVP needs about 17–22 engineer-weeks (2–2.5 months with two engineers), and full coverage 39–54. **Decided:** phase-1 work waits for the plugin prototype's measured parity and latency, which §10 holds as placeholders.

## 2. Background

### 2.1 The System One format

A request carries a **state** (text, a JSON object or an array) and a set of named, typed **questions**:

| Type | Input | Answer |
|---|---|---|
| Choice | Instructions and 2–255 named options | `choice`, `probabilities`, `confidence` |
| Noul | Instructions, with optional `false` / `true` descriptions | `noul`, which is P(true) |
| Score | Instructions and 2–10 ordered levels | `score` (the expected level), `probabilities`, `confidence`, `legend` |

A failed question returns its own error (`max_length_exceeded`, `invalid_question` or `invalid_model_output`), and over-length input is never truncated. The hosted System One API and each Decision 2.0 package's bundled runtime (`Decision2.system_one`) use this format.

### 2.2 What exists today

- `vllm-sr serve` starts only the router, Envoy and the dashboard; per its help text, it "does not download or launch the physical LLM engines".
- External models are reachable only through `http_classify` and `http_chat`. Their contracts take labels from a static mapping, so a question cannot vary per request.
- The in-process backend, candle or ORT, is chosen at build time.
- The System One engine exists only as Python code in the training tree. A package is identified by its root `config.json` pointer, `{"decision_format": "vllm-sr-decision", "format_version": 2, "package_schema": "dev2-package/1"}`, beside a `MODEL_MANIFEST.json` of per-file SHA-256.
- A [plugin prototype](https://github.com/vllm-project/semantic-router/tree/678ac2060ea5d0b2e6d63dea347aa9ce461fb34b/src/vllm-sr-plugins) already serves Decision 2.0 on unmodified vLLM (§8.2).

**Goals:** one API and runtime, standalone or in the router; vLLM's kernels and hardware coverage without a fork; third-party models with measured parity; routing that stays available; auditable answers.

**Non-goals for v1:** generative serving, decisions inside the served LLM's engine, diffusion deciders and training.

## 3. Decisions already made

| Area | Decision | Status |
|---|---|---|
| CLI | `vllm-sr serve <hf-model> [options]` detects a decision model from its manifest (root `config.json` with `decision_format: vllm-sr-decision`) and starts a System One engine, like `vllm serve`. With no model and `--config`, it starts the router as today. | **Decided** |
| API | A native `POST /v1/decisions`, plus `GET /v1/models`, `/health` and `/metrics` | **Decided** |
| API | A `POST /v1/systemone` alias, for ecosystem and client compatibility | **Recommended** (OD-1) |
| API | An optional `POST /classify` shim for the router's `label_distribution.v1` contract, off by default | **Recommended** (OD-2) |
| Router | The engine is router-managed by default (the same code as engine mode), with a remote engine as an option. A `decision` signal type (Choice / Noul / Score templates) and algorithms: model selection as a Choice, escalation as a Noul, complexity as a Score. One batched call per request, per-signal deadlines, fail-open. | **Decided** |
| Scope | Open: third-party models are served through family adapters | **Decided** |
| Sequencing | Phase-1 items wait for the prototype's measured parity and latency | **Decided** |
| MoE tier | Explored after the current 27B training milestone reports | **Recommended** (coordinator default) |

## 4. User experience

### 4.1 Engine mode

```bash
vllm-sr serve <org>/<decision-model> --revision <40-hex>              # Hub package
vllm-sr serve ./my-package --backend vllm --dtype bfloat16            # local package, forced backend
vllm-sr serve <org>/<third-party-decider> --manifest <overlay.yaml>   # reviewed overlay
vllm-sr serve --config config.yaml                                    # router mode, unchanged
```

Detection is deterministic:

1. Resolve a local directory, or a Hub repository at `--revision` (default `main`; the resolved commit is recorded).
2. Read only the root `config.json`. `decision_format: vllm-sr-decision` marks a first-party package; `format_version` selects its schema.
3. If there is no such pointer, use the registry overlay for `repo@revision`, or the one given with `--manifest`.
4. If there is no overlay either, exit and point to `vllm serve`. The CLI never guesses a family or enables `trust_remote_code`.
5. Download the files, verify their hashes, probe the hardware, place the model (§5.5) and serve.

New options are `--revision`, `--manifest`, `--backend auto|ort|candle|llamacpp|vllm`, `--device`, `--dtype`, `--port` (default 8000) or `--uds`, `--api-key-file`, `--enable-classify-shim` and `--accept-licence`. **OD-3:** what should `serve <model> --config` mean?

### 4.2 API v1

Request to `POST /v1/decisions`:

```json
{"model": "my-decision-model",
 "state": {"ticket": "The invoice total is wrong and I was charged twice."},
 "questions": {
   "team":   {"type": "choice", "instructions": "Which team should handle this?",
              "criteria": {"billing": "Payments and invoices", "tech": "Product defects"}},
   "urgent": {"type": "noul", "instructions": "Does the customer report a double charge?"},
   "effort": {"type": "score", "instructions": "How long will resolving this take?",
              "criteria": ["minutes", "hours", "days"]}},
 "options": {"deadline_ms": 50}}
```

Response (numbers illustrative):

```json
{"model": "my-decision-model",
 "answers": {
   "team":   {"type": "choice", "choice": "billing",
              "probabilities": {"billing": 0.96, "tech": 0.04}, "confidence": 0.76},
   "urgent": {"type": "noul", "noul": 0.93},
   "effort": {"type": "score", "score": 0.41, "probabilities": {"0": 0.63, "1": 0.33, "2": 0.04},
              "confidence": 0.28, "legend": {"0": "minutes", "1": "hours", "2": "days"}}},
 "usage": {"input_tokens": 412, "output_tokens": 0}}
```

| Endpoint | Purpose | Status |
|---|---|---|
| `POST /v1/decisions` | Native Open Decision API | **Decided** |
| `POST /v1/systemone` | The hosted System One body, unchanged | **Recommended** (OD-1) |
| `POST /classify` | One fixed question in `label_distribution.v1` or `score.v1` shape, with no Go changes | **Recommended**, off by default (OD-2) |
| `GET /v1/models` | Identity, revision, manifest hash, limits, licence flags, placement | **Decided** |
| `GET /health` | 200 when ready or degraded but serving, else 503 | **Decided** |
| `GET /metrics` | Prometheus: requests, latency by tier, queues, restarts, escalations | **Decided** |

`/v1/decisions` is a strict superset of System One. It adds three optional features:

- `options.deadline_ms`, which aborts late work;
- `options.return_meta`, which reports the placement, the cascade path and latencies;
- the errors `deadline_exceeded`, `unavailable` and `too_many_options`.

Answers keep the caller's question ids and order. API v1 freezes at the end of P0, with a JSON Schema and a conformance suite. **OD-4:** should one call accept several states?

### 4.3 Router mode

In the default **managed** mode, `vllm-sr serve --config` starts and supervises the engine (§5.8). Until P2 delivers the in-process runtime, that engine is an engine-mode container on a shared UDS. In **remote** mode, the router uses the new `http_decision` adapter. The config below is illustrative, and its field names are **OD-5**:

```yaml
global:
  model_catalog:
    deployments:
      decision-local: {provider: decision, artifact: <org>/<decision-model>,
                       revision: <40-hex>, placement: auto}
routing:
  signals:
    decisions:
      - {name: needs_reasoning, deployment: decision-local, type: noul, deadline_ms: 50,
         instructions: "Does answering this request need multi-step reasoning?"}
      - {name: difficulty, deployment: decision-local, type: score, deadline_ms: 50,
         instructions: "How hard is this request?", criteria: [easy, medium, hard]}
  decisions:
    - name: hard-reasoning
      rules:
        operator: AND
        conditions:
          - {type: decision, name: needs_reasoning, predicate: {gte: 0.7}}
          - {type: decision, name: difficulty, predicate: {gte: 1.5}}
      modelRefs: [{model: large-reasoner}, {model: medium-reasoner}]
      algorithm:
        type: decision
        decision: {deployment: decision-local, deadline_ms: 60,
                   instructions: "Which model should answer this request?"}
```

| Decision point | Question | Router surface |
|---|---|---|
| Model selection | Choice over the decision's `modelRefs` | New `decision` selector; probabilities become selection scores |
| Escalation | Noul, e.g. "Does the draft answer the request?" | Gate for the `confidence` looper |
| Complexity | Score over easy / medium / hard | `decision` signal with numeric predicates |

- **One batched call per request.** The planner (§5.4) splits the call by backbone.
- **Per-question deadlines.** A default of 50 ms is recommended; remote classifiers default to 5 s today.
- **Fail-open.** A late or failed answer leaves its signal unknown, so the rule's `on_error` / `on_unknown` policy applies (default `no_match`). A failed selector falls back to the first `modelRef`.

**OD-6:** `modelRefs` are known only after a decision wins. Should the planner send (a) one Choice per candidate decision, which is exact, or (b) one Choice over the union of candidates, renormalised to the winner's refs, which is cheaper?

## 5. Architecture

### 5.1 Embed the contract, not the engine

| Evidence | Consequence |
|---|---|
| vLLM's EngineCore runs in its own process by default (`VLLM_ENABLE_V1_MULTIPROCESSING=1`), and parallelism above 1 spawns workers. | Embedding still spawns processes |
| After CUDA initialises, vLLM forces `spawn`, which re-executes `sys.executable`. | A Python interpreter must exist on disk |
| Go-to-Python calls need `runtime.LockOSThread` and `PyGILState_Ensure`; Python 3.12+ aborts on misuse, and a maintained Go binding documents crashes in about one run in three without its fix. | A fragile request path |
| An in-process CUDA fault or OOM kills the ExtProc server. | All routing fails |
| NVIDIA Dynamo's 2026 proposals ([#10835](https://github.com/ai-dynamo/dynamo/issues/10835), [#10838](https://github.com/ai-dynamo/dynamo/issues/10838)) move from an in-process PyO3 shim to a sidecar driving stock `vllm serve` over a UDS, citing dependency entanglement, private-API fragility and GIL contention. TEI and Ollama also run engines out of process. | The established pattern |
| A UDS round trip costs 4–11 µs and gRPC over a UDS 0.1–0.2 ms, against a 20–30 ms decoder call. | Little latency to save |

Embedding saves about 0.1 ms at the cost of fault isolation, large router images, slow start-up and lock-step upgrades. The nearest supported path is vLLM's experimental Rust frontend, which drives a headless EngineCore but lacks pooling so far ([#54144](https://github.com/vllm-project/vllm/pull/54144)); we adopt it in P3.

Alternatives considered and rejected:

- **candle:** no Qwen3.5 / Gated DeltaNet (GDN) support outside open PRs, and no ROCm.
- **ORT for decoders:** the linear-attention ops have CPU and CUDA kernels only. ORT stays the encoder backend.
- **A thin Python service only:** exact answers, but FP32 weights stay resident (about 103 GB, against 55 GB in BF16, at 27B). It remains the P0 stopgap.
- **Inside the served LLM's vLLM instance:** one runner type per instance, closed pooling-in-generate PRs ([#40804](https://github.com/vllm-project/vllm/pull/40804), [#30672](https://github.com/vllm-project/vllm/pull/30672)), and weak early-layer signals.
- **SGLang:** a second engine to maintain, no listed ModernBERT or DeBERTa support, and reported gRPC regressions (**OD-7**).

### 5.2 Components

```text
            +------------- vllm-sr router (Go, Envoy ExtProc) --------------+
request --> | signals --> decision engine (rules) --> selector              |
            |   decision signals + selector questions: one bundle/request   |
            | RouterModelRuntime (Go interface) === one cgo boundary ===+   |
            +-----------------------------------------------------------|---+
                                                                        v
 Rust core: manifest | renderer | calibration | answers | planner | placement | supervisor
   |-- Tier 0   in-process: ORT (MIGraphX, CUDA, CPU, CoreML, OpenVINO), candle, llama.cpp
   |-- Tier 1a  UDS sidecar: vLLM generate runner                  (label-token deciders)
   |-- Tier 1b  UDS sidecar: vLLM pooling runner + vllm-sr-plugins (scoring heads)
   '-- Tier 2   shared or remote: ODR engine or vLLM fleet         (MoE, >=27B, judges)
 Engine mode: the same core behind a thin HTTP server.  Evaluation: the same core via Python bindings.
```

### 5.3 RouterModelRuntime and the Rust core

```go
type RouterModelRuntime interface { // package decisionruntime; names illustrative
    Load(ctx context.Context, spec ModelSpec) (ModelInfo, error) // resolve, verify, place
    Decide(ctx context.Context, b Bundle) (Answers, error)      // one call per routed request
    Models() []ModelInfo
    Health() HealthReport
    Close() error
}
```

Go makes one cgo call per routed request. The call carries the state, the questions, each question's deadline and the deployment ids, and it is cancelled with the Go context. The Rust core runs its own executor, and it handles:

- manifest checks;
- rendering, from a template to token IDs and answer positions;
- readouts, calibration and answer assembly;
- planning, placement and all backend clients.

For decision workloads, the core gradually replaces the router's four cgo bindings. The evaluation harness uses the same core through Python bindings, so parity receipts measure the serving code. **OD-8:** which language should the renderer use? The prototype runs each package's vendored Python renderer inside a vLLM endpoint plugin, so it matches training byte for byte.

### 5.4 Request planner

The planner groups a request's questions by **backbone** and runs one concurrent plan per backbone; each question keeps its own deadline.

- **Tier 1a:** the state becomes a shared prefix and each question a suffix. The planner primes the prefix once and then fans out, so prefix caching reuses the state.
- **Tier 1b:** vLLM disables prefix caching for hybrid pooling, so each question re-reads the state; packed models (§14) would need a single pass.
- **Tier 0:** each encoder runs one batched forward pass.
- **Shared backbones:** adapters on one backbone can share a plan once multi-LoRA heads ([#53555](https://github.com/vllm-project/vllm/pull/53555), merged on 2026-09-16 but absent from the v0.30.0 tag) are released.

### 5.5 Placement at run time

Placement replaces the build-time backend choice. Its inputs are:

- a hardware probe (GPU, memory, loadable ORT providers, CPU features, the sidecar's vLLM build);
- the manifest;
- parity receipts;
- operator policy (overrides, latency target, memory budget).

The runtime keeps the backends that support the model, pass a receipt and fit the budget, and picks one by manifest preference and latency. The choice is reported in `/v1/models`. A backend without a receipt needs explicit approval, and the model is then shown as `unverified`. **OD-9:** what receipt threshold?

### 5.6 Tiers

| Tier | Runs | Backends | Models |
|---|---|---|---|
| **0** | In-process (Rust core) | ORT (MIGraphX, CUDA/TensorRT, CPU, CoreML, OpenVINO), candle, llama.cpp | Encoders ≤0.5B (Vela, Kai/Lex, marker heads), GLiNER2 via ONNX, Qwen3-0.6B-class on CPU, GGUF deciders |
| **1a** | vLLM sidecar, generate runner | Stock vLLM: `allowed_token_ids`, logprobs (`--max-logprobs 256`), prefix caching (`--mamba-cache-mode align` for hybrids) | Label-token deciders, optional thinking escalation |
| **1b** | vLLM sidecar, pooling runner | vLLM + `vllm-sr-plugins` | Scoring-head decoders (Decision 1.0 / 2.0 candidate head, K-slot heads) |
| **2** | Shared or remote | Remote ODR engine, vLLM fleet, `http_chat` judges | Resident MoE and ≥27B models, JSON judges, diffusion (deferred) |

A vLLM instance runs one runner type and one model, plus its LoRA adapters. That is why Tier 1 is split into 1a and 1b. Engines can share a GPU through `--gpu-memory-utilization`. MoE memory scales with total parameters, so one large Tier 2 GPU can serve many routers.

### 5.7 Cascades

A question escalates to the next tier under two conditions: its calibrated confidence is below the manifest or policy threshold, and its remaining deadline covers that tier's p95 latency. In Tier 1a, a bounded thinking step followed by a second label read over the cached prefix also counts as an escalation. Confidences are compared only after per-backend calibration, and the tier that answered is reported.

### 5.8 Sidecar supervision

- **Readiness:** vLLM's health check must pass, and a warm-up request must reproduce stored golden answers, which proves identity and numerics.
- **Health:** checks and canaries run over the UDS. Timeouts degrade the sidecar; restarts back off from 1 s to 60 s, and a crash loop marks it failed.
- **Memory:** vLLM pre-allocates KV memory, so every sidecar and every in-process CUDA context gets an explicit budget.
- **Degraded mode:** questions fall back to a placement that has a receipt; if there is none, they return `unavailable` and fail open.
- **Ownership:** on a bare host, the runtime restarts `vllm serve`. Under Docker or Kubernetes, the platform restarts it, so the router never needs the Docker socket. The router does not wait for sidecars to become ready, unless a deployment is `required`.

### 5.9 Latency budget

| Component | Figure | Basis |
|---|---|---|
| HTTP keep-alive plus vLLM's API-server-to-engine hop | About 1–4 ms | Estimate |
| The sidecar's Python frontend | Not measured | **TBD: prototype results** |
| 27B-class package | About 73 ms p50 (BF16, short prompts); 138–150 ms p50 per item on the package-native path | Internal measurement |
| The shipped runtime's per-call FP32-to-BF16 weight recast | About 10 ms at 9B, 30 ms at 27B | Estimate; avoided by keeping Linear weights in BF16 |

## 6. Decision Manifest v1

Each model has one content-addressed manifest, which doubles as the evaluation harness's adapter spec.

| Section | Contents |
|---|---|
| Identity, licence | Name, repository, 40-hex revision, per-file SHA-256, SPDX licence, flags (non-commercial, research-only, gated, no-redistribution) |
| Backbone | Family (e.g. Qwen3.5 GDN hybrid, Gemma 4, ModernBERT, DeBERTa-v3), total and active parameters, dtypes per backend, LoRA base pin, weight codec (plain or `bf16z`) |
| Renderer | Template id and SHA-256; layout (shared-prefix branches, packed slots, encoder markers, GLiNER schema); label alphabet; thinking; tokenizer hash |
| Readout | `label_token`, `head` (kind, weights, FP32), `slots`, `span`, or `canvas` (reserved) |
| Answer positions | Last token, marker tokens, option endpoints, or slot k |
| Calibration | Global, per-type or option-count temperature; Score offsets; softcapping. Each fit is bound to a model hash, backend and dtype. |
| Score semantics | Ordinal distribution or isolated levels; key base; how `score` is computed |
| Limits, overflow | Maximum tokens, options, levels and questions; overflow `refuse`, `shortlist` or `multi_pass`; over-length input always returns `max_length_exceeded` |
| Escalation | Thinking gate and budget; cascade targets |
| Parity receipts | Per backend, version, hardware class and dtype: panel id and hash, items, answer changes, maximum probability drift, calibration reference, harness version, date |

**First-party packages** extend the existing pointer: the root `config.json` gains a `decision_manifest` entry naming `DECISION_MANIFEST.json`, whose hash `MODEL_MANIFEST.json` records. For older packages, the runtime derives the manifest from `MODEL_MANIFEST.json`, `decision_config.json`, `calibration.json` and `score_bias.json`.

**Third-party models** use overlay manifests from a registry that ships with vllm-sr.

- An overlay pins the author's repository, revision and hashes, names a reviewed adapter and renderer, and carries our harness's calibration and receipts.
- Weights are downloaded unmodified and are never redistributed.
- A local overlay passed with `--manifest` is marked `unreviewed`.

```yaml
schema: decision-manifest/1
identity: {name: <org>/<decider>, source: {repo: <org>/<decider>, revision: <40-hex>},
           licence: {spdx: Apache-2.0, flags: []}}
backbone: {architecture: Qwen3_5ForCausalLM, family: qwen3_5-hybrid, text_only: true}
renderer: {template: <template-id>, sha256: <64-hex>, layout: branches, labels: letters}
readout: {kind: label_token, position: last_token}
calibration: {form: temperature_by_option_count, fits: [{backend: vllm-generate, dtype: bfloat16}]}
score: {semantics: ordinal_distribution, key_base: 0}
limits: {max_input_tokens: 32768, max_options: 255, overflow: refuse}
escalation: {thinking: {confidence_below: 0.7, max_tokens: 512}}
parity_receipts: []
```

**OD-10:** where should the manifest and the registry live?

## 7. Adapter families

Counts come from the 31-model survey in Appendix A.

| Family | Readout | Engine path | Fork? | Count | First-party |
|---|---|---|---|---|---|
| Label-token decider | LM-head label-token logits at the answer position, one pass | Tier 1a stock vLLM (`allowed_token_ids` + logprobs); GGUF on llama.cpp | No | 21 | None yet (§14) |
| Scoring-head decoder | Candidate-endpoint head or last-position K-slot head | Tier 1b pooling runner + `vllm-sr-plugins` | Plugin | 3 | Decision 1.0 decoders, Decision 2.0 |
| Multi-slot packing | Several answer slots after one state | STEP token pooling ([#55307](https://github.com/vllm-project/vllm/pull/55307)) or endpoint plugin | Plugin | Layout variant | Future models |
| Encoder + marker head | ModernBERT / mmBERT with option-marker head | Tier 0 ORT or candle; or vLLM ModernBERT + plugin head | No | 2 | Vela, Kai/Lex |
| GLiNER2 / DeBERTa-v3 | Span / label extractor, labels at call time | Tier 0 ORT only (no DeBERTa in vLLM, SGLang or llama.cpp) | No | 3 | None |
| Diffusion canvas | Denoise an answer canvas, then read labels | vLLM DiffusionGemma mode ([#57250](https://github.com/vllm-project/vllm/pull/57250)) plus out-of-tree fixes | **Deferred** | 2 | None |

Without a fork, the families cover 29 of the 31 models: 21 on stock vLLM, 5 through the plugin (the two marker-head encoders also fit Tier 0), and 3 on ORT. The remaining work is renderers and calibration, not engine changes.

## 8. vLLM integration without forks

### 8.1 The plugin registry

`vllm-sr-plugins` (package `vllm_sr_plugins`) registers through vLLM's documented entry points.

| Entry point | Provides |
|---|---|
| `vllm.general_plugins` | `ModelRegistry.register_model` for our model classes. The first is `Decision2Qwen3_5ForScoring`: the text-only Qwen3.5 backbone plus the FP32 candidate head, with no LM head. Classes for Qwen3, Decision 1.0 and K-slot heads follow. |
| Poolers | The candidate pooler gathers the option-endpoint and query rows from every prefill chunk and returns one FP32 logit per candidate |
| `vllm.endpoint_plugins` | A route that passes per-request positions in `PoolingParams.extra_kwargs`: `/v1/system_one` in the prototype, a thin positions-to-logits route in the target design |

To launch, run `vllm serve <package> --runner pooling --hf-overrides '{"architectures": [...]}'`, with `VLLM_PLUGINS` naming both plugins (endpoint plugins load only when listed). The endpoint plugin exists because HTTP `/pooling` forwards only `task`, `use_activation` and `dimensions`, and IO processors copy a single `PoolingParams` to every prompt. **OD-11:** serve the 27B adapter through vLLM LoRA, or merge it offline?

### 8.2 What the prototype has shown

The prototype runs on a pinned vLLM `0.29.1rc1.dev187` (ROCm, MI325X / gfx942). Findings so far:

- **No vLLM source changes are needed.**
- **Model Runner V2 rejects the `plugin` pooling task**, so the pooler serves the built-in `token_classify` task.
- **Chunked prefill works**, because the pooler accumulates rows across chunks.
- **Vela 1.0 classifiers need no plugin**; they run on native ModernBERT classification (parity TBD).
- **Limits of v0:** only `qwen-full` packages, with no 27B adapters, `bf16z` or Qwen3 yet; no hybrid prefix caching; and a request without positions returns `invalid_model_output`.

Parity and latency are **TBD: prototype results** (§10).

### 8.3 Version pinning and compatibility CI

- **Pinning:** each engine image pins one vLLM build and one plugin version, and the plugin warns on any other build.
- **Release testing:** for every candidate vLLM release, CI runs unit tests (head, chunked-prefill gathering, registration, service contract) and a GPU parity panel for each model family. An image ships only if its receipts pass.
- **Coverage and early warning:** the tests exercise every interface the plugin touches (`Pooler`, `PoolingMetadata`, `PoolingParams.extra_kwargs`, weight loading, endpoint plugins and `EngineClient`), and a nightly run against `main` flags breakage before it reaches a release.
- **Hardware plugins:** they lag upstream by one to six releases, so each gets its own pin.

### 8.4 Upstream contributions

| Contribution | Benefit | Status |
|---|---|---|
| Per-request pooling params through `/pooling` | No Tier 1b endpoint plugin | Not started |
| Prefix caching for hybrid pooling | No per-question state re-reads | Disabled; generation has `align` mode ([#30877](https://github.com/vllm-project/vllm/pull/30877)) |
| A classify / pooling RPC in `vllm serve --grpc` | A typed UDS transport (only Generate and Embed today, [#36169](https://github.com/vllm-project/vllm/pull/36169)) | RFC open ([#47768](https://github.com/vllm-project/vllm/issues/47768)) |
| Pooling in the Rust frontend client | The Rust core drives EngineCore (P3) | Open ([#54144](https://github.com/vllm-project/vllm/pull/54144), [#54648](https://github.com/vllm-project/vllm/pull/54648)) |
| The `plugin` task on Model Runner V2 | Custom poolers keep their own task | Found by the prototype; not filed |

Multi-LoRA with per-adapter heads (§5.4) needs no contribution; it only needs the next vLLM release.

## 9. Router integration details

### 9.1 CLI and packaging

| Extension point | Change |
|---|---|
| `src/vllm-sr/cli/commands/runtime.py` (`_execute_serve`) | Optional positional `MODEL`; branch into engine mode before `_resolve_serve_config` |
| `runtime_help.py`, `website/docs/api/cli.md` | Replace the "does not launch engines" text; regenerate with `tools/docs/generate_cli_reference.py` |
| Download | `huggingface_hub` (already a dependency) at a pinned revision into `.vllm-sr/models`, reusing the `HF_ENDPOINT` / `HF_HOME` passthrough; tokens come from a file |
| Engine images | `vllm-sr-engine` (CPU, CUDA, ROCm) for the server, core and Tier 0. Tier 1 images are built from `vllm/vllm-openai` / `vllm/vllm-openai-rocm` plus the plugin wheel. Both reuse the container runner and GPU passthrough (**OD-12**). |
| Lifecycle | A `decision` service name for `status`, `logs` and `stop` |

### 9.2 Contract, adapter, signal and selector

- **Contract `decision_answers.v1`:** the `/v1/decisions` request and response, registered in `pkg/config/classifier_backend.go`.
- **Adapter `http_decision`:** built on `pkg/modelruntime/connector`, reusing its timeouts, byte caps, bearer auth and metrics.
  - Decisions are idempotent, so it allows one retry within the deadline.
  - It raises `MaxIdleConnsPerHost` above the Go default of two that the connector inherits.
  - It connects over a UDS to managed engines and over HTTP(S) to remote ones.
  - Consumers `decision.<name>` are validated in `model_deployments.go`, and admission control applies per deployment.
- **`decision` signal:** a type constant, a catalog entry with label-qualified references (`routing_surface_catalog.go`), a YAML collection (`canonical_config.go`) and a dispatcher.
  - Each signal is a template: type, instructions, criteria, a state mapping and a deadline.
  - A Choice matches by label, and a predicate can test the chosen option's probability.
  - Noul and Score use the existing `gt` / `gte` / `lt` / `lte` predicates on P(true) and on the expected level.
- **`decision` selector:** modelled on `prompt`, which asks a chat model for `{"selected_model", "rationale"}` JSON with a 5 s timeout. The new selector asks one Choice over the `modelRefs` instead, and its calibrated probabilities can be combined with cost and latency. The `confidence` looper can use a Noul as its gate.
- **Observability:**
  - `/api/v1/inventory/models` shows identity, placement and receipts.
  - `routing/preview` traces show answers, tiers and latency.
  - A new `x-vsr-decision-latency-ms` header is added.
  - States are never logged.

### 9.3 Migration path from classifier signals

1. **Today, with no Go changes:** a fixed question behind the `/classify` shim feeds a `sequence_classifier` signal, or feeds `complexity` through `score.v1`.
2. **Shadow:** `decision` signals marked `shadow: true` run and log their agreement with existing signals; no rule reads them.
3. **Switch:** move rules to decision signals one domain at a time, keeping the classifier as the `on_error` fallback.
4. **Consolidate:** move built-in model signals to decision templates where they do better. Tier 0 encoders such as Vela remain as fast paths.

## 10. Evaluation and parity

- **Shared harness.** The existing evaluation contract already fits: one JSONL row per prompt with answers and latency, with over-budget inputs counted as invalid rather than truncated. The harness runs each model on its author's native path and on ODR, and emits parity receipts.
- **Receipts for every placement** (fields in §6). First-party releases already work this way; one recent release changed no answers on its full prompt set after a real download.
- **Calibration per backend.** Temperatures and Score offsets are bound to the scored model's identity. The scored Decision 2.0 path keeps embeddings, norms, convolution filters, recurrent-state parameters and the residual stream in FP32 under BF16 autocast, while vLLM runs BF16 throughout. A new backend or dtype therefore needs its own fit, or a receipt showing that the old fit changes no answers.
- **Golden tests:**
  - identical token IDs, answer positions and prompt SHA-256 from the Python and Rust renderers (training already records these);
  - stored answers for each backend;
  - API conformance.
- **Latency benchmarks:** p50, p95 and p99; throughput by concurrency; cold start; memory. Each is measured per tier, backend and hardware class.

**TBD: prototype results.** The prototype's session tool measures the items below; its record fills these tables.

| Decision 2.0 0.8B through the plugin, compared with stored predictions | Items | Answer changes (BF16) | Max drift (BF16) | Answer changes (FP32) | Max drift (FP32) |
|---|---|---|---|---|---|
| Release panels (typed, human-transfer, public, multilingual subset) | TBD | TBD | TBD | TBD | TBD |

| Latency, Decision 2.0 0.8B, same GPU | p50 (ms) | p95 (ms) | Throughput by concurrency |
|---|---|---|---|
| Shipped runtime (FP32 master weights, BF16 autocast) | TBD | TBD | TBD |
| Shipped runtime, Linear weights resident in BF16 | TBD | TBD | TBD |
| vLLM plus plugin, BF16 | TBD | TBD | TBD |
| vLLM plus plugin, FP32 | TBD | TBD | TBD |

Also TBD: a Vela classifier on native ModernBERT classification compared with Transformers (argmax changes and maximum drift by length bucket), and Decision 2.0 co-located with Vela on one GPU. **OD-13:** what should the default Tier 1b dtype be?

## 11. Hardware and backend matrix

| Backend | NVIDIA | AMD Instinct | Intel | Apple silicon | CPU |
|---|---|---|---|---|---|
| ORT (Tier 0) | CUDA, TensorRT | MIGraphX (ROCm provider removed in 1.23) | OpenVINO | CoreML | x86, ARM |
| candle (Tier 0) | CUDA | None (unmerged PR) | None | Metal | MKL / Accelerate |
| llama.cpp (Tier 0) | CUDA | HIP | SYCL | Metal | x86, ARM, RISC-V |
| vLLM (Tier 1) | CUDA | ROCm | XPU | vllm-metal (pooling experimental) | x86, AArch64 |

vLLM also reaches TPU, Ascend, Gaudi and Neuron through hardware plugins (§8.3).

| Model class | NVIDIA | AMD Instinct | Apple silicon | CPU only |
|---|---|---|---|---|
| Encoders ≤0.5B, GLiNER2 | Tier 0: ORT or candle | Tier 0: ORT MIGraphX (GLiNER2 fidelity to verify) | Tier 0: candle or ORT CoreML | Tier 0: ORT |
| Label-token deciders ≤12B | Tier 1a | Tier 1a | Tier 0: llama.cpp GGUF | Tier 0: llama.cpp, small only |
| Scoring-head hybrids 0.8B–9B | Tier 1b | Tier 1b | Not in v1 (OD-14) | Impractical above ~1B |
| MoE and ≥27B | Tier 2 | Tier 2 (27B ≈ 55 GB BF16; 35B MoE 52–70 GB) | None | None |

## 12. Security and governance

- **No `trust_remote_code` by default.** Repositories supply weights, tokenizers and manifests; executable code comes only from reviewed adapters in vllm-sr and `vllm-sr-plugins`.
- **P0 exception.** P0 runs a first-party package's bundled runtime only when its hashes are on an allowlist shipped with vllm-sr (**OD-15**). Bundled third-party code never runs.
- **Reviewed adapters.** Each family or renderer lands with golden tests and a receipt, and each overlay records its reviewer.
- **Licence flags** appear in `/v1/models`. Non-commercial and research-only models require `--accept-licence` (**OD-16**), and gated models need a token file.
- **Pinned revisions and hashes.** Every file is verified before load, and a new upstream revision counts as a new identity.
- **Exposure.** Images pin their versions and ship SBOMs. Managed engines listen only on a UDS or localhost; remote engines require TLS and a bearer key.

## 13. Existing issue: ORT's ROCm execution provider

ONNX Runtime removed its ROCm execution provider in 1.23; ROCm 7.0 was its last supported release, and ORT points users to MIGraphX. vllm-sr's AMD path still depends on that provider:

- the ROCm image installs `onnxruntime_rocm` 1.22.1 for ROCm 7.0, and checks that both the MIGraphX and ROCm provider libraries resolve;
- the `migraphx-dynamic` feature of `onnx-binding` also enables `rocm`;
- the CK flash-attention op pins the ORT 1.22.1 C header;
- `rocm:N` devices still select the ROCm provider, and only the implicit `migraphx:0` avoids it.

AMD Tier 0 is therefore frozen on ORT 1.22.1 and ROCm 7.0. The migration belongs in P1 and takes about 3–4 engineer-weeks. It is critical path for GLiNER2 / DeBERTa, which can only run in Tier 0.

1. Make MIGraphX the only AMD provider: `rocm:N` becomes a deprecated alias of `migraphx:N`, and `migraphx-dynamic` stops enabling `rocm`.
2. Move to a supported ORT release with MIGraphX, on a current ROCm.
3. Rebuild the CK op against the new header, or replace it with MIGraphX attention fusion. That fusion is disabled today because it produced non-finite mmBERT outputs on ROCm 7.0, so it needs re-qualifying.
4. Record receipts and latency for every Tier 0 model on MIGraphX.

## 14. Model-side co-design

These levers are for future Decision generations. New human-rated data is currently the main training lever, so each change should run as a preregistered, matched-control arm within a scheduled milestone.

- **Multi-question packing:** several questions follow one state, and each is read at its own slot in one pass (STEP pooling or an endpoint plugin). This removes the re-reads on hybrid pooling. Later questions see earlier ones, so training must use the packed format with shuffled order.
- **A readout stock engines can run:** label-token rows or a last-position slot head. It brings stock vLLM, prefix caching, thinking and GGUF / MLX builds, but it needs a panel A/B test first (**OD-17**).
- **An MoE tier:** Qwen3.5/3.6-35B-A3B- or Gemma-4-26B-A4B-class bases give large-tier quality at about 4B active compute, served in Tier 2. The timing follows the §3 default; the choice of base and LoRA on experts are open.
- **Small Tier 0 encoders:** encoders lead the survey's frontier up to about 0.5B. Decision 2.0 0.6B has no GDN layers, so it exports to ONNX.
- **BF16 / FP8 serving:** drop resident FP32 weights (about 103 GB, against 55 GB, at 27B) and keep only the head in FP32. FP8 on MI300-class GPUs needs conversion from OCP to fnuz, and each dtype gets its own calibration.
- **Prefix sharing:**
  - The Decision 2.0 renderer puts the state first (`Context:` … `Decision:`), but tokenizes state and question as one segment.
  - Tokenizing the state separately would guarantee identical prefixes. It changes model inputs, so it must pass the training gates.
  - Hybrid prefix caching reuses state only at block boundaries (`align` mode).
  - A LoRA per question type breaks sharing; shared-backbone heads or activated LoRA (aLoRA) keep it.
- **Packaging:** full-weight 27B packages use the lossless `bf16z` codec, which Tier 1b must load or the CLI must restore. Adapter packages pin their base by revision and hashes.

## 15. Roadmap and effort

Estimates are planning figures in engineer-weeks (ew), not measurements.

| Phase | Scope | Exit criteria | Estimate |
|---|---|---|---|
| **P0** | Engine mode wrapping the current runtime (all five endpoints, micro-batching); CLI detection and download; engine images; API v1 schema, conformance suite and docs; BF16-resident Linear weights behind a receipt | API v1 frozen; a Decision 2.0 package matches its release receipt with zero answer changes | 5–6 ew |
| **P1** | Rust core v1 (manifest, calibration, answers, Python bindings); Tier 1a label-token adapter with 2–3 renderer ports; Tier 1b plugin (27B adapter, Qwen3 0.6B, `bf16z`, compatibility CI); Tier 0 MIGraphX migration; router `decision_answers.v1`, `http_decision`, `decision` signal and selector | Receipts per backend; router E2E tests pass, including fail-open; AMD Tier 0 on a supported ORT | 16–21 ew |
| **P2** | Managed sidecar and supervisor (process, Docker, Kubernetes); in-process `RouterModelRuntime`; planner; run-time placement; cascades; multi-LoRA heads | Killing a sidecar degrades answers, never routing; one call per request | 8–12 ew |
| **P3** | Rust engine-core client after upstream pooling lands; third-party breadth (4–6 renderers, GLiNER2, marker-head encoders); MoE placement; multi-slot packing | 29 of 31 surveyed models served with receipts | 10–15 ew |

- **MVP:** P0, plus the P1 items that Decision 2.0 needs in router mode: the Rust core, Tier 1b for `qwen-full`, the label-token adapter and the router work. MIGraphX runs in parallel. About 17–22 ew, or 2–2.5 months with two engineers.
- **Full coverage (P0–P3):** about 39–54 ew (9–12.5 engineer-months), or 5–6 months with two engineers.
- P1 starts once the prototype results are reviewed (**Decided**).

## 16. Risks and open questions

### 16.1 Risks

| Risk | Mitigation |
|---|---|
| vLLM's BF16 numerics flip answers relative to the scored FP32-master path | Per-backend calibration and receipts; an FP32 option (OD-13) |
| Plugin interfaces change between vLLM releases | Pins, compatibility CI, upstream contributions |
| Hybrid pooling is lightly exercised upstream, with no prefix caching | Packing, a stock-friendly readout, upstream work |
| Large decoders miss routing deadlines and fail open too often | Latency-aware placement, cascades, smaller tiers |
| Third-party revisions drift, or licences are misused | Pinned hashes, receipts re-run on change, licence flags (OD-16) |

### 16.2 Open decisions

| ID | Question | Recommendation |
|---|---|---|
| OD-1 | Ship the `/v1/systemone` alias? | Yes, on by default |
| OD-2 | Ship the `/classify` shim? | Yes, off by default |
| OD-3 | What does `serve <model> --config` mean? | Reject it in v1 |
| OD-4 | Several states per call? | Not in v1 |
| OD-5 | Router config names | As in §4.3, pending config review |
| OD-6 | Selector planning | One Choice per decision up to three candidates, a union Choice beyond |
| OD-7 | SGLang as an alternative Tier 1 engine? | Not in v1 |
| OD-8 | Renderer language | Vendored Python for first-party packages until Rust golden tests pass; Rust for new renderers |
| OD-9 | Receipt threshold for placement | Zero answer changes for a first-party default backend; a published tolerance elsewhere |
| OD-10 | Manifest file and registry location | A separate `DECISION_MANIFEST.json`; the registry in the vllm-sr repository |
| OD-11 | 27B adapter: vLLM LoRA or an offline merge? | LoRA first; merge only with new receipts |
| OD-12 | One engine image or several? | Split: a small Tier 0 image plus vLLM-based Tier 1 images |
| OD-13 | Default Tier 1b dtype: BF16 recalibrated, or FP32 (exact, twice the memory)? | Decide from prototype parity |
| OD-14 | Apple silicon and CPU for scoring-head hybrids; a Tier 0 export of the 0.6B | Hybrids out of v1; evaluate the export in P3 |
| OD-15 | First-party bundled runtimes in P0? | Hash-allowlisted packages only |
| OD-16 | Licence defaults | Refuse non-commercial and research-only models without `--accept-licence` |
| OD-17 | A stock-engine-friendly readout for future models? | A preregistered panel A/B in the next training milestone |

Diffusion deciders are deferred rather than open, until their structured read works upstream without out-of-tree fixes.

### 16.3 Open technical questions

1. Do vLLM's processed logprobs match our temperature pipeline, including Gemma's softcapping?
2. Is hybrid `align` prefix caching stable on ROCm, and will upstream support prefix caching for hybrid pooling?
3. What does the sidecar's Python frontend cost? (**TBD: prototype results**)
4. Do concurrent requests with a shared prefix each recompute it?
5. Are community GLiNER2 / DeBERTa ONNX exports faithful under MIGraphX?
6. Does the router's candle Qwen3 multi-adapter forward skip LoRA deltas, as the research found?

## Appendix A. Ecosystem survey summary

A 2026-09-28 snapshot of 31 distinct models: the top 20 open entrants on a public decision-model leaderboard, plus its size–quality frontier. Only patterns and counts are reported.

| Pattern | Top 20 | Frontier (14) | All 31 |
|---|---|---|---|
| LM-head label-token logits, one pass | 16 | 9 | 21 |
| Last-position K-slot linear head | 2 | 0 | 2 |
| Candidate-endpoint scoring head (the Decision 1.0 / 2.0 design) | 1 | 0 | 1 |
| Diffusion answer canvas | 1 | 0 | 2 |
| Encoder with an option-marker head | 0 | 2 | 2 |
| GLiNER2 span or label extractor | 0 | 3 | 3 |
| Generative JSON | 0 | 0 | 0 |

- **Backbones (top 20):** 15 Qwen3.5/3.6/3.8 GDN hybrids (2 of them MoE), 4 Gemma 4 (1 MoE) and 1 diffusion model; 12 of the 20 have ≥25B total parameters.
- **API:** every public runtime uses a call shaped like System One. Most expose `/v1/systemone`, and at least one exposes `/v1/decisions`.
- **Amortisation:** every runtime reuses the shared state, through prefix-cached branches or packed slots. One reports label-logit reads as 7–10× faster than generating constrained JSON.
- **Engines and contracts:** vLLM `main` supports every decoder backbone studied, but DeBERTa is missing from vLLM, SGLang and llama.cpp. Several authors' runtimes are CUDA-only, option caps range from about 20 to 255, and at least one runtime silently truncates long states.
- **Licences:** mostly Apache-2.0 or MIT; a few CC BY-NC 4.0, one research-only and one gated.

## Appendix B. Glossary

- **Label-token decider:** reads the LM-head logits of label tokens at an answer position.
- **Scoring-head decoder:** a decoder with a trained head. Decision 1.0 / 2.0 score each option's endpoint against a query position.
- **GDN hybrid:** Qwen3.5-family layers that mix Gated DeltaNet linear attention with periodic full attention.
- **Parity receipt:** evidence that a model on a given backend, dtype and hardware reproduces reference answers on a named panel.

## Appendix C. References

- **vLLM docs:** [plugin system](https://github.com/vllm-project/vllm/blob/main/docs/design/plugin_system.md), [IO-processor plugins](https://github.com/vllm-project/vllm/blob/main/docs/design/io_processor_plugins.md), [endpoint plugins](https://github.com/vllm-project/vllm/blob/main/docs/design/endpoint_plugins.md), [pooling models](https://github.com/vllm-project/vllm/blob/main/docs/models/pooling_models/README.md), [multiprocessing](https://github.com/vllm-project/vllm/blob/main/docs/design/multiprocessing.md), [Rust frontend](https://github.com/vllm-project/vllm/blob/main/rust/README.md), [`vllm serve`](https://docs.vllm.ai/en/stable/cli/serve/), [environment variables](https://docs.vllm.ai/en/v0.26.0/configuration/env_vars/).
- **vLLM PRs and issues:**
  - pooling and prefix caching: [#20930](https://github.com/vllm-project/vllm/pull/20930), [#30877](https://github.com/vllm-project/vllm/pull/30877), [#50172](https://github.com/vllm-project/vllm/pull/50172), [#55307](https://github.com/vllm-project/vllm/pull/55307);
  - pooling in the generate runner, closed: [#30672](https://github.com/vllm-project/vllm/pull/30672), [#40804](https://github.com/vllm-project/vllm/pull/40804);
  - models: [#35520](https://github.com/vllm-project/vllm/pull/35520), [#42094](https://github.com/vllm-project/vllm/pull/42094), [#57250](https://github.com/vllm-project/vllm/pull/57250);
  - gRPC and the Rust frontend: [#36169](https://github.com/vllm-project/vllm/pull/36169), [#47768](https://github.com/vllm-project/vllm/issues/47768), [#54144](https://github.com/vllm-project/vllm/pull/54144), [#54648](https://github.com/vllm-project/vllm/pull/54648);
  - hidden states and LoRA: [#57185](https://github.com/vllm-project/vllm/pull/57185), [#49705](https://github.com/vllm-project/vllm/issues/49705), [#53555](https://github.com/vllm-project/vllm/pull/53555).
- **Precedents and other engines:**
  - Dynamo [#10835](https://github.com/ai-dynamo/dynamo/issues/10835) and [#10838](https://github.com/ai-dynamo/dynamo/issues/10838); [TEI](https://github.com/huggingface/text-embeddings-inference); the [Ollama runner](https://github.com/ollama/ollama/blob/da09488fbfc437c55a94bc5374b0850d935ea09f/llm/server.go);
  - ORT: the [ROCm provider notice](https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html), [execution providers](https://onnxruntime.ai/docs/execution-providers/), [#27907](https://github.com/microsoft/onnxruntime/pull/27907), and [onnxruntime-genai#2043](https://github.com/microsoft/onnxruntime-genai/pull/2043);
  - candle [#3396](https://github.com/huggingface/candle/pull/3396) and [#3424](https://github.com/huggingface/candle/pull/3424); llama.cpp [#16095](https://github.com/ggml-org/llama.cpp/pull/16095) and [#19504](https://github.com/ggml-org/llama.cpp/pull/19504); [Python multiprocessing](https://docs.python.org/3/library/multiprocessing.html).
- **Co-location research:**
  - aLoRA: [2504.12397](https://arxiv.org/abs/2504.12397), [2512.17910](https://arxiv.org/abs/2512.17910);
  - routing from activations: [2602.09924](https://arxiv.org/abs/2602.09924), [2603.20895](https://arxiv.org/abs/2603.20895), [2509.10625](https://arxiv.org/abs/2509.10625).
- **vllm-sr:** the Decision 1.0 launch post (`website/blog/2026-09-22-introduce-decision.md`), `website/docs/installation/runtime/external.md`, and the [plugin prototype](https://github.com/vllm-project/semantic-router/tree/678ac2060ea5d0b2e6d63dea347aa9ce461fb34b/src/vllm-sr-plugins).
