# Open Decision Runtime for vLLM Semantic Router

This proposal makes vLLM Semantic Router (vllm-sr) a runtime for **every model the router runs**: open-question decision models, fixed-task classifiers, embedders and rerankers. One runtime serves four API surfaces, standalone and inside the router, and embeds the model contract rather than the inference engine.

| | |
|---|---|
| Status | Draft v2 for review, 2026-10-01 (v1: 2026-09-30) |
| Scope | `vllm-sr` CLI, Go router, a new router-model runtime (Go facade, Rust core), the `vllm-sr-plugins` package, and the migration of the router's legacy model bindings |
| Markers | **Decided**: by the user. **Recommended**: coordinator recommendation awaiting confirmation. **OD-n**: open decision, all listed in §17.2. **TBD**: awaiting prototype results. |

## Changes from v1

- **Scope:** a general router-model runtime **[Decided]**; the name stays.
- **API:** four surfaces: `/v1/decisions` (+ `/v1/systemone`), `/v1/classify` (first-class), `/v1/embeddings`, `/v1/rerank`. `/v1/decisions` gains Set / Span, presets, `over`, thresholds, typed state parts, and spans / sets / thresholds / abstain / windows in responses (§4).
- **Overflow:** decision models reject by default, windowing only where declared, never silently cut; per-head policies for fixed heads; three token limits (§4.5).
- **Architecture:** the final design, with principles, seven modules, placement, plugin points and a worker protocol (§5). **Tier 1 is one vLLM sidecar tier with `generate` and `pooling` runner modes**; "Tier 1a / 1b" is gone. llama.cpp positioning added.
- **Families:** `task_heads`, `multimodal_embedding`, and the `schema_encoder` profile for Vela 2.0 (§7).
- **Renderer:** in-process encoders need the Rust renderer first (§5.10).
- **Manifest:** one schema, with Vela 2.0 Unified and Decision 2.0 examples (§6).
- **Execution plan:** three strangler-fig stages; NLI and OpenVINO retired **[Decided]** (§10).
- **Prototype:** measured BF16 parity and latency (§8.2).
- **Roadmap:** 59–78 engineer-weeks. **Open decisions:** re-numbered, 5 resolved, 7 added (§17).

## 1. Summary

Today vllm-sr reaches decision models only as HTTP classifiers with labels fixed in a mapping file. Every model it runs itself (all 40 entries in its default registry) is a fixed-task encoder, embedder or reranker, bound through four cgo bindings. This proposal defines the **Open Decision Runtime (ODR)**, one runtime for all of them:

1. **One runtime, two shells.** `vllm-sr serve <hf-model>` starts a standalone engine, as `vllm serve` does; the router runs the same code, router-managed by default.
2. **Four surfaces** share one manifest schema, planner and backend set: open questions on `/v1/decisions`, fixed heads on `/v1/classify`, plus `/v1/embeddings` and `/v1/rerank`.
3. **Embed the contract, not the engine.** The router links `RouterModelRuntime`, a Go interface over one cgo boundary into a Rust core that holds all decision logic. Backends only turn tokens into hidden states or logits, and vLLM runs as a supervised sidecar.
4. **Placement follows isolation needs, gated by parity receipts.** Small encoders run in-process on ONNX Runtime (ORT) or candle; GPU decoders run in one vLLM sidecar tier; MoE and ≥27B models run remotely.
5. **Strangler-fig migration.** Decision models first; then Vela 2.0 natively and Vela 1.0 by shadow dual-run; then every other legacy model and binding is retired.
6. **No vLLM fork:** one pinned plugin package, plus upstream contributions.

A prototype already serves Decision 2.0 0.8B on unmodified vLLM at 15 ms p50, changing about 0.6% of answers in BF16 (§8.2). The Stage 1 MVP needs about 23–29 engineer-weeks, and the whole plan 59–78 (§16).

## 2. Background

### 2.1 The System One format

A request carries a **state** and a set of named, typed **questions**. The state is text, a JSON object, an array, or (new in v2) typed parts such as user, context and answer.

| Type | Input | Answer |
|---|---|---|
| Choice | Instructions and 2–255 named options | `choice`, `probabilities`, `confidence`; `abstain_probability` if the model has an abstain option |
| Noul | Instructions, with optional `false` / `true` descriptions | `noul`, which is P(true) |
| Score | Instructions and 2–10 ordered levels | `score` (the expected level), `probabilities`, `confidence`, `legend` |
| Set (v2) | Instructions and 1–255 options, each judged independently | selected options, per-option probabilities, the threshold used |
| Span (v2) | 1–255 entity labels | spans with code-point offsets, label and probability, the threshold used |

A failed question returns its own error (`max_length_exceeded`, `invalid_question` or `invalid_model_output`). Set and Span exist only where the manifest declares them, as Vela 2.0's own server already does.

### 2.2 What the router runs today

| Consumer | Contract | Models today |
|---|---|---|
| Domain, fact-check, feedback, modality, guard, safety | `label_distribution.v1` | Vela 1.0 heads; ModernBERT / mmBERT / BERT classifiers; Qwen3Guard over chat |
| Hazard, generic classifiers | `label_scores.v1` | Vela Hazard (12 labels, sealed operating point) |
| Complexity | `score.v1` | Remote regression only |
| PII | `token_spans.v1` | Vela PII (35 BIO labels), mmBERT and BERT PII |
| Hallucination detector | `token_spans.v1` (grounded) | Vela Halu, LettuceDetect v1 / v2 |
| Hallucination explainer, response-cache polarity guard | `text_pair_distribution.v1` | ModernBERT-base-nli |
| Embedding signal, cache, memory, vector stores, RAG | `embedding.v1` | MiniLM-class default, Qwen3-Embedding, EmbeddingGemma, mmBERT / Vela Embedding, multi-modal-embed-small, Vela Omni |
| RAG reranker | `relevance_scores.v1` | Vela Reranker (one head per layer exit) |

- **Contracts and backends.** Each consumer accepts one fixed contract with no per-request options. Backends are candle, ORT, OpenVINO and HTTP; ORT accepts only `cpu`, `rocm:N` and `migraphx:N`, so local models reach NVIDIA only through candle.
- **Overflow** is `reject | truncate | window` per deployment; guard, hazard and PII scan long input in windows.
- **One forward per signal:** the AMD recipe loads ten separate 307M Vela artifacts.
- **Decision models** exist only as Python in the training tree. A package is identified by its root `config.json` pointer, `{"decision_format": "vllm-sr-decision", "format_version": 2, "package_schema": "dev2-package/1"}`, next to a `MODEL_MANIFEST.json` of per-file SHA-256.

**Goals:** one runtime and API for every router model; vLLM's kernels without a fork; third-party models with measured parity; routing that stays available; auditable answers; one legacy stack deleted. **Non-goals:** generative serving, decisions inside the served LLM's engine, diffusion deciders and training.

## 3. Decisions

| Area | Decision | Status |
|---|---|---|
| CLI | `vllm-sr serve <hf-model>` detects a model from its manifest; `--config` alone starts the router | **Decided** |
| Scope | General router-model runtime, open to third-party models through family adapters | **Decided** |
| API | `/v1/decisions`, `/v1/classify` (alias `/classify`), `/v1/embeddings`, `/v1/rerank`, `/v1/models`, `/health`, `/metrics` | **Decided** |
| API | `/v1/systemone` alias | **Recommended**, strongly (OD-1) |
| API | Set / Span, presets, `over`, thresholds, typed state parts, new response fields | **Recommended** |
| Overflow | Decision models reject by default; declared windowing; never silently cut | **Decided** |
| Overflow | Per-head policies for fixed heads; three token limits | **Recommended** |
| Router | Router-managed by default; `decision` signal and algorithms; one call per request; fail-open | **Decided** |
| Runtime | The §5 design, with Tier 1 as one sidecar tier and two runner modes | **Decided** |
| Renderer | Worker first for out-of-process models; Rust first for in-process encoders; Rust everywhere in the end | **Decided** |
| Families | `task_heads`, `multimodal_embedding`, `schema_encoder` profile | **Recommended** |
| Migration | Strangler-fig plan (§10) | **Decided** |
| Retirement | The NLI model and its features; the OpenVINO provider | **Decided** |
| Retirement | Other Stage 3 dispositions (§10.2) | **Recommended** (OD-19) |
| Sequencing | Phase 1 waits for prototype parity and latency (first results in §8.2) | **Decided** |
| MoE tier | Exploration started 2026-09-30, licence gate first | **Recommended** (OD-20) |

## 4. User experience and API

### 4.1 Engine mode

```bash
vllm-sr serve <org>/<decision-model> --revision <40-hex>              # first-party package
vllm-sr serve <org>/<vela-model> --backend ort --device cpu           # encoder, in-process backend
vllm-sr serve <org>/<third-party-decider> --manifest <overlay.yaml>   # reviewed overlay
vllm-sr serve --config config.yaml                                    # router mode, unchanged
```

Detection is deterministic:

1. Resolve a local directory or a Hub repository at `--revision`, recording the commit.
2. A root `config.json` with `decision_format: vllm-sr-decision` marks a first-party package.
3. Otherwise use the registry overlay for `repo@revision` (as for Vela 2.0), or `--manifest`.
4. With neither, exit and point to `vllm serve`; the CLI never guesses a family or enables `trust_remote_code`.
5. Verify hashes, probe hardware, place the model (§5.4) and serve the declared surfaces.

New options are `--revision`, `--manifest`, `--backend auto|ort|candle|llamacpp|vllm|worker`, `--device`, `--dtype`, `--port` or `--uds`, `--api-key-file` and `--accept-licence`.

### 4.2 Surfaces

| Surface | Serves | Router contracts |
|---|---|---|
| `POST /v1/decisions` (+ `/v1/systemone`) | Open Choice / Noul / Score, plus Set / Span where declared; inline or preset questions | `decision_answers.v1`; `label_decision.v1`; Score as `score.v1` |
| `POST /v1/classify` (alias `/classify`) | Fixed heads: sequence, multi-label scores with operating points, regression, token spans (BIO); text / pair / grounded inputs; several heads per call; window scans | `label_distribution.v1`, `label_scores.v1`, `score.v1`, `token_spans.v1`, `text_pair_distribution.v1` |
| `POST /v1/embeddings` | OpenAI-compatible, plus `dimensions`, `layer` (exit), and image / audio parts | `embedding.v1` |
| `POST /v1/rerank` | A query and documents, returning raw logits; the exit is fixed per deployment | `relevance_scores.v1` |
| `GET /v1/models`, `/health`, `/metrics` | Capability descriptor (surfaces, question types, heads, labels, limits, dimensions and exits, modalities, revision and hash, placement, receipts, licence); readiness; Prometheus metrics | None |

`/v1/classify` keeps the HF-pipeline shape for a single task, so today's `http_classify` adapters work unchanged, and a `tasks` list runs heads sharing a backbone in one forward. Signal semantics (rules, thresholds, recipes) stay in Go, which is why it is not `/v1/signals`. Every response reports its representation identity: revision, manifest hash, head, dimension and exit.

### 4.3 `/v1/decisions` v2

Router signals bind to **presets**, not to free-text questions. A preset is a question the manifest defines and maps to a contract, and inline questions still work. This request to Vela 2.0 Unified runs in one forward pass:

```json
{"model": "vela-2.0-unified",
 "state": {"parts": [{"role": "user",
   "text": "Hi, I'm Tom Baker. Paint me a watercolour of a lighthouse at sunset."}]},
 "questions": {
   "modality": {"preset": "modality"},
   "attack":   {"preset": "attack"},
   "pii":      {"preset": "pii"},
   "topics":   {"type": "set", "over": "user", "threshold": 0.4,
                "instructions": "Which topics does this request touch?",
                "criteria": {"art": "Art or design", "travel": "Travel", "money": "Payments"}},
   "urgency":  {"type": "score", "instructions": "How urgent is this request?",
                "criteria": ["Routine", "Needs prompt attention", "Critical"]}},
 "options": {"deadline_ms": 30}}
```

Response (numbers illustrative):

```json
{"model": "vela-2.0-unified",
 "identity": {"revision": "a8fb3849", "manifest_sha256": "<64-hex>",
              "backend": "ort-cpu", "precision": "fp32"},
 "answers": {
   "modality": {"type": "choice", "choice": "DIFFUSION", "confidence": 0.81,
                "probabilities": {"AR": 0.07, "DIFFUSION": 0.90, "BOTH": 0.03},
                "abstain_probability": 0.02},
   "attack":   {"type": "noul", "noul": 0.03},
   "urgency":  {"type": "score", "score": 0.12, "probabilities": {"0": 0.89, "1": 0.10, "2": 0.01},
                "confidence": 0.84, "legend": {"0": "Routine", "1": "Needs prompt attention", "2": "Critical"}}},
 "sets":  {"topics": {"selected": ["art"], "probabilities": {"art": 0.93, "travel": 0.21, "money": 0.01}}},
 "spans": {"pii": [{"label": "PERSON", "start": 8, "end": 17, "text": "Tom Baker", "probability": 0.97}]},
 "thresholds": {"topics": 0.4, "pii": 0.62},
 "input": {"tokens": 212, "sequences": 1, "windows": [], "shortened_parts": []},
 "usage": {"input_tokens": 212, "output_tokens": 0}}
```

- **State parts and `over`.** Roles come from the manifest's renderer; `over` names the part a question reads (default: all).
- **Thresholds.** A request may override a Set or Span threshold; the response always reports the one applied, including length-dependent rules.
- **Abstain.** `abstain_probability` appears only where the model has an abstain marker, flagged uncalibrated until the manifest says otherwise.
- **`question_mode: fixed_tasks`** models accept presets only.
- **Compatibility.** `/v1/decisions` is a strict superset of System One, adding `options.deadline_ms`, `options.return_meta` and the errors `deadline_exceeded`, `unavailable` and `too_many_options`. API v1 freezes at the end of P0 with a JSON Schema and a conformance suite; `vela2_serve.py` is one conformance target.

### 4.4 The other surfaces

`/v1/classify` takes `input`, an optional `text_pair` or grounded `context` / `question`, and optional `tasks`, and reports `input.windows` with the reduction used. `/v1/embeddings` adds `dimensions`, `layer` and `image` / `audio` content parts, rejecting undeclared ones. `/v1/rerank` batches one query against N documents.

### 4.5 Limits and overflow

| Model kind | Default | Declared alternatives | Always |
|---|---|---|---|
| Decision models (`question_mode: open`) | `reject` → `max_length_exceeded` | `window` with a per-type aggregation (e.g. Choice mean, Score mean, Set max, Span mean) | Never silently cut. Windows are listed in `input.windows`, and shortened unasked parts in `input.shortened_parts`. |
| Fixed-task heads (`task_heads`) | Per head, from the manifest | `reject \| truncate \| window`; a window needs a declared reduction (`max(label)`, `span_union`) | Truncation happens only where declared, and it is reported |

- **When windowing is allowed.** Only when every question in the sequence reads the same part; otherwise the planner splits questions by part, or rejects. This closes the one silent-cut case in Vela 2.0's own server.
- **Three token limits:** the **architecture** limit (32,768 for Vela 2.0's YaRN backbone), the **trained** limit (8,192 for Vela 2.0; 512 for several heads on 32K bases) and the **deployment** limit, which may exceed the trained one only with a receipt.
- **Question tokens count** against the window; if the questions alone do not fit, the request is rejected.

### 4.6 Router mode

In the default **managed** mode, `vllm-sr serve --config` starts and supervises the runtime. Until P2, that runtime is an engine-mode container on a shared Unix domain socket (UDS). In **remote** mode, the router calls a remote engine over HTTP(S). Field names below are illustrative (OD-4):

```yaml
global:
  model_catalog:
    deployments:
      decision-local: {provider: odr, artifact: <org>/<decision-model>, revision: <40-hex>, placement: auto}
      vela-unified:   {provider: odr, artifact: <org>/<vela-2.0-unified>, revision: <40-hex>, placement: auto}
routing:
  model_bindings:
    domain_classifier: {deployment: vela-unified, runtime: new,    contract: label_distribution.v1, preset: domain}
    pii_classifier:    {deployment: vela-unified, runtime: new,    contract: token_spans.v1,        preset: pii}
    prompt_guard:      {deployment: vela-guard,   runtime: legacy, contract: label_distribution.v1}
  signals:
    decisions:
      - {name: needs_reasoning, deployment: decision-local, type: noul, deadline_ms: 50,
         instructions: "Does answering this request need multi-step reasoning?"}
  decisions:
    - name: hard-reasoning
      rules: {operator: AND, conditions: [{type: decision, name: needs_reasoning, predicate: {gte: 0.7}}]}
      modelRefs: [{model: large-reasoner}, {model: medium-reasoner}]
      algorithm: {type: decision, decision: {deployment: decision-local, deadline_ms: 60,
                  instructions: "Which model should answer this request?"}}
```

| Decision point | Question | Router surface |
|---|---|---|
| Model selection | Choice over the decision's `modelRefs` | New `decision` selector; probabilities become selection scores |
| Escalation | Noul, e.g. "Does the draft answer the request?" | Gate for the `confidence` looper |
| Complexity | Score over easy / medium / hard | `decision` signal with numeric predicates |
| Existing signals | Presets on Vela 2.0, or heads on Vela 1.0 | Existing contracts, `runtime: new` per binding |

One batched call per request carries every question and head. A late or failed answer leaves its signal unknown, so the rule's `on_error` / `on_unknown` policy applies, and a failed selector falls back to the first `modelRef`. **OD-5:** since `modelRefs` are known only after a decision wins, ask one Choice per candidate decision (exact) or one over their union (cheaper)?

## 5. Runtime architecture

### 5.1 Principles

1. **Decision logic is separate from compute.** Planning, rendering, readout, calibration, answer assembly and head compute live in the Rust core, in-process. Backends only turn tokens into hidden states or logits.
2. **The in-process / out-of-process split follows isolation needs, not model names.** Stable, light, Python-free compute stays in-process. GPU-heavy, crash-prone, Python-dependent or custom-kernel compute runs in supervised workers.
3. **Placement is gated by parity receipts.** A model may run on a backend only after answer parity has been recorded for that exact model × backend × dtype × hardware combination.
4. **Engine mode and router mode share one runtime,** and so does the evaluation harness.

### 5.2 Embed the contract, not the engine

| Evidence | Consequence |
|---|---|
| vLLM's EngineCore runs in its own process by default, and parallelism above 1 spawns workers | Embedding still spawns processes |
| After CUDA initialises, vLLM forces `spawn`, which re-executes `sys.executable` | A Python interpreter must exist on disk |
| Go-to-Python calls need `runtime.LockOSThread` and `PyGILState_Ensure`; a maintained Go binding documents crashes without its fix | A fragile request path |
| An in-process CUDA fault or OOM kills the ExtProc server | All routing fails |
| NVIDIA Dynamo is moving from an in-process shim to a sidecar ([#10835](https://github.com/ai-dynamo/dynamo/issues/10835), [#10838](https://github.com/ai-dynamo/dynamo/issues/10838)); TEI uses a UDS backend; Ollama uses runner subprocesses | The established pattern |
| A UDS round trip costs about 0.01–0.2 ms, against a 15–30 ms decoder call | Little latency to save |

The nearest thing to embedding is vLLM's experimental **Rust frontend** driving a headless EngineCore; it lacks pooling so far ([#54144](https://github.com/vllm-project/vllm/pull/54144)), and we adopt it in P3. Rejected alternatives: candle or ORT for decoders (no Gated DeltaNet (GDN) support on our hardware); the served LLM's own vLLM instance (one runner type per instance; pooling-in-generate PRs [#40804](https://github.com/vllm-project/vllm/pull/40804) and [#30672](https://github.com/vllm-project/vllm/pull/30672) closed); SGLang (a second engine, no ModernBERT or DeBERTa; OD-6).

### 5.3 Seven modules

```text
            +---------------- vllm-sr router (Go, Envoy ExtProc) ----------------+
request --> | signals --> decision engine (rules) --> selector                   |
            |   all questions, heads, embeddings of a request: one bundle         |
            | RouterModelRuntime (Go facade) ===== one cgo boundary =====+        |
            +--------------------------------------------------------------|-----+
                                                                           v
 Rust core  [1 API: Decide/Classify/Embed/Rerank/Models/Health over HTTP + UDS]
            [2 planner/scheduler] [3 readout adapters] [5 registry + artifacts]
            [6 placement engine]  [7 supervisor + observability]
   |-- 4 backends, in-process: ORT (CPU, MIGraphX, CUDA, OpenVINO EP), candle, llama.cpp (CPU), Rust heads
   |-- 4 backends, out-of-process (supervised workers over UDS, worker protocol):
   |       vLLM sidecar, runner mode `generate` or `pooling` + vllm-sr-plugins;
   |       Python reference runtime; llama.cpp on GPU; model servers (stop-gap)
   '-- 4 backends, remote: shared large-model services (MoE, >=27B, judges)
 Engine mode: the same core behind a thin HTTP server.  Evaluation: the same core via Python bindings.
```

1. **API layer:** `Decide`, `Classify`, `Embed`, `Rerank`, `Models`, `Health`, over HTTP and UDS; the Go facade crosses one cgo boundary.
2. **Planner / scheduler:** grouping, batching, deadlines, admission, cascades, and state, prefix and result caches (§5.9).
3. **Readout adapters (Rust trait):** `label_token`, `head`, `encoder_marker`, `span`, `generative` (escalation only), `diffusion` (future), `task_heads`, `multimodal_embedding`. Each bundles a renderer (template → tokens and a position map), readout, calibration, answer assembly and an overflow policy; one renderer may feed several readouts in one pass.
4. **Backend interface (Rust trait):** `forward(tokens, positions | label-token ids) → hidden states | logits`, plus a capability descriptor (§5.8).
5. **Registry and artifacts:** manifests, pinning, download cache, licence and access policy, graph variants (ONNX, GGUF, BF16, lossless `bf16z`) kept separate from heads, and parity receipts.
6. **Placement engine:** hardware probe + requirements + receipts + policy (latency, cost, isolation) → backend and process class, with config overrides.
7. **Supervisor and observability:** spawn, health, back-off restarts, readiness gated on golden answers, GPU memory budgets, degraded mode; metrics, traces and shadow parity sampling.

The Go facade (names illustrative):

```go
type RouterModelRuntime interface { // one cgo boundary into the Rust core
    Load(ctx context.Context, spec ModelSpec) (ModelInfo, error)       // resolve, verify, place
    Decide(ctx context.Context, r DecideRequest) (DecideResponse, error) // /v1/decisions
    Classify(ctx context.Context, r ClassifyRequest) (ClassifyResponse, error)
    Embed(ctx context.Context, r EmbedRequest) (EmbedResponse, error)
    Rerank(ctx context.Context, r RerankRequest) (RerankResponse, error)
    Models() []ModelInfo
    Health() HealthReport
    Close() error
}
```

Each routed request makes one cgo call per surface, cancelled by the Go context. After Stage 3, the router reaches models only through this interface.

### 5.4 Placement

| Where | What | Why |
|---|---|---|
| **In-process (Rust core)** | All decision logic and head compute | Pure CPU, no GPU state, lowest latency |
| **In-process (ORT / candle)** | Encoders ≤0.5B: Vela 1.0 / 2.0, mmBERT classifiers, Kai / Lex, GLiNER2, embedders; on CPU or GPU execution providers | Stable C++ runtimes, already in-process today; a process hop costs more than it saves for small models |
| **In-process (llama.cpp / ORT GenAI, CPU)** | Small decoders on CPU and edge (0.6B / 0.8B decision models) | No GPU memory contention; GGUF quantisation gives good CPU throughput |
| **Out-of-process (supervised workers)** | GPU decoders ≥1B, GDN hybrids and multi-LoRA on vLLM; the Python reference runtime (byte-exact parity); llama.cpp on GPU; a model's own server as a stop-gap | Python and custom kernels, pre-allocated GPU memory, fault isolation: a GPU fault or OOM must not take down the router |
| **Remote** | Resident MoE and ≥27B models, judges, diffusion deciders | Memory scales with total parameters, so these are better as a shared service on large-memory GPUs |

Tier 0 is in-process, Tier 1 the local vLLM sidecar and Tier 2 remote. `/v1/models` reports the placement; a backend without a receipt needs explicit approval and shows as `unverified` (OD-7).

### 5.5 Tier 1: one vLLM sidecar tier, two runner modes

Tier 1 is a **single local vLLM sidecar tier**. Each model's manifest readout picks its runner mode:

| Runner mode | For | How it reads | Caching and escalation |
|---|---|---|---|
| `generate` | Label-token deciders, the majority of third-party decision models | LM-head logits at answer positions, with `allowed_token_ids` and logprobs (`--max-logprobs 256`) | Prefix caching (`--mamba-cache-mode align` for hybrids); optional thinking escalation over the cached prefix |
| `pooling` + `vllm-sr-plugins` | Head-based models: Decision 1.0 / 2.0 candidate heads, K-slot heads | The plugin pooler gathers hidden states at option endpoints; the head runs in FP32 | No prefix caching for hybrid pooling today, so the planner packs questions |

A vLLM engine runs one runner type and one model (plus LoRA adapters), so each model uses exactly one mode; a deployment serving both kinds runs both kinds of engine, sharing a GPU through `--gpu-memory-utilization`. The modes could converge later: a stock-engine-friendly readout for our models puts everything in `generate` (pending an A/B test, OD-15), or a label-token head in the plugin puts everything in `pooling`, losing prefix caching and thinking.

### 5.6 Why encoders stay out of vLLM

vLLM supports encoders, so this is a latency and reach choice: 8K tokens of FP32 hidden states (about 25 MB) would cross a process boundary per call for a 307M model, Vela 2.0's readout is not a stock architecture, and DeBERTa-v3 / GLiNER2 are unsupported in vLLM, SGLang and llama.cpp.

### 5.7 llama.cpp positioning

llama.cpp links **in-process** through its C API, via a Rust binding. It is recommended in-process on **CPU and edge**, and in a **supervised subprocess on GPU** for fault isolation, as Ollama does even though it links llama.cpp directly.

- **Strengths:** the widest hardware reach (x86 / ARM CPUs, CUDA, HIP, Metal, Vulkan, SYCL, CANN); the best CPU engine (GGUF quantisation); BERT, XLM-R and ModernBERT with classifier heads; Qwen3-Next / Qwen3.5 hybrids with a fused GDN op on CPU, CUDA, Metal and SYCL.
- **Weaknesses:** GDN on HIP is unoptimised (reported slow on RDNA 3.5, unmeasured on CDNA3); no DeBERTa; LoRA per request, not batched across adapters.
- **Our candidate head:** `pooling none` hidden states at the option endpoints, with the head applied in Rust.
- **Conditions:** benchmark GDN on CDNA3 (MI325X) before listing llama.cpp for AMD datacenter GPUs (OD-21). Every GGUF conversion or quantisation needs its own calibration and receipts.

### 5.8 Plugin points and the worker protocol

There are three versioned plugin points, each with compatibility CI:

1. **Model manifests (data):** most new models need only a manifest and receipts.
2. **Readout adapters (Rust trait):** only for a new readout or template family; reviewed code in our registry.
3. **Backends (Rust trait and capability descriptor):** the descriptor declares `hidden_states`, `logits` or `embedded_heads` (heads baked into the graph), the index inputs, dtypes and shape constraints. **Prefer graphs that return hidden states, with heads in Rust:** for Vela 2.0, an encoder-only ONNX graph, keeping the published graph as the parity reference.

Out-of-process backends implement a narrow **worker protocol**:

```text
Forward  {model, revision, mode: generate|pooling, want: hidden_states|logits|embedded_heads,
          sequences: [{token_ids, positions | label_token_ids, lora?}], deadline_ms}
Result   {per sequence: rows[positions x hidden] | logprobs[label ids] | head outputs,
          backend_build, dtype, timings}
Control  {health, warmup(golden_set), drain, memory_budget}
```

Any engine can be wrapped this way: vLLM (through our plugin), SGLang, a llama.cpp server, the Python reference runtime, or a model's own server.

### 5.9 Planner rules

| Rule | Detail |
|---|---|
| Grouping | By **(revision, state, phase)**; the phase separates request-time signals from response-time ones such as hallucination |
| Canonical packing | Bidirectional encoders let packed questions shift each other's scores, so bundles are ordered canonically (preset id, then question digest) and always render the same tokens |
| Cache key | The **question bundle**: representation identity, state digest, bundle digest, packing and window policy |
| Span packing | At most one span question per sequence; each extra one adds a sequence |
| Question-token budget | Schema tokens count against the window; questions that alone do not fit are rejected |
| Window planning | Only when every question reads the same part; each window carries the full schema; results aggregate by the declared rule |
| Receipts | A packed-versus-alone check: answer changes and maximum drift (OD-18) |
| Representation identity | Revision, head, dimension and exit in every cache key and response; different exits or widths never mix |
| Decoders | `generate`: shared state prefix, question suffixes. `pooling` on hybrids: packed layouts avoid state re-reads |
| Fixed heads | One forward for all heads sharing a renderer; rerank batches a query with N documents; window scans reduce by `max` or `span_union` |

For Vela 2.0, one sequence replaces about seven request-time Vela 1.0 passes (domain, guard, safety, fact-check, modality, feedback, PII).

### 5.10 Renderer placement

1. **Out-of-process models, phase 1:** the renderer runs in the worker (a Python endpoint plugin), byte-exact with training, as the prototype does.
2. **In-process encoders need the Rust renderer first,** built on HF `tokenizers` with golden tests on token ids, marker positions, word units and code-point offsets. Vela 2.0's porting hazards: Python regex lookarounds (Rust's `regex` has none), control-character whitespace, offsets, marker stripping, the window loop and float formatting.
3. **Stop-gap:** a supervised worker runs the model's own server (`vela2_serve.py`) for the Stage 2 shadow runs.
4. **Target:** Rust renders every model, and workers become generic forward servers.

### 5.11 Cascades and supervision

- **Cascades:** a question escalates when its calibrated confidence is below threshold and its remaining deadline covers the next placement's p95; in `generate` mode, a bounded thinking step plus a second label read counts.
- **Supervision:** readiness needs vLLM's health check and a warm-up reproducing golden answers; restarts back off from 1 s to 60 s; every worker and GPU context has a memory budget. Degraded questions fall back to another receipted placement, or return `unavailable` and fail open. Under Docker or Kubernetes the platform restarts workers, so the router never needs the Docker socket.

### 5.12 Latency budget

| Component | Figure | Basis |
|---|---|---|
| UDS hop | 0.01–0.2 ms | Published measurements |
| Decision 2.0 0.8B through vLLM + plugin, BF16, over HTTP | 15.2 ms p50, single request | Prototype (§8.2) |
| The same model on the shipped Python runtime | 22.7 ms p50 (FP32 master weights); 21.4 ms with Linear weights resident in BF16 | Prototype (§8.2) |
| 27B-class package | About 73 ms p50 (BF16, short prompts) | Internal measurement |
| Per-call FP32-to-BF16 recast in the shipped runtime | About 10 ms at 9B, 30 ms at 27B | Estimate; BF16-resident weights avoid it |

## 6. Manifest

### 6.1 One schema for every model

Decision models, `task_heads` models and schema encoders share one content-addressed manifest schema. The manifest is also the evaluation harness's adapter spec.

| Section | Contents |
|---|---|
| `model` | Repository, 40-hex revision, `access` (public / gated / private), per-component `licence` (weights, tokenizer, code), `remote_code` |
| `family` / `profile` | Adapter family, or a profile such as `schema_encoder` (one renderer, several readouts) |
| `question_mode`, `question_types`, `limits` | `open \| fixed_tasks`; the types served; option and label counts; `span_per_sequence` |
| `backbone` | Architecture and geometry, total and active parameters, LoRA base pin |
| `renderer` | Kind and template hash; layout; markers and their token ids; state parts; word units; tokenizer hash |
| `input` | Three token limits; overflow policy; window size, overlap and aggregation |
| `readouts` / `heads` | For decision models, readouts: kind, pooling, positions, abstain. For fixed tasks, heads: kind, labels, activation, operating point, contract, per-head overflow. |
| `calibration` | Temperatures, thresholds and rules, with **provenance**: run, the dtype of the fitted scores, and the model hash it binds to |
| `presets` | Named questions or heads and their router contract mapping |
| `runner` | Per-backend hints, such as the vLLM mode and plugin class |
| `artifacts` | Files with SHA-256. **Graph variants** (ORT graphs, GGUF, safetensors, MIGraphX caches, custom ops) are separate from **heads**. |
| `receipts` | Per backend, build, device, dtype: panel, items, answer changes, maximum drift, packing and window policy, harness version, date |

First-party packages add a `decision_manifest` entry to the root `config.json`, naming a `DECISION_MANIFEST.json` whose hash `MODEL_MANIFEST.json` records; older packages get a derived manifest. Third-party models use reviewed overlays shipped with vllm-sr (OD-8), and weights are never redistributed.

### 6.2 Worked example: Vela 2.0 Unified

This is the manifest from the Vela 2.0 check, renamed to the §6.1 fields. File digests and the revision are abbreviated here; the registry overlay carries the full values.

```yaml
manifest_version: 1
model:
  repo: llm-semantic-router/Vela-2.0-Encoder-307M-Unified
  revision: a8fb3849…                  # 40-hex in the overlay
  access: private
  licence: {weights: apache-2.0, tokenizer: gemma-terms-of-use}
  remote_code: not_required            # ORT and candle paths
profile: schema_encoder                # encoder_marker + span, one renderer, one forward
question_mode: open
question_types: [choice, noul, score, set, span]
limits: {choice: [2, 255], score: [2, 10], set: [1, 255], span: [1, 255], span_per_sequence: 1}
backbone:
  arch: modernbert                     # Vela-1.0-Encoder-307M geometry
  layers: 22
  hidden: 768
  vocab: 256008
  attention: {local_window: 128, global_every: 3}
  rope: {theta: 160000, yarn: {factor: 4.0, original_max: 8192, beta_fast: 32, beta_slow: 1, truncate: true}}
renderer:
  kind: vela2_schema
  layout: "<bos> ([Q] q ([O] name: desc)* [ABS]?)* ([E] label: desc)* [SEP_SCHEMA] ([SEG_role] part)* <eos>"
  markers: {"[Q]": 256000, "[O]": 256001, "[ABS]": 256002, "[SEP_SCHEMA]": 256003,
            "[SEG_user]": 256004, "[SEG_answer]": 256005, "[E]": 256006, "[SEG_context]": 256007}
  text_tokenizer: {file: tokenizer.json, drop_added_tokens: markers, special_tokens: false, offsets: char}
  parts: [user, context, answer]
  word_units: schema_lib_r4
input:
  limits: {architecture: 32768, trained: 8192, deployment: 8192}   # per sequence, schema included
  overflow: window                     # only when all questions read one part; otherwise reject, never cut
  window: {overlap: 512, aggregate: {choice: mean, score: mean, set: max, span: mean}}
readouts:
  - {kind: encoder_marker, types: [choice, noul, score, set], logit: cosine_plus_mlp,
     pool: {question: "[Q]", option: "[O]", state: mean_over_parts},
     abstain: {marker: "[ABS]", calibrated: false}}
  - {kind: span, unit: first_subword_of_word, label_marker: "[E]", decoder: span_decode_r4_v2}
calibration:
  provenance: {file: calibration.json, run: R4b_s44, fitted_on: bf16_dev_scores}
  temperature: {choice: 1.3786, score: 1.1278, set: 1.0152, span: 0.2927}
  thresholds: {set: 0.3, "set:r_cats": 0.5, "span:halu": 0.55, "span:toxic": 0.1, span: 0.5}
  rules:
    pii: {kind: loglin_length, anchors: "calibration.json#pii_length_rule",
          sparse_gate: {k: 3, floor: 0.1, probe: 0.5}}
  confidence: {choice: top2_margin, score: one_minus_normalised_variance}
presets:
  modality:
    type: choice
    over: user
    text: What kind of output does this request ask for?
    options:
      AR: a text answer only, including code, analysis or describing an existing image
      DIFFUSION: a newly generated image only
      BOTH: a newly generated image together with a separate written explanation
    contract: label_distribution.v1    # label order = option order
  pii: {type: span, over: user, source: "calibration.json#pii_schema", rule: pii, contract: token_spans.v1}
  halu: {type: span, over: answer, source: "calibration.json#halu_schema", contract: grounded_text.v1,
         context_template: "User request: {question}\n\n{context}"}
  # domain, attack, p_harm, factcheck, feedback, relevance: same shape
artifacts:
  graphs:
    - {file: onnx/model.onnx, sha256: 5096731c…, precision: fp32, opset: 18, heads: embedded,
       inputs: [input_ids, attention_mask, q_index, opt_index, unit_index, ent_index],
       outputs: [opt_logits, span_logits]}
    - {file: onnx/model_fp16.onnx, sha256: aaea0639…, precision: fp16_encoder_fp32_heads, heads: embedded}
  weights:
    - {file: model.safetensors, sha256: bd37dd0d…, dtype: f32}   # candle: backbone + readout tensors
  tokenizer: {file: tokenizer.json, sha256: e4b670c5…}
receipts:                              # imported from the model's parity notes: 230 dev rows, 0 decision diffs
  - {backend: ort, device: cpu, precision: fp32, max_prob_diff: {choice: 2.2e-6, span: 2.7e-5}}
  - {backend: ort, device: cpu, precision: fp16_encoder, max_prob_diff: {choice: 0.006, span: 0.050}}
```

Placement is Tier 0: ORT on CPU with the published graph (receipted), or candle with heads in Rust. Still needed: FP16 long-document PII receipts (FP16 moves span scores by up to 0.38 while the PII threshold falls to 0.0003 at 8K); MIGraphX shape buckets for the four index inputs, or the encoder-only graph; the ORT CUDA provider (§9.2); and less long-input memory (full attention masks cost about 3.2 GB of FP32 scores per layer at 8K on CPU).

### 6.3 Worked example: a Decision 2.0 package

```yaml
manifest_version: 1
model:
  repo: <org>/<decision-2.0-package>
  revision: <40-hex>
  access: public
  licence: {weights: <spdx>, code: <spdx>}
  remote_code: not_required            # the vendored runtime runs only inside the worker, hash-allowlisted
family: head
question_mode: open
question_types: [choice, noul, score]
limits: {choice: [2, 255], score: [2, 10]}
backbone: {arch: qwen3_5_hybrid, text_only: true, parameters: 753446208}
renderer:
  kind: dev2_prompt                    # phase 1: vendored Python in the worker; later Rust with golden tests
  template_sha256: <64-hex>
  layout: "Context: {state} ... Decision: {question}{options}"
input:
  limits: {architecture: <n>, trained: <n>, deployment: <n>}
  overflow: reject                     # max_length_exceeded; never truncated
readouts:
  - {kind: head, head: candidate_endpoint, positions: [option_endpoints, query], dtype: fp32}
calibration:
  provenance: {fitted_on: package_native_fp32_master_bf16_autocast, model_sha256: <64-hex>}
  temperature: {by: question_type, file: calibration.json}
  score_offsets: {file: score_bias.json}
runner:
  vllm: {mode: pooling, plugins: [vllm_sr_decision2, vllm_sr_decisions],
         architecture: Decision2Qwen3_5ForScoring, pooling_task: token_classify}
artifacts:
  graphs:
    - {variant: safetensors-bf16, files: "backbone/*.safetensors", sha256: MODEL_MANIFEST.json}
  heads:
    - {name: candidate_head, file: decision_head.safetensors, dtype: fp32}
receipts:
  - {backend: vllm-pooling, build: 0.29.1rc1.dev187-rocm, device: mi325x, dtype: bf16,
     panels: 4, items: 11053, answer_changes: 63, max_drift: 0.124}
  - {backend: python-reference, device: mi325x, dtype: bf16_resident_linear, items: 11053, answer_changes: 0}
```

## 7. Adapter families

### 7.1 Families

| Family | Readout | Typical models | Placement |
|---|---|---|---|
| `label_token` | LM-head label-token logits at answer positions (single or multi-slot) | Most third-party deciders; Qwen3Guard-style label scoring | Tier 1 `generate`; llama.cpp on CPU |
| `head` | Candidate-endpoint or K-slot head on hidden states | Decision 1.0 decoders, Decision 2.0 | Tier 1 `pooling` + plugin |
| `encoder_marker` | Option and question markers in an encoder; Choice / Noul / Score / Set; `[ABS]`; per-part pooling | Vela 2.0, Kai / Lex | Tier 0 |
| `span` | Word-level span scores against label markers; span decoder; threshold rules | Vela 2.0, GLiNER2 | Tier 0 |
| `task_heads` (new) | One backbone forward, N named fixed heads | Every Vela 1.0 artifact, LettuceDetect, every text embedder, the reranker | Tier 0; Tier 1 `pooling` for decoder embedders |
| `multimodal_embedding` (new) | Per-modality towers plus fusion | Vela Omni | Tier 0 ORT |
| `generative` | Constrained or free generation | Escalation only | Tier 1 `generate`, remote |
| `diffusion` | Denoise an answer canvas, then read labels | Two surveyed deciders | **Deferred** (OD-22) |

- **`task_heads`:** head kinds `sequence`, `scores` (multi-label, sealed operating point), `regression`, `token` (BIO, offsets), `pooled` (mean or last-token, L2, Matryoshka dimensions, exits) and `relevance` (per exit); renderers `text`, `pair` and `grounded`; overflow per head. Rerank is a `pair` renderer plus a `relevance` head, not a family.
- **`multimodal_embedding`:** image and audio processors belong to the sealed bundle and its receipts; vLLM cannot serve it.
- **Guards on `label_token`:** Qwen3Guard-style scoring gets calibrated probabilities instead of today's regular expression over chat text.

In the 31-model survey (Appendix A), the families cover 29 without a fork: 21 on stock vLLM, 5 through the plugin and 3 on ORT. Every router model kept after Stage 3 maps to `task_heads`, `multimodal_embedding` or `schema_encoder`.

### 7.2 The `schema_encoder` profile

Vela 2.0 Unified is not a fixed multi-head classifier but an **open-vocabulary schema encoder**: questions, options and labels are rendered as marker tokens, and one forward returns option scores and word-level span scores. The profile is **one renderer feeding two readouts** (`encoder_marker` and `span`), with no new family or tier:

- **Adapter interface (A3):** several readouts per sequence, with packing limits (one span question per sequence).
- **`encoder_marker` (A4):** per-part mean pooling, the `[ABS]` option (flagged uncalibrated), Set (a sigmoid per option) and per-type temperatures.
- **`span` (A4):** word units (first sub-word of each word), a span decoder with code-point offsets, and threshold rules: constant, per question, length-dependent (PII), and a sparse-document gate.
- **Backend (A5):** `embedded_heads` for the published graph, `hidden_states` for the preferred encoder-only graph.

Vela 2.0 has **no embedding output**, so embeddings and the semantic cache stay on Vela Embedding.

### 7.3 What each family means for each backend

| Family | vLLM | ORT | candle | llama.cpp |
|---|---|---|---|---|
| `label_token` | Stock `generate`, no plugin | ORT GenAI for small dense decoders on CPU | Dense Qwen3 only; no GDN | Yes: GGUF logits at the answer position; best on CPU / edge |
| `head` | `pooling` + plugin (the prototype) | Only models without GDN layers (e.g. a 0.6B export) | No GDN | `pooling none` hidden states plus a Rust head; GDN hybrids work; receipts per quantisation |
| `schema_encoder` | Possible but not worth it (non-stock readout, 8K hidden states over a process boundary) | Yes: published graph now, encoder-only graph preferred; MIGraphX needs shape buckets | Yes: the backbone exists; readout tensors in Rust | ModernBERT hidden states, to verify; no GLiNER2 or DeBERTa |
| `task_heads` | `classify`, `token_classify`, `embed` for BERT / ModernBERT (verify mmBERT 32K YaRN and ModernBERT token heads); several exits need the plugin; not preferred at ≤0.5B | Yes; today heads are baked into graphs, so each head is a graph variant until encoder-only graphs ship | Yes; already shares one backbone across heads | BERT-family heads and BERT / Gemma 3 embeddings; no DeBERTa |
| `multimodal_embedding` | No | Yes (sealed four-graph bundle) | Only the model being retired | No |

## 8. vLLM integration without forks

### 8.1 The plugin registry

`vllm-sr-plugins` registers through vLLM's documented entry points, with no vLLM file modified:

| Entry point | Provides |
|---|---|
| `vllm.general_plugins` | `Decision2Qwen3_5ForScoring`: vLLM's own Qwen3.5 text backbone plus the FP32 candidate head, with no LM head. Classes for Qwen3, Decision 1.0 and K-slot heads follow. |
| Poolers | The candidate pooler serves the built-in `token_classify` task, because Model Runner V2 rejects the `plugin` task. It gathers option-endpoint and query rows from every prefill chunk and returns one FP32 logit per candidate. |
| `vllm.endpoint_plugins` | `/v1/decisions`, with an alias route. The alias is spelled `/v1/system_one` in the prototype and will be renamed to `/v1/systemone` to match Vela 2.0 and the ecosystem (OD-1). |

The endpoint plugin exists because HTTP `/pooling` does not forward per-request `extra_kwargs`; in the target design it shrinks to a worker-protocol route. **OD-9:** serve the 27B adapter through vLLM LoRA, or merge it offline?

### 8.2 Prototype results

The prototype ran on one MI325X with a pinned vLLM `0.29.1rc1.dev187` (ROCm) and served Decision 2.0 0.8B from its package directory. These results are preliminary: a continuation worker is finishing the FP32, Vela, label-token and co-location runs.

**Parity, BF16 plugin vs stored scored predictions:**

| Panel | Items | Answer changes | Max probability drift |
|---|---|---|---|
| Public panel | 231 | 3 (also 3 at concurrency 32) | 0.022 |
| Typed release panel | 2,000 | 20 | 0.061 |
| Human-transfer panel | 6,547 | 33 | 0.031 |
| Multilingual diagnostic | 2,275 | 7 | 0.124 |

**Latency, same GPU:**

| Path | p50 (ms) | Throughput (items/s) | Peak memory |
|---|---|---|---|
| vLLM + plugin, BF16, HTTP | 15.2 | 66 / 208 / 449 / 622 at concurrency 1 / 8 / 32 / 128 | TBD |
| Shipped runtime, FP32 master weights | 22.7 | 44.1 | 2.91 GiB |
| Shipped runtime, Linear weights resident in BF16 | 21.4 | 46.8 | 2.01 GiB |
| vLLM + plugin, FP32 | TBD | TBD | TBD |

**Findings:**

- No vLLM source changes; every request answered; the alias answered identically; chunked prefill works.
- **BF16 changes about 0.6% of answers** (63 of 11,053), so `pooling`-mode BF16 needs its own calibration fit or an accepted tolerance (OD-11).
- **FP32 does not start:** vLLM's chunked GDN kernel asserts on FP32. An opt-in custom-op override (`CustomOp.register_oot`) is being measured.
- **BF16-resident weights are exact** in the shipped runtime: bit-identical on 400 items and 0 answer changes on all four panels, so the quick win is safe.
- **Still TBD:** Vela 1.0 on native ModernBERT classification, a label-token demo in `generate` mode, co-location on one GPU, p95 / p99.

### 8.3 Pinning, CI and upstream

Each engine image pins one vLLM build and plugin version, and the plugin warns on any other build. For each candidate release, CI runs the unit tests (27 today) and a GPU parity panel per family; a nightly run against `main` catches breakage early. Upstream contributions:

| Contribution | Benefit | Status (2026-10-01) |
|---|---|---|
| Per-request pooling params through `/pooling` | No endpoint plugin | Not started |
| Prefix caching for hybrid pooling | No per-question state re-reads | Disabled; generation has `align` mode |
| A classify / pooling RPC in `vllm serve --grpc` | A typed UDS transport | RFC open ([#47768](https://github.com/vllm-project/vllm/issues/47768)) |
| Pooling in the Rust frontend client | The Rust core drives EngineCore | Open ([#54144](https://github.com/vllm-project/vllm/pull/54144), [#54648](https://github.com/vllm-project/vllm/pull/54648)) |
| The `plugin` task on Model Runner V2 | Custom poolers keep their own task | Found by the prototype; not filed |

Multi-LoRA with per-adapter heads ([#53555](https://github.com/vllm-project/vllm/pull/53555)) is merged but absent from the v0.30.0 tag; it needs only the next release.

## 9. Router integration

### 9.1 CLI and packaging

| Extension point | Change |
|---|---|
| `_execute_serve` in `src/vllm-sr/cli/commands/runtime.py` | Optional positional `MODEL`; branch into engine mode before config resolution |
| `runtime_help.py`, CLI docs | Replace the "does not launch engines" text |
| Download | `huggingface_hub` at a pinned revision into `.vllm-sr/models`; tokens from a file |
| Images | A small runtime image (CPU, CUDA, ROCm) for the core and Tier 0; Tier 1 images from the vLLM images plus the plugin wheel (OD-10) |
| Lifecycle | A `decision` service name for `status`, `logs` and `stop` |

### 9.2 Router-side work list

| Item | Why | Stage |
|---|---|---|
| **One Go provider for `runtime: new`**, implementing every typed task over `RouterModelRuntime` | The per-binding switch needs one provider for all contracts | 1 (decisions), 2 (rest) |
| **`decision_answers.v1`**: answers, spans, sets, thresholds, abstain, windows, identity; a preset-to-contract table | Existing contracts cannot express per-request questions | 1 |
| `http_decision` adapter, `decision` signal and selector | Remote engines (one retry within the deadline, `MaxIdleConnsPerHost` above Go's 2); §4.6 | 1 |
| **`http_rerank` adapter** | The reranker is local-only | 2 |
| **HTTP adapters for fact-check, feedback, modality** | No HTTP path today | 2 |
| **ORT CUDA EP**: `cuda:N` in the Rust provider enum and Go validation | NVIDIA is reachable only through candle | 2 |
| **MIGraphX migration** (§14), with shape buckets or encoder-only graphs | ORT dropped ROCm; Vela 2.0's index inputs clash with static shapes | 1–2 (critical) |
| Receipts for `decision_answers.v1`; runtime shown per signal in dashboard and `status` | Principle 3; guardrail | 2 |
| `vllm-sr config migrate` with label remaps | Stage 3 | 3 |

`/api/v1/inventory/models` shows identity, placement and receipts; `routing/preview` traces show answers and latency; states are never logged.

## 10. Execution plan: strangler-fig migration

The new runtime is introduced **alongside** the legacy one, with no big-bang switch **[Decided]**.

### 10.1 Stages

| Stage | New runtime serves | Legacy runtime | Exit gate |
|---|---|---|---|
| **1. New models first** | Decision 2.0 family and Decision 1.0 decoders: engine mode, `/v1/decisions`, the router's `decision` signal and selector through `decision_answers.v1` | Every existing signal, unchanged | Decision packages match release receipts; router E2E tests, including fail-open |
| **2. Vela** | Vela 2.0 natively (A1–A8); Vela 1.0 migrated through `task_heads` manifests | Serves every binding still set to `runtime: legacy` | Per-signal parity gate before each flip |
| **3. Retire legacy** | Only Decision models, Vela 1.0 and Vela 2.0 | Deleted; the router reaches models only through `RouterModelRuntime` | Config validation fails on removed types, with migration guidance |

Stage 2 details:

- **Vela 2.0** is a new model served through presets and the `schema_encoder` profile; until the Rust renderer passes, shadow runs use a supervised `vela2_serve.py` worker.
- **Vela 1.0 is the only legacy family migrated.** Each artifact gets a single-head `task_heads` manifest, and each binding goes through **shadow dual-run** on live traffic (recording receipts), a **per-binding switch** (`runtime: new | legacy`), a **default flip** after parity, and **per-binding rollback**.
- A signal moves to one Vela 2.0 forward only after its own parity gate; Feedback and Hazard wait on model-side presets (OD-16, OD-17).

**Guardrails:** parity gates per signal, feature flags, dashboard and `status` visibility of the serving runtime, `vllm-sr config migrate`, and release notes for every removal.

### 10.2 Stage 3 disposition of non-Vela models

| Non-Vela model / path | Used by | Disposition | Status |
|---|---|---|---|
| BERT LoRA trio (`LoRABatch`), ModernBERT and mmBERT-32K classifiers, `halugate-sentinel`, `feedback-detector` | domain, fact-check, guard, feedback, modality, PII | **Replace** with Vela 1.0 heads, or one Vela 2.0 forward. Labels change (feedback 4 → 5; modality AR / DIFFUSION / BOTH → text / image / both), so `config migrate` remaps rules. | Recommended |
| LettuceDetect v1 / v2 | hallucination detector | **Replace** with Vela Halu (same grounded input) | Recommended |
| ModernBERT-base-nli | hallucination explainer, response-cache polarity guard | **Retire** the model and both features | **Decided** |
| MiniLM-class default embedder, Qwen3-Embedding, EmbeddingGemma, mmBERT 2D-Matryoshka | embedding signal, semantic cache, memory, tools, vector stores, RAG | **Replace the default** with Vela Embedding; Qwen3 / Gemma stay available through external OpenAI `/embeddings`. Stored vectors must be re-embedded. | Recommended |
| multi-modal-embed-small | multimodal cache and memory, image complexity | **Replace** with Vela Omni and re-embed (a different vector space) | Recommended |
| Qwen3Guard via `http_chat` | prompt guard | **Keep external**; a `label_token` manifest later gives probabilities | Recommended |
| candle Qwen3 multi-LoRA, candle Qwen3Guard, DeBERTa via candle `auto` | nothing wired | **Retire** | Recommended |
| candle MLP selector | the `mlp` selection algorithm | **Port** to Rust head compute or Go before the candle binding is deleted | Recommended |
| OpenVINO provider | embeddings, sequence labels | **Retire**; Intel uses ORT's OpenVINO execution provider | **Decided** |
| Chat-LLM paths (prompt selector, preference, `llm` classifiers) | selection and signals | Keep on external chat endpoints; migrate each signal to Decision models after parity | Recommended |

### 10.3 Stage 3 hazards

- **Re-embedding:** a new embedding default invalidates stored vectors in the cache, memory, vector stores and RAG, so each store needs a drain, rebuild and verify migration.
- **MLP selector:** port and receipt it before the candle binding is deleted.
- **Label remaps:** `vllm-sr config migrate` rewrites rules on changed labels and refuses to guess ambiguous mappings.
- **Removed features:** configs enabling the NLI explainer or polarity guard fail validation with guidance.
- **One backbone:** the gain from Vela 2.0 needs many signals per forward; ORT graphs have heads baked in, while candle can share a backbone.

## 11. Evaluation and parity

- **Shared harness:** one JSONL row per prompt with answers and latency, over-budget inputs counted invalid; each model runs on its author's native path and on ODR.
- **Receipts for every placement:** `compatibility-receipt/v1` extends from `label_distribution.v1` to every contract, including `decision_answers.v1` with bundle digest, packing and window policy. Vela 2.0's parity rows import as its first receipts.
- **Calibration per backend:** fits bind to model identity and dtype; a new backend or dtype needs its own fit, or a receipt showing the old fit changes no answers.
- **Golden tests:** identical token ids, positions, word units, offsets and prompt hashes from Python and Rust renderers; stored answers per backend; API conformance, including against `vela2_serve.py`.
- **Shadow parity:** in Stage 2 the supervisor samples live traffic through both runtimes and the dashboard shows gate status.
- **Benchmarks:** p50 / p95 / p99, throughput by concurrency, cold start and memory, per placement and hardware.

## 12. Hardware and backend matrix

| Backend | NVIDIA | AMD Instinct | Intel | Apple silicon | CPU |
|---|---|---|---|---|---|
| ORT (Tier 0) | CUDA EP (to add, §9.2), TensorRT | MIGraphX (ROCm EP removed in 1.23) | OpenVINO EP | CoreML (not wired; use candle Metal) | x86, ARM |
| candle (Tier 0) | CUDA | None | None | Metal | MKL / Accelerate |
| llama.cpp | CUDA (subprocess) | HIP (subprocess; GDN unbenchmarked on CDNA3) | SYCL | Metal | x86, ARM (in-process) |
| vLLM (Tier 1) | CUDA | ROCm | XPU | Experimental | x86, AArch64 |

| Model class | NVIDIA | AMD Instinct | Apple silicon | CPU only |
|---|---|---|---|---|
| Vela 1.0 / 2.0, embedders, GLiNER2 | ORT CUDA or candle | ORT MIGraphX (shape buckets or encoder-only graph) | candle Metal | ORT |
| Vela Omni | ORT CUDA | ORT MIGraphX (to verify) | ORT CPU | ORT |
| Label-token deciders ≤12B | Tier 1 `generate` | Tier 1 `generate` | llama.cpp GGUF | llama.cpp, small only |
| Head-based hybrids 0.8B–9B | Tier 1 `pooling` | Tier 1 `pooling` | Not in v2 (OD-12) | Impractical above ~1B |
| MoE and ≥27B | Tier 2 | Tier 2 (27B ≈ 55 GB BF16) | None | None |

## 13. Security and governance

- **No `trust_remote_code`:** executable code comes only from reviewed adapters in vllm-sr and `vllm-sr-plugins`. Bundled runtimes and stop-gap model servers run only in supervised workers, and only when hash-allowlisted (OD-13).
- **Licence and access per component:** weights, tokenizer and code carry separate licences (Vela 2.0's tokenizer is under the Gemma terms). Flags (non-commercial, research-only, gated, private) appear in `/v1/models` and drive policy (OD-14); private repositories need a token file, never a command-line token.
- **Pinning:** every file is verified before load; a new upstream revision is a new identity.
- **Marker injection:** renderers insert markers by token id, so user text can never create one; golden tests check this.
- **Exposure:** managed engines listen only on a UDS or localhost; remote engines require TLS and a bearer key.

## 14. Existing issue: ORT's ROCm execution provider

ONNX Runtime removed its ROCm execution provider in 1.23, yet vllm-sr's ROCm image pins `onnxruntime_rocm` 1.22.1 for ROCm 7.0, `migraphx-dynamic` also enables `rocm`, the CK flash-attention op pins the 1.22.1 header, and `rocm:N` still selects the ROCm provider. Tier 0 is the only AMD home for DeBERTa, GLiNER2 and Vela, so the migration (about 4–5 engineer-weeks) is critical path:

1. Make MIGraphX the only AMD provider, with `rocm:N` a deprecated alias.
2. Move to a supported ORT release on a current ROCm.
3. Rebuild the CK op, or re-qualify MIGraphX attention fusion (disabled today with `MIGRAPHX_MLIR_USE_SPECIFIC_OPS=~attention` after non-finite mmBERT outputs).
4. Add length buckets instead of one static shape, and padded fixed-size index tensors for Vela 2.0 (or its encoder-only graph).
5. Record receipts and latency for every Tier 0 model on MIGraphX.

## 15. Model-side co-design

- **Vela 2.0 owner asks (OD-16):** a machine-readable `presets.json`, `NO_FEEDBACK`, `[ABS]` calibration, corrected licensing text (inherited from Kai), an encoder-only ONNX graph, FP16 long-document PII receipts, and the Hazard preset wording (OD-17).
- **Packing for decoders:** several questions read at their own slots after one state removes hybrid-pooling re-reads; training must use the packed format with shuffled order.
- **A stock-engine-friendly readout** (label-token or last-slot head) brings prefix caching, thinking and GGUF / MLX builds, after a panel A/B test (OD-15).
- **An MoE tier** (35B-A3B / 26B-A4B class) for large-tier quality at about 4B active compute (OD-20).
- **BF16 / FP8 serving** with only the head in FP32, each dtype calibrated.
- **Shared-backbone Vela heads,** including an embedding head, so one forward serves a whole request.

## 16. Roadmap and effort

Planning figures in engineer-weeks (ew). They grew from v1's 39–54 ew with the four surfaces, new families, Vela 2.0, the Vela 1.0 migration and legacy deletion.

| Phase | Stage | Scope | Exit criteria | Estimate |
|---|---|---|---|---|
| **P0** | 1 | Engine mode around the current Python runtime; CLI detection and download; images; API v1 schemas for all four surfaces, conformance suite and docs; BF16-resident Linear weights behind a receipt | API v1 frozen; a Decision 2.0 package matches its release receipt | 6–7 ew |
| **P1** | 1 | Rust core v1 (manifest, calibration, answers, Python bindings); Tier 1 `pooling` (27B adapter, Qwen3, `bf16z`, compatibility CI) and `generate` (label-token adapter, 2–3 renderers); worker protocol; router `decision_answers.v1`, `http_decision`, `decision` signal and selector; MIGraphX migration in parallel | Receipts per backend; router E2E tests with fail-open; AMD Tier 0 on a supported ORT | 17–22 ew |
| **P2** | 2 | In-process `RouterModelRuntime`, planner (§5.9), placement, supervisor; Rust renderer with golden tests; `schema_encoder` (A1–A8); `task_heads` and Vela 1.0 manifests; `/v1/classify`, `/v1/embeddings`, `/v1/rerank`; `multimodal_embedding`; the single Go provider; shadow dual-run and the `runtime` switch; ORT CUDA EP; `http_rerank` and HTTP adapters | Each Vela signal passes its parity gate and flips; killing a worker degrades answers, never routing | 22–29 ew |
| **P3** | 3 + breadth | Stage 3 dispositions; `config migrate` with label remaps; re-embedding migration; MLP selector port; legacy binding deletion; Rust EngineCore client after upstream pooling lands; third-party breadth (4–6 renderers, GLiNER2); MoE placement | Legacy code deleted; 29 of 31 surveyed deciders served with receipts | 14–20 ew |

- **Stage 1 MVP (P0 + P1):** 23–29 ew, or 3–3.5 months with two engineers.
- **Stage 2 (P2):** 22–29 ew; the Rust renderer and Go provider are critical path, after MIGraphX on AMD.
- **Everything:** 59–78 ew (13.5–18 engineer-months): 7–9 months with two engineers, 5–6 with three. P1 starts once prototype results are reviewed **[Decided]**.

## 17. Risks and open decisions

### 17.1 Risks

| Risk | Mitigation |
|---|---|
| vLLM's BF16 numerics flip answers vs the scored path (about 0.6% in the prototype) | Per-backend calibration and receipts; the FP32 override (OD-11) |
| Plugin interfaces change between vLLM releases | Pins, compatibility CI, upstream contributions |
| The Rust renderer drifts from the training renderer | Golden tests on token ids, markers, word units and offsets; the model server as stop-gap |
| Packing changes schema-encoder answers | Canonical packing, bundle cache keys, packed-vs-alone receipts (OD-18) |
| Stage 3 breaks stored vectors or rules | Re-embedding migration, `config migrate`, validation errors with guidance |
| Large decoders miss routing deadlines | Latency-aware placement, cascades, smaller tiers |
| Private or restrictively licensed components are misused | Per-component licence and access flags, token files (OD-14) |

### 17.2 Open decisions

Resolved since v1: v1 OD-2 (`/classify` is first-class), v1 OD-8 (renderer path, §5.10), the Tier 1 structure, the NLI and OpenVINO dispositions, and the overflow rule. v1 OD-3 to OD-7 and OD-9 to OD-17 carry over as OD-2 to OD-15.

| ID | Question | Recommendation |
|---|---|---|
| OD-1 | Confirm the `/v1/systemone` alias | Yes, strongly recommended: Vela 2.0 already serves it. Rename the prototype's `/v1/system_one`. |
| OD-2 | What does `serve <model> --config` mean? | Reject it in v1 |
| OD-3 | Several states per call? | Not in v1 |
| OD-4 | Router config field names (`provider: odr`, `preset`, `runtime`) | As in §4.6, pending config review |
| OD-5 | Selector planning over `modelRefs` | One Choice per decision up to three candidates, a union Choice beyond |
| OD-6 | SGLang as an alternative worker engine? | Not in v1; the worker protocol keeps it possible |
| OD-7 | Receipt threshold for placement | Zero answer changes for a first-party default backend; a published tolerance elsewhere |
| OD-8 | Manifest file and registry location | `DECISION_MANIFEST.json` in first-party packages; overlays (including Vela 2.0) in the vllm-sr repository |
| OD-9 | 27B adapter: vLLM LoRA or an offline merge? | LoRA first; merge only with new receipts |
| OD-10 | One engine image or several? | Split: a small runtime image plus vLLM-based Tier 1 images |
| OD-11 | Default dtype for `pooling` mode: BF16 with its own calibration, or FP32 through the GDN override? | BF16 with a per-backend fit, if the FP32 run confirms the drift is numerical |
| OD-12 | Apple and CPU for head-based hybrids; a Tier 0 export of the 0.6B | Hybrids out of v2; evaluate the export in P3 |
| OD-13 | Which bundled runtimes and model servers may run as supervised workers? | Hash-allowlisted first-party packages and `vela2_serve.py`, Stage 2 only |
| OD-14 | Licence and access defaults | Refuse non-commercial and research-only without `--accept-licence`; private repos need a token file; report tokenizer terms separately |
| OD-15 | A stock-engine-friendly readout for future Decision models? | A preregistered panel A/B in the next training milestone |
| OD-16 | Vela 2.0 model-owner asks: `presets.json`, `NO_FEEDBACK`, `[ABS]` calibration, licensing text, encoder-only graph, FP16 long-PII receipts | Request all six; Stage 2 needs `presets.json`, `NO_FEEDBACK` and the licensing fix before its flips |
| OD-17 | Hazard presets: wait for published wording, or keep the Vela 1.0 Hazard head? | Keep the Vela 1.0 head until the presets are published and pass parity |
| OD-18 | Packing effect: what packed-vs-alone tolerance, and is canonical packing the default for router signals? | Canonical packing by default; tolerance set like OD-7, measured in Stage 2 |
| OD-19 | Confirm the Recommended Stage 3 dispositions (§10.2) | Confirm as listed |
| OD-20 | MoE tier: replace the 27B tier, or add a family member? | Decide after the exploration reports |
| OD-21 | llama.cpp for AMD datacenter GPUs | List it only after a GDN benchmark on MI325X |
| OD-22 | Long-term DeBERTa / GLiNER2 path (ORT only), and diffusion deciders | Keep DeBERTa / GLiNER2 on ORT; keep diffusion deferred until its read works upstream without out-of-tree fixes |

### 17.3 Open technical questions

1. Do vLLM's processed logprobs match our temperature pipeline, including Gemma's softcapping?
2. Is hybrid `align` prefix caching stable on ROCm?
3. What do the sidecar frontend and HTTP hop cost at p95 / p99? (**TBD: prototype results**)
4. Do ModernBERT token classification and 32K YaRN mmBERT work on the pinned vLLM?
5. Are GLiNER2 / DeBERTa ONNX exports faithful under MIGraphX?

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

Top-20 backbones: 15 Qwen3.5/3.6/3.8 GDN hybrids (2 MoE), 4 Gemma 4 (1 MoE), 1 diffusion; 12 have ≥25B total parameters. Every public runtime uses a System One-shaped call (most expose `/v1/systemone`) and reuses the shared state; at least one silently truncates long states.

## Appendix B. Glossary

- **Label-token decider:** reads LM-head logits of label tokens at an answer position.
- **Head-based decoder:** a decoder with a trained head scoring option endpoints (Decision 1.0 / 2.0).
- **Schema encoder:** an encoder reading questions, options and labels rendered as input markers, answering all in one pass.
- **Preset:** a manifest-defined question or head mapped to a router contract.
- **Parity receipt:** evidence that a model on a given backend, dtype and hardware reproduces reference answers on a named panel.

## Appendix C. References

- **vLLM docs:** [plugin system](https://github.com/vllm-project/vllm/blob/main/docs/design/plugin_system.md), [endpoint plugins](https://github.com/vllm-project/vllm/blob/main/docs/design/endpoint_plugins.md), [pooling models](https://github.com/vllm-project/vllm/blob/main/docs/models/pooling_models/README.md), [multiprocessing](https://github.com/vllm-project/vllm/blob/main/docs/design/multiprocessing.md), [Rust frontend](https://github.com/vllm-project/vllm/blob/main/rust/README.md).
- **vLLM PRs and issues:** [#30877](https://github.com/vllm-project/vllm/pull/30877), [#50172](https://github.com/vllm-project/vllm/pull/50172), [#55307](https://github.com/vllm-project/vllm/pull/55307), [#30672](https://github.com/vllm-project/vllm/pull/30672), [#40804](https://github.com/vllm-project/vllm/pull/40804), [#57250](https://github.com/vllm-project/vllm/pull/57250), [#36169](https://github.com/vllm-project/vllm/pull/36169), [#47768](https://github.com/vllm-project/vllm/issues/47768), [#54144](https://github.com/vllm-project/vllm/pull/54144), [#54648](https://github.com/vllm-project/vllm/pull/54648), [#53555](https://github.com/vllm-project/vllm/pull/53555).
- **Precedents and other engines:** Dynamo [#10835](https://github.com/ai-dynamo/dynamo/issues/10835) and [#10838](https://github.com/ai-dynamo/dynamo/issues/10838); [TEI](https://github.com/huggingface/text-embeddings-inference); the [Ollama runner](https://github.com/ollama/ollama/blob/da09488fbfc437c55a94bc5374b0850d935ea09f/llm/server.go); the ORT [ROCm provider notice](https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html) and [execution providers](https://onnxruntime.ai/docs/execution-providers/); candle [#3396](https://github.com/huggingface/candle/pull/3396); llama.cpp [#16095](https://github.com/ggml-org/llama.cpp/pull/16095) and [#19504](https://github.com/ggml-org/llama.cpp/pull/19504).
- **vllm-sr:** the Decision 1.0 launch post (`website/blog/2026-09-22-introduce-decision.md`), the Vela models tutorial (`website/docs/tutorials/global/vela-models.md`), and the plugin prototype on branch `xunzhuo/decision-2-vllm-plugin` (`src/vllm-sr-plugins/`).
