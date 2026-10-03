# vLLM Semantic Router Model Runtime

Design for the built-in model runtime of vLLM Semantic Router (vllm-sr).
Phase 1 serves Decision 2.0; later phases move Decision 1.0, Vela 1.0 and
Vela 2.0 onto the same runtime and remove the legacy native bindings.

| | |
| --- | --- |
| Status | Phase 1 in implementation |
| Domain | `src/model-runtime/` (Python package `vllm_sr_runtime`) |
| Router side | `src/semantic-router/pkg/modelservice/`, the `decision` signal and the `decision` selector |
| CLI | `vllm-sr serve <hf-model>` (engine mode); `vllm-sr serve --config ...` (router mode) |

## 1. Goals

1. **Out-of-the-box decision models.** `vllm-sr serve vllm-sr/Decision-2.0-Kai-0.6B`
   downloads the pinned package, verifies every byte, loads it with model code
   built into the runtime, and serves typed answers. The router can use the
   same runtime for routing signals and model selection without extra setup.
2. **Byte-identical by default.** The default path reproduces the runtime that
   ships inside the released Hugging Face packages bit for bit. Every faster
   path that changes numerics is an opt-in profile with a measured accuracy
   record.
3. **Pluggable.** Model families, execution engines and hardware accelerators
   are separate plugin layers. Third parties add a model family, an engine or
   an accelerator through Python entry points without patching the runtime.
4. **One contract.** The router and every client talk to the runtime through a
   checked-in OpenAPI schema. The Go client is generated from it.
5. **Universal hardware, honestly labelled.** CPU and ROCm are validated in
   Phase 1, CUDA is implemented and unit-tested but marked unvalidated, and XPU
   and MPS are plugin slots.

Non-goals: text generation, training, serving the routed LLM itself, and
running packaged remote code (`trust_remote_code`).

## 2. Decisions

These were decided before implementation and are not reopened here.

| Area | Decision |
| --- | --- |
| Architecture | A contract-first, standalone-process Python runtime, supervised by the router. Three plugin layers: model family, engine, accelerator. No Go/cgo/Rust core. |
| Protocol | HTTP/JSON with a checked-in OpenAPI schema; a Unix domain socket (UDS) between router and runtime, TCP in engine mode; a generated Go client. |
| Naming | Domain `src/model-runtime/`, package `vllm_sr_runtime`, engine mode `vllm-sr serve <hf-model>`. Router mode keeps working and the router manages the runtime's lifecycle. |
| API (Phase 1) | `POST /v1/decisions` (a strict superset of System One) with the alias `POST /v1/systemone`; `GET /v1/models`, `GET /health`, `GET /metrics`. Classify, embeddings and rerank come in Phase 3 with room reserved now. |
| Numerics | The default path is byte-identical to the released packages' runtime. Shared context, cross-request batching and max speed are opt-in profiles, each with a measured accuracy impact. |
| Router | A `decision` signal (a question and a predicate on a probability or score) and a `decision` selector (a Choice over a decision's `modelRefs`), plus the router-managed lifecycle. Fail-open: a late or failed answer leaves the signal unknown and the selector falls back. |
| Hardware | ROCm (MI300 / MI325X) and CPU validated; CUDA implemented and unit-tested, unvalidated. |
| Delivery | One PR, green CI, commits split by module, this design, parity and performance records. |

## 3. Overview

```text
                     +-------------------- vllm-sr router (Go) --------------------+
 request ----------> | signals ---> decision rules ---> selector ---> backend      |
                     |   decision signal          decision selector                 |
                     |        \___________ modelservice ___________/                |
                     |          generated client, batching per request,             |
                     |          supervisor (managed runtimes), fail-open            |
                     +------------------------|-----------------------------------+
                                              | HTTP/JSON over UDS (managed) or TCP
                                              v
 +------------------------------ vllm_sr_runtime (Python process) -------------------------+
 | api: /v1/decisions /v1/systemone /v1/models /health /metrics   (OpenAPI, checked in)     |
 | scheduler / planner: validate, render, group, micro-batch, deadlines, admission, queue   |
 |   profile hook: exact | shared_context | batching | max_speed                          |
 | model family plugin (decision2): renderer, readout (heads), answers, limits             |
 | engine plugin (native PyTorch): backbones (Qwen3, Qwen3.5 GDN), LoRA, dtype policy      |
 | accelerator plugin (cpu | cuda | rocm): detection, kernel registry, pure-torch fallback |
 | registry: HF resolution at a pinned revision, cache, manifest verification, licence     |
 | placement: device and memory budget; supervision: golden readiness, health, metrics     |
 +-----------------------------------------------------------------------------------------+
```

One runtime process serves one model in Phase 1 (one deployment = one
process). That keeps fault isolation simple: a GPU fault or an out-of-memory
error ends one runtime, never the router, and the router marks that
deployment's answers unknown until the supervisor brings it back.

## 4. Package layout

```text
src/model-runtime/
  pyproject.toml            # distribution vllm-sr-runtime; entry points for built-in plugins
  AGENTS.md                 # domain rules and ownership
  README.md
  Dockerfile                # CPU image (CI, E2E, Kubernetes)
  docs/
    design.md               # this document
    records/                # parity and performance records
  vllm_sr_runtime/
    __main__.py, cli.py     # `vllm-sr-runtime serve ...`; `vllm-sr serve <hf-model>` delegates here
    config.py               # ServeConfig: every engine-mode option in one typed object
    api/
      openapi.yaml          # THE contract (served at /openapi.yaml; the Go client is generated from it)
      app.py                # ASGI app (Starlette) and routes
      server.py             # uvicorn over TCP or UDS
      errors.py             # error codes shared by API, scheduler and families
    plugins/
      base.py               # ModelFamily, LoadedModel, Engine, EngineModel, Accelerator, Profile
      registry.py           # entry-point discovery and selection
    registry/               # model registry (artifacts, not plugins)
      resolve.py            # local directory or Hub repo at a pinned 40-hex revision, cache
      manifest.py           # pointer + MODEL_MANIFEST.json verification (inventory, SHA-256, tensor counts)
      builtin.py            # the first-party model table: repo, revision, identity, goldens
      policy.py             # licence and access policy, token handling
    scheduler/
      planner.py            # grouping, token budgets, micro-batches
      scheduler.py          # admission, queue, deadlines, worker thread, profile hook
    placement.py            # device choice and memory budget
    supervision/
      readiness.py          # warm-up and golden answers gate readiness
      metrics.py            # Prometheus metrics
    families/decision2/     # Decision 2.0 model family (first party)
      package.py            # package format: pointer, manifest, decision_config, score bias, calibration
      renderer.py           # System One question -> tokens, option endpoints, query position
      readout.py            # candidate heads (FP32)
      answers.py            # temperature, score offsets, softmax, confidence, legend
      family.py             # ModelFamily implementation
    engines/native/         # native PyTorch engine
      engine.py
      models/qwen3.py       # Qwen3 dense backbone
      models/qwen3_5.py     # Qwen3.5 Gated DeltaNet hybrid backbone
      models/lora.py        # unmerged PEFT LoRA
      weights.py            # safetensors loading, sharded checkpoints, BF16 residency
    accel/
      cpu.py, cuda.py, rocm.py
      kernels.py            # kernel registry and pure-torch reference kernels
    profiles/
      exact.py, shared_context.py, batching.py, max_speed.py
    testing/
      fixtures.py           # tiny random-weight Qwen3 / Qwen3.5 packages and a tiny tokenizer
  tests/                    # unit, contract, registry, scheduler, server and integration tests
```

Ownership inside the domain: the inference track owns
`families/decision2` numerics, `engines/native`, `accel/*`, `profiles/*` and the
GPU parity and performance records. Everything else (layout, interfaces, API,
registry, scheduler, placement, supervision, CLI and router integration) is
owned by the runtime lead. Both are reviewed in the same PR.

## 5. Plugin layers

All plugin types are abstract base classes in `vllm_sr_runtime.plugins.base`.
Built-in plugins register through the same entry points as third-party ones,
so the runtime has one discovery path.

| Entry-point group | Base class | Built-in (Phase 1) | Later |
| --- | --- | --- | --- |
| `vllm_sr_runtime.families` | `ModelFamily` | `decision2` | `decision1` (Phase 2), `vela1_task_heads`, `vela2_schema_encoder` (Phase 3), third-party families |
| `vllm_sr_runtime.engines` | `Engine` | `native` | `vllm`, `onnxruntime`, `llamacpp` |
| `vllm_sr_runtime.accelerators` | `Accelerator` | `cpu`, `cuda`, `rocm` | `xpu`, `mps` |
| `vllm_sr_runtime.profiles` | `Profile` | `exact`, `shared_context`, `batching`, `max_speed` | per-family experimental profiles |

### 5.1 ModelFamily

A family owns the task contract of a model format: how a package is
recognised and described, how a request becomes model inputs, how model
outputs become answers, and which limits apply.

```python
class ModelFamily(ABC):
    name: ClassVar[str]                    # "decision2"
    surfaces: ClassVar[frozenset[str]]     # {"decisions"}; Phase 3 adds "classify", "embeddings", "rerank"

    def detect(self, package: PackageRef) -> bool: ...          # cheap: reads the pointer only
    def verify(self, package: PackageRef) -> VerifiedPackage: ...  # inventory, hashes, identity; no weights
    def describe(self, package: VerifiedPackage) -> ModelSpec: ... # backbone, head, LoRA, dtype policy, limits
    def load(self, spec: ModelSpec, engine: EngineModel) -> LoadedModel: ...

class LoadedModel(ABC):
    info: ModelInfo                        # identity, limits, question types, licence
    def plan(self, request: DecisionRequest) -> RequestPlan: ...   # render + validate every question
    def run(self, batch: list[RenderedItem]) -> list[ItemResult]: ... # engine forward + readout
    def answer(self, plan: RequestPlan, results) -> DecisionResponse: ...
```

`ModelSpec` is the hand-off to an engine: the backbone architecture
(`model_type` and its config), the weight files, the head and LoRA tensors,
the dtype policy (for Decision 2.0: FP32 weights, BF16 autocast for the
backbone on GPU, FP32 head and residual stream), and the token budget. The
family never imports an engine; an engine never parses a package format.

### 5.2 Engine

An engine executes a `ModelSpec` on an accelerator.

```python
class Engine(ABC):
    name: ClassVar[str]                     # "native"
    def supports(self, spec: ModelSpec, accel: Accelerator) -> Support: ...
    def load(self, spec: ModelSpec, accel: Accelerator, profile: Profile) -> EngineModel: ...

class EngineModel(ABC):
    def forward(self, batch: ForwardBatch) -> ForwardOutput: ...   # token ids + gather positions -> rows
    def parameter_count(self) -> int: ...
    def memory_bytes(self) -> int: ...
    def close(self) -> None: ...
```

`ForwardBatch` carries padded token ids, the attention mask and the positions
to gather (option endpoints and the query token). `ForwardOutput` returns the
gathered hidden rows; the family's readout applies the head. A future vLLM
engine therefore returns the same rows from a pooling runner, and the readout
does not change. The native engine keeps the backbone architectures
(`engines/native/models/`) because architectures are execution code: Decision
1.0 reuses the Qwen3.5 backbone in Phase 2, and the vLLM engine brings its own.

### 5.3 Accelerator

An accelerator describes one device class and supplies kernels.

```python
class Accelerator(ABC):
    name: ClassVar[str]                     # "cpu" | "cuda" | "rocm"
    validated: ClassVar[bool]               # cpu, rocm: True; cuda: False
    def available(self) -> bool: ...
    def devices(self) -> list[DeviceInfo]: ...                     # name, memory, bf16, arch (gfx942, sm90...)
    def capabilities(self, device) -> Capabilities: ...            # bf16, graphs, triton, fla, causal_conv1d
    def kernels(self, device) -> KernelSet: ...                    # registry with pure-torch fallbacks
    def autocast(self, device, policy: DtypePolicy) -> ContextManager: ...
```

The kernel registry maps names (`rms_norm`, `causal_conv1d`,
`chunk_gated_delta_rule`, `attention`, fused element-wise kernels) to an
implementation per device, each with a pure-torch reference. Every
accelerated kernel registers the reference it must reproduce and whether it is
bit-exact; a kernel that is not bit-exact is only selectable by an opt-in
profile. Capability detection is explicit (installed packages, architecture,
dtype support), so a CPU-only CI runner exercises every reference kernel and
skips GPU kernels cleanly.

### 5.4 Profile

A profile is a numerics and scheduling policy.

```python
class Profile(ABC):
    name: ClassVar[str]
    numerics: ClassVar[Literal["exact", "approximate"]]
    def configure(self, engine_options: EngineOptions) -> EngineOptions: ...  # kernels, graphs, LoRA path
    def plan(self, scheduler: SchedulerView, pending: list[Job]) -> list[Batch]: ...  # batch formation
```

| Profile | Numerics | What it does | Phase 1 evidence |
| --- | --- | --- | --- |
| `exact` (default) | exact | The released runtime's batching: one request's questions as one padded batch (split by the forward token budget), FP32 on CPU, BF16 autocast with BF16-resident Linear weights on GPU, exact-shape graphs and the bit-exact fused kernels where they are validated | Answers equal the released packages bit for bit on the four scored panels |
| `shared_context` | approximate | Runs the shared state of a multi-question request once (tree mode, with the cache-mode fallback, the τ guard and automatic break-even) | Decision changes and accuracy deltas per size |
| `batching` | approximate | Coalesces questions from concurrent requests into shared padded batches within a bounded window | Throughput by concurrency; answer changes against `exact` |
| `max_speed` | approximate | Numerics-changing kernels (non-bit-exact fusion, tuned GEMM selection, merged LoRA) on top of the above | Latency and answer changes per size |

"Approximate" means the profile can change answers by rounding, not that it
is unmeasured: a profile ships with a record of decision changes, maximum
probability drift and accuracy on the scored panels. A request may select
`exact` at any time; it may select another profile only if the server enabled
it (`--profile`).

### 5.5 Third-party plugins

A third party ships a normal Python distribution:

```toml
[project.entry-points."vllm_sr_runtime.families"]
acme_label_token = "acme_decider.family:AcmeFamily"
```

The runtime loads entry points at startup, refuses duplicate names, and lists
every active plugin with its distribution and version in `/v1/models`. Plugins
run in the runtime process; the runtime never executes code from a model
package, so third-party execution is opt-in by installation, never by
download.

## 6. API

The contract is `vllm_sr_runtime/api/openapi.yaml` (OpenAPI 3.0.3). It is
served at `GET /openapi.yaml`, checked by contract tests on both sides, and the
Go client in `pkg/modelservice/api` is generated from it.

### 6.1 `POST /v1/decisions`

A strict superset of System One: every System One request is accepted and
every System One response field keeps its meaning. `POST /v1/systemone` is an
alias with the same body and response.

```json
{
  "model": "Decision-2.0-Kai-0.6B",
  "state": "Write a Python function that merges two sorted lists.",
  "questions": {
    "domain": {"type": "choice", "instructions": "Which domain is this request about?",
               "criteria": {"code": "Programming", "math": "Mathematics", "other": "Anything else"}},
    "needs_reasoning": {"type": "noul", "instructions": "Does this need multi-step reasoning?"},
    "difficulty": {"type": "score", "instructions": "How difficult is this request?",
                   "criteria": ["Trivial", "Moderate", "Hard"]}
  },
  "options": {"deadline_ms": 80, "profile": "exact"}
}
```

| Field | System One | Superset additions |
| --- | --- | --- |
| `model` | optional | The served model ID; required only when a runtime serves several models (Phase 3) |
| `state` | text, object or array | unchanged |
| `questions.<id>.type` | `choice`, `noul`, `score` | Phase 3 reserves `set` and `span` |
| `questions.<id>.criteria` | object (Choice, Noul) or ordered list (Score) | unchanged |
| `questions.<id>.choices` | | Ordered `[{key, description}]`, an alternative to a Choice / Noul `criteria` object for clients whose maps do not keep order |
| `questions.<id>.levels` | | Ordered level descriptions, the same as a Score `criteria` list |
| `options.deadline_ms` | | Work not started by the deadline is not run; its questions return `deadline_exceeded` |
| `options.profile` | | `exact`, or the server's enabled profile |
| `options.return_meta` | | Include `meta` (default true) |

Response:

```json
{
  "model": "Decision-2.0-Kai-0.6B",
  "answers": {
    "domain": {"type": "choice", "choice": "code", "confidence": 0.81,
               "probabilities": {"code": 0.93, "math": 0.05, "other": 0.02}},
    "needs_reasoning": {"type": "noul", "noul": 0.31},
    "difficulty": {"type": "score", "score": 0.84, "confidence": 0.42,
                   "probabilities": {"0": 0.28, "1": 0.60, "2": 0.12},
                   "legend": {"0": "Trivial", "1": "Moderate", "2": "Hard"}}
  },
  "usage": {"input_tokens": 412, "output_tokens": 0},
  "meta": {"revision": "881bee413681d80ebeac86afcda8b4138dae516e", "model_sha256": "<64-hex>",
           "profile": "exact", "engine": "native", "accelerator": "rocm", "device": "rocm:0",
           "queue_ms": 0.2, "compute_ms": 4.6}
}
```

Answers keep the order of `questions`, and probability maps keep option
order. A question that cannot be answered returns
`{"type": ..., "error": <code>}` in place and never fails its siblings:

| Code | Meaning |
| --- | --- |
| `invalid_question` | Malformed question (type, criteria, option count 2–255, level count 2–10, payload) |
| `max_length_exceeded` | The rendered question exceeds the model's input limit; nothing is ever truncated |
| `invalid_model_output` | Non-finite or malformed model output |
| `deadline_exceeded` | The request deadline passed before the question ran |
| `unavailable` | The model cannot run the question now (for example, it is being restarted) |

Request-level errors use HTTP status codes with
`{"error": {"code", "message"}}`: 400 `invalid_request`, 404 `model_not_found`,
413 `request_too_large`, 429 `overloaded` (admission refused; retryable), 503
`not_ready`.

### 6.2 `GET /v1/models`, `GET /health`, `GET /metrics`

`/v1/models` returns an OpenAI-style list whose entries describe the served
model: ID, family, repository, revision, `model_sha256`, `manifest_sha256`,
surfaces, question types, limits (`max_input_tokens`, options 2–255, levels
2–10), licence, profile and enabled profiles, engine, accelerator, device,
dtype policy, plugin versions, readiness and golden-check results.

`/health` returns 200 with `{"status": "ready", ...}` only when the model is
loaded, verified and has passed its golden check; otherwise 503 with
`starting`, `loading`, `warming`, `degraded` or `failed` and a reason.
`/health/live` returns 200 while the process serves HTTP at all.

`/metrics` is Prometheus text: request and question counters by outcome,
request, queue and forward latency histograms, batch rows and tokens, queue
depth, readiness, restarts of the model worker, and a model info series.

### 6.3 Room for Phase 3

`/v1/classify`, `/v1/embeddings` and `/v1/rerank` are reserved paths. A family
declares the surfaces it serves; the router asks `/v1/models` before routing a
contract to a deployment. Phase 3 adds the paths, schemas and families without
changing `/v1/decisions`.

## 7. Model registry

### 7.1 Resolution

`vllm-sr serve <model>` accepts a local package directory or a Hub repository
ID. A Hub model is always resolved to a 40-hex commit. With `--revision`, the
runtime downloads that revision; without it, a built-in model uses its pinned
revision from the first-party table and any other repository needs an explicit
`--revision`. Downloads use `huggingface_hub.snapshot_download` into the HF
cache (`HF_HUB_CACHE`, or `<cache>/hf` in router-managed mode) with an
allow-list derived from the package manifest. A token, when needed, comes from
the environment or the HF token file, never from a command-line argument.

### 7.2 Verification

Before any model code runs, the family verifies the package:

- the root `config.json` pointer is exactly
  `{"decision_format": "vllm-sr-decision", "format_version": 2, "package_schema": "dev2-package/1"}`;
- `MODEL_MANIFEST.json` lists every packaged file with its SHA-256; the
  on-disk inventory must equal it exactly (no extra, missing or linked files;
  Hub-added `.gitattributes` and cache metadata are ignored);
- the parameter count in each safetensors header equals the manifest;
- the model identity (`identity.model_sha256` over the scored model files and,
  for an adapter, the pinned base) equals the manifest and, for a built-in
  model, the first-party table;
- an adapter's base (`base.repo_id` at `base.revision`) is resolved and every
  base file is checked against `base.files_sha256`;
- after loading, the loaded parameter count equals `parameters.loaded`.

The runtime ignores the package's bundled Python (`decision2/`,
`modeling_decision2.py`, `pipeline_decision2.py`) and never sets
`trust_remote_code`. Those files are still hashed, because the manifest
covers them.

### 7.3 First-party models (Phase 1)

| Model | Revision | Backbone | Profile in package |
| --- | --- | --- | --- |
| `vllm-sr/Decision-2.0-Kai-0.6B` | `881bee41` | Qwen3 dense | `qwen-full` |
| `vllm-sr/Decision-2.0-Eos-0.8B` | `ad0aa724` | Qwen3.5 GDN hybrid | `qwen-full` |
| `vllm-sr/Decision-2.0-Sol-2B` | `4b75b521` | Qwen3.5 GDN hybrid | `qwen-full` |
| `vllm-sr/Decision-2.0-Nox-4B` | `ce1bdc9d` | Qwen3.5 GDN hybrid | `qwen-full` |
| `vllm-sr/Decision-2.0-Lux-9B` | `214ffa43` | Qwen3.5 GDN hybrid | `qwen-full` |
| `vllm-sr/Decision-2.0-Vega-27B` | `9b067a95` | Qwen3.8-27B + LoRA (rank 512) | `qwen-adapter` |

`registry/builtin.py` carries the full revisions, the expected
`model_sha256`, the manifest digest, the loaded parameter count and the golden
questions with their reference answers per device class. A new model revision
is a new table entry; the runtime never follows a moving branch for a built-in
model.

### 7.4 Licence and access

`/v1/models` reports the manifest's per-component licences. The policy
refuses a component marked non-commercial or research-only unless the operator
passes `--accept-licence <spdx>`; gated or private repositories need a token
from the environment or token file. The six Phase 1 models and their bases are
Apache-2.0.

## 8. Decision 2.0 family

### 8.1 Rendering

The renderer reproduces the scored prompt exactly
(`decision2-segmented-options-global-query-v1`):

```text
Context:\n<state>\n\nTask type: <type>\nQuestion:\n<instructions>\nOptions:
(\n<option>\n{"description": ..., "key": ...}\n</option>)*        # canonical JSON per option
\n\nSelect the single option best supported by the context and instructions.\nDecision:
```

Each segment is tokenized separately without special tokens; the last token
of each option segment is that option's endpoint and the last token overall is
the query position. Non-string payloads are canonical JSON (sorted keys,
compact separators, `ensure_ascii=False`, no NaN). Noul is a two-option Choice
over `false` / `true` with the default descriptions `No` / `Yes`; Score is a
Choice over the level indices `"0"`..`"L-1"`. A rendered question longer than
the model's input limit is rejected (`max_length_exceeded`), never truncated.
Tokenization uses the package's `tokenizer.json` through the `tokenizers`
library; golden tests pin the token IDs against the Transformers tokenizer the
packages were scored with.

### 8.2 Readout and answers

The candidate head (`shared`, the only head of the released models) reads the
option endpoint rows and the query row in FP32, outside autocast:

```text
c = LayerNorm(candidates), q = LayerNorm(query)
logit = <K c, Q q> / sqrt(d) + w . GELU(Mc c + Mq q)
```

Answers follow the released runtime: optional per-level Score offsets bound to
the model identity, a per-type temperature (1 for the released models),
softmax, `choice` = the arg-max key (ties resolve to the first option in
request order), `noul` = P(`true`), `score` = the expected level,
`confidence` = 1 − normalised entropy, and the Score `legend`.

### 8.3 Backbones and LoRA

The native engine implements the Qwen3 dense decoder and the Qwen3.5 hybrid
(three Gated DeltaNet layers per full-attention layer, with output gates and
partial rotary embeddings) in plain PyTorch, plus the unmerged PEFT LoRA used
by Vega-27B (FP32 factors, scaling α/r = 2). On CPU everything is FP32. On a
GPU the backbone runs under BF16 autocast with BF16-exact Linear weights held in
BF16 and every other tensor in FP32, as in the released runtime. Kernels come
from the accelerator: FLA's chunked gated-delta rule and causal-conv1d where
validated, the pure-torch references otherwise.

### 8.4 Byte-identity

Byte-identity is defined against the released runtime on the same device
class: identical token IDs, identical answer floats. The evidence is a parity
record per model and device: the four scored panels (10,653 prompts) compared
answer by answer with the released package runtime, plus a latency record.
CPU CI checks the same property on tiny random-weight fixtures against the
Transformers reference implementation, and checks the renderer and answer
assembly against golden values from the released runtime.

## 9. Scheduler and planner

```text
request -> validate (400) -> admission (429) -> plan: render every question (per-question errors)
        -> group by model and profile -> micro-batches within the forward token budget
        -> queue (deadline-ordered) -> model worker (one per device) -> readout -> answers
```

- **Validation and rendering** happen on the request thread, so malformed
  requests never take queue capacity.
- **Grouping.** In `exact`, one request's questions form one padded batch
  (padded to a multiple of 8, rows in request order), split only when the
  batch would exceed the forward token budget (the GDN kernels' 32-bit offsets
  bound q/k/v tensors to 2^30 elements). This is the released runtime's
  batching, which is what makes byte-identity possible: GEMM results depend on
  batch shape.
- **Deadlines.** A request deadline (`options.deadline_ms`, or the router's
  context deadline) travels with its jobs. A job whose deadline has passed
  when the worker picks it up is not run; its questions return
  `deadline_exceeded`. A running forward is never interrupted.
- **Admission.** The queue is bounded by jobs (`--max-queue`) and by pending
  tokens (`--max-queued-tokens`); a request beyond either bound gets 429
  `overloaded` at once, so a client can fail open without waiting.
- **Workers.** One worker thread per loaded model owns its device stream, so
  forwards on one device never interleave.
- **Profile hook.** The scheduler asks the active profile to turn pending jobs
  into batches: `exact` maps one job to its own batch, `batching` merges jobs
  inside a short window (`--batch-window-ms`, `--max-batch-tokens`),
  `shared_context` replaces a multi-question job's batch with the shared-prefix
  tree, and `max_speed` changes the engine options.

## 10. Placement and supervision

### 10.1 Placement

`--device auto` picks the first available validated GPU (`rocm`, then
`cuda`) with BF16 support and enough free memory, else `cpu`. An explicit
`--device` (`cpu`, `cuda:N`, `rocm:N`) is honoured or fails. The memory
estimate is the weight bytes under the dtype policy plus the activation bound
of the forward token budget; `--memory-budget` caps it, and the runtime refuses
to load rather than overcommit. `/v1/models` reports the placement.

### 10.2 Readiness and health

States: `starting` → `loading` (resolve, verify, load, identity) → `warming`
(warm-up and golden answers) → `ready`, or `failed` with a reason. Readiness
is gated on golden answers: every built-in model revision carries golden
questions (Choice, Noul and Score) with reference answers per device class.
On a matching device class the answers must equal the reference (bit for bit
on CPU, within the published tolerance on GPUs); without a reference the
runtime checks determinism and well-formed answers and reports
`golden: unverified`. A device error during serving marks the runtime
`degraded`; the model worker exits the process so the supervisor restarts it
with a clean device context.

### 10.3 Router-managed lifecycle

For a deployment with `provider: model_runtime` and no `endpoint`, the router
starts `vllm-sr-runtime serve <artifact> --revision <rev> --device <dev>
--profile <p> --uds <path>` as a child process when its configuration loads
and stops it when the deployment is removed or the router exits (SIGTERM, then
SIGKILL after a grace period). The supervisor polls `/health` over the UDS,
restarts a dead or failed runtime with exponential back-off (1 s to 60 s), and
records restarts in metrics. The socket lives in a private directory (mode
0700). With an `endpoint` (`unix:///path` or `http(s)://host:port`), the router
attaches to an engine it does not manage, such as a Kubernetes sidecar or a
shared engine started with `vllm-sr serve <hf-model>`.

`VLLM_SR_RUNTIME_COMMAND` overrides the command (default `vllm-sr-runtime` on
`PATH`), `VLLM_SR_RUNTIME_DIR` the socket directory and
`VLLM_SR_RUNTIME_CACHE_DIR` the model cache. The `vllm-sr` image ships the
runtime with CPU PyTorch and keeps its cache in the model volume, so managed
CPU deployments work out of the box; GPU deployments attach to a runtime built
with a GPU wheel.

Fail-open is the rule on the router side: while a deployment is not ready, a
question to it is unknown immediately (no network wait), a late answer is
unknown at the rule's timeout, and a failed selector falls back to the first
`modelRef`. Unknown signals follow the decision's `on_unknown` policy and the
condition's `on_error`.

## 11. Hardware

| Accelerator | Status | Detection | Kernels |
| --- | --- | --- | --- |
| `cpu` | validated | always | pure-torch references, FP32 |
| `rocm` | validated (MI300X, MI325X) | `torch.version.hip` | BF16 autocast, FLA gated delta, causal-conv1d, exact-shape HIP graphs with host-built masks, bit-exact fused Triton element-wise kernels (gfx942) |
| `cuda` | implemented, unit-tested, **unvalidated** | `torch.version.cuda` | same kernel slots; CUDA graphs; fused kernels only after a bit-exactness record |
| `xpu`, `mps` | plugin slots | | Phase 3+ |

CI runs on GitHub-hosted CPU runners. GPU tests are marked and skip cleanly
when no device is present; GPU evidence comes from parity and performance
records produced on an authorized MI300X / MI325X environment and committed
under `docs/records/`.

## 12. CLI

Engine mode runs one model server in the current Python environment:

```bash
pip install ./src/model-runtime          # or the vllm-sr image, or src/model-runtime/Dockerfile
vllm-sr serve vllm-sr/Decision-2.0-Kai-0.6B --device cpu --port 8100
vllm-sr serve vllm-sr/Decision-2.0-Lux-9B --device rocm:0 --profile shared_context
vllm-sr serve ./my-package --uds /run/vllm-sr/kai.sock
```

| Option | Meaning |
| --- | --- |
| `MODEL` | Hub repository or local package directory; selects engine mode |
| `--revision` | 40-hex commit (required for a non-built-in Hub model) |
| `--device` | `auto`, `cpu`, `cuda[:N]`, `rocm[:N]` |
| `--host`, `--port` / `--uds` | TCP listener (default `127.0.0.1:8100`) or a Unix socket |
| `--profile` | `exact` (default), `shared_context`, `batching`, `max_speed` |

`vllm-sr-runtime serve` takes the same options plus plugin overrides
(`--engine`, `--family`), resources and admission (`--threads`,
`--memory-budget`, `--max-queue`, `--max-batch-tokens`) and registry options
(`--cache-dir`, `--offline`, `--accept-licence`).

`vllm-sr serve` without `MODEL` keeps its router-mode behaviour. In engine
mode, `--profile` names the runtime profile; in router mode it keeps naming
the Kubernetes deployment profile. `vllm-sr serve <model>` delegates to
`vllm-sr-runtime serve` and prints an install hint when the runtime is absent.
Router-mode options are rejected in engine mode and engine options in router
mode.

## 13. Router integration

### 13.1 Configuration

```yaml
global:
  model_catalog:
    deployments:
      decision-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        revision: 881bee413681d80ebeac86afcda8b4138dae516e
        device: cpu                 # cpu | cuda:N | rocm:N | auto
        profile: exact              # exact | shared_context | batching | max_speed
        # endpoint: http://decision-kai:8100   # attach instead of manage
routing:
  signals:
    decision:
      - name: needs_reasoning
        deployment: decision-kai
        question:
          type: noul
          instructions: Does answering this request need multi-step reasoning?
        predicate: {gte: 0.7}       # on P(true); on the expected level for score
        timeout_ms: 150
      - name: request_kind
        deployment: decision-kai
        question:
          type: choice
          instructions: What kind of request is this?
          choices:
            - {key: code, description: Writing or debugging code}
            - {key: math, description: Mathematics}
            - {key: chat, description: Anything else}
  decisions:
    - name: hard_code
      rules:
        operator: AND
        conditions:
          - {type: decision, name: request_kind, label: code}
          - {type: decision, name: needs_reasoning}
        on_unknown: no_match
      modelRefs: [{model: large-coder}, {model: small-coder}]
      algorithm:
        type: decision
        decision:
          deployment: decision-kai
          instructions: Which model should answer this request?
          timeout_ms: 150
```

- **Deployment** `provider: model_runtime`: `artifact` (a Hub repository or an
  absolute package path) with an optional 40-hex `revision` (built-in models
  default to their pinned revision; other Hub models need one); `device`;
  `profile`; or `endpoint` to attach instead. Other providers reject `profile`
  and `endpoint`, and task bindings cannot name a `model_runtime` deployment.
- **Signal** `decision`: a named question to a deployment. Noul answers match
  when P(true) satisfies `predicate` (default `gte: 0.5`); Score answers match
  when the expected level satisfies `predicate` (required); Choice answers are
  label-qualified like classifier signals: `{type: decision, name, label}`
  matches the arg-max option, optionally only when its probability satisfies
  `predicate`. Every probability is published as a signal value
  (`decision:<name>` and `decision:<name>:<key>`), so conditions can also use
  their own `predicate`.
- **Selector** `decision`: a Choice whose options are the decision's
  `modelRefs` (keys are model names, descriptions come from the model cards or
  `candidates` overrides). The probabilities become the selection scores and
  the arg-max model is selected. Any failure falls back to the first
  `modelRef`, through the router's existing selection-fallback path.

### 13.2 Request flow

All `decision` signals of one request that target the same deployment are
sent as **one** `/v1/decisions` call with the request's text as the state, so
the runtime batches them together. Calls to different deployments run in
parallel with the other signal families. The selector question is asked after
a decision matches, because the candidate set is known only then. Each call
carries the smallest rule timeout as its context deadline and
`options.deadline_ms`.

### 13.3 Client, supervisor, metrics

`pkg/modelservice` owns the generated client (`api/`, from the OpenAPI file),
transports for UDS and TCP with keep-alive pools, the per-deployment state
machine (`starting`, `ready`, `unavailable`), the supervisor for managed
runtimes, and metrics: request latency and outcomes per deployment and
question type, unknown answers by reason, readiness, and restarts. The
`decision` signal lives in `pkg/classification` and the selector in
`pkg/selection` behind small interfaces, so neither depends on transport code.

## 14. Security

- No package code is executed and `trust_remote_code` is never used; model
  code is built into the runtime or installed plugins.
- Every packaged byte is verified against the manifest before load; built-in
  models are pinned by revision and identity.
- Managed runtimes listen only on a UDS inside a 0700 directory; engine mode
  binds `127.0.0.1` unless `--host` says otherwise.
- Tokens come from the environment or token files, never argv, logs or
  responses. Request text is never logged; metrics carry no request content.

## 15. Testing and CI

| Layer | Where | What |
| --- | --- | --- |
| Unit (CPU, tiny fixtures) | `src/model-runtime/tests` | renderer, answers, heads, backbones against the Transformers reference, LoRA, kernels and fallbacks, planner, scheduler, profiles, placement, plugins |
| Registry | same | pointer and manifest verification, tampered, extra, missing and linked files, header counts, identity, adapter base pinning, offline cache |
| API contract | same | every response validated against `openapi.yaml`; System One compatibility cases; alias equality; error codes; UDS and TCP |
| Integration | same | `vllm-sr-runtime serve` on a generated fixture package, readiness gating, deadlines, overload |
| Router | `pkg/modelservice`, `pkg/classification`, `pkg/selection`, `pkg/config` | generated client against an httptest server, supervisor with a fake runtime binary, signal and selector semantics, fail-open, config validation |
| E2E | `e2e/profiles/decision-runtime` | Kind, CPU: the router and the runtime image with a tiny fixture; routed requests prove the signal and the selector, a dead deployment proves fail-open |
| GPU | records under `docs/records/` | parity on the four scored panels and latency per size on ROCm |

The `model-runtime` domain is registered in `tools/agent/domains.yaml` with
`make model-runtime-test` and `make model-runtime-client-check` as its checks,
its own CI verification on the model-tools worker (CPU PyTorch), and the
generated-contracts check, which fails when the router's generated client
drifts from `openapi.yaml`. CI builds the runtime image as a provider fixture
for the E2E profile.

## 16. Phases 2–4

| Phase | Work | What the Phase 1 interfaces already provide |
| --- | --- | --- |
| 2. Decision 1.0 | A `decision1` family (its prompt version and K-slot / endpoint heads), reusing the Qwen3.5 backbone and the scheduler; supersedes the separate Decision 1.0 runtime proposal (#4086) | Family plugin, backbones in the engine, profiles, registry table |
| 3. Vela 1.0 and 2.0 | `vela1_task_heads` and `vela2_schema_encoder` families; `/v1/classify`, `/v1/embeddings`, `/v1/rerank`; the router's existing signals switch per binding (`runtime: new` or `runtime: legacy`) after shadow parity | Surfaces in `ModelFamily`, reserved API paths, `/v1/models` capabilities, router `modelservice` |
| 3. Engines | `vllm` (pooling runner plus head readout), `onnxruntime`, `llamacpp` | `Engine` returns gathered rows; readout stays in the family |
| 3. Hardware | `xpu`, `mps` accelerators; CUDA validation | `Accelerator` and the kernel registry |
| 4. Legacy removal | Delete `candle-binding`, `onnx-binding`, `openvino-binding`, `ml-binding`, `nlp-binding` once every consumer reads through `modelservice` | One router-side client and lifecycle |

## 17. Risks

| Risk | Mitigation |
| --- | --- |
| A ported backbone drifts from the scored numerics | CPU tests against the Transformers reference on fixtures; GPU parity records on the released models before a profile is called exact |
| Kernel or library updates change bits | Pinned versions per image; parity re-run on every image change; kernels declare bit-exactness |
| A runtime crash or GPU fault | Separate process, supervisor restarts with back-off, fail-open in the router |
| Large models outgrow a device | Placement refuses to overcommit; Vega-27B needs one ≥ 64 GB device |
| Tokenizer drift between `tokenizers` and Transformers | Golden token-ID tests |
| Deadlines too tight for big models | Per-rule timeouts, `on_unknown`, metrics on unknown answers |
