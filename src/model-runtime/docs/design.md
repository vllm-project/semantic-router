# vLLM Semantic Router Model Runtime

Design for the built-in model runtime of vLLM Semantic Router (vllm-sr). The
runtime serves every model the router runs: Decision 2.0 and 1.0, Vela 1.0 and
2.0, and third-party models that a family plugin understands. Phase 1 served
Decision 2.0. Phases 2–4 move Decision 1.0 and Vela 1.0 onto the runtime, add
Vela 2.0 and the classify, embeddings and rerank surfaces, and delete the
legacy native bindings.

| | |
| --- | --- |
| Status | Phase 1 merged ([#4481](https://github.com/vllm-project/semantic-router/pull/4481)); Phases 2–4 in implementation ([#4496](https://github.com/vllm-project/semantic-router/issues/4496)) |
| Domain | `src/model-runtime/` (Python package `vllm_srun`) |
| Router side | `src/semantic-router/pkg/modelservice/` (client, bundles, lifecycle) and `pkg/modelruntime/serving/` (typed task bindings) |
| CLI | `vllm-sr serve ARTIFACT --engine` (managed Engine instance); `vllm-srun serve` (direct worker); `vllm-sr serve --config ...` (Router mode) |

This document records the original phased design. For current startup commands,
replica placement and API publication, use the [deployment guide](../../../website/docs/model-runtime/deploy.md)
and [managed instance guide](../../vllm-sr/INSTANCE_MODES.md). The managed frontend
now supervises a separate worker for each replica of a logical deployment.

## 1. Goals

1. **Out-of-the-box router models.** `vllm-sr serve vllm-sr/Decision-2.0-Kai-0.6B --engine`
   or a direct worker command such as `vllm-srun serve vllm-sr/Vela-1.0-Encoder-307M-Domain` downloads the
   pinned package, verifies every byte, loads it with model code built into
   the runtime, and serves typed answers. The router uses the same runtime for
   every signal, cache, store and selector that needs a model.
2. **Byte-identical by default where a release defines the numerics.** The
   Decision 2.0 default path reproduces the runtime inside the released
   packages bit for bit. Migrated models (Decision 1.0, Vela 1.0) carry parity
   records against the path that served them before. Every faster path that
   changes numerics is an opt-in profile with a measured accuracy record.
3. **Pluggable.** Model families, execution engines and hardware accelerators
   are separate plugin layers. Third parties add a family, an engine or an
   accelerator through Python entry points without patching the runtime.
4. **One contract.** The router and every client talk to the runtime through a
   checked-in OpenAPI schema with five surfaces: decisions, classify,
   embeddings, rerank and bundle. The Go client is generated from it.
5. **One call per request.** A runtime process can serve several models, and
   the router sends all of a request's model work for one runtime as one
   bundled call.
6. **Universal hardware, honestly labelled.** CPU and ROCm are validated, CUDA
   is implemented and unit-tested but marked unvalidated, and XPU and MPS are
   plugins marked unvalidated.
7. **One stack.** After Phase 4 the router has no native model bindings: the Go
   router is free of cgo model code, and every model runs in the runtime.

Non-goals: text generation, training, serving the routed LLM itself, and
running packaged remote code (`trust_remote_code`).

## 2. Decisions

### 2.1 Phase 1

| Area | Decision |
| --- | --- |
| Architecture | A contract-first, standalone-process Python runtime, supervised by the router. Three plugin layers: model family, engine, accelerator. No Go/cgo/Rust core. |
| Protocol | HTTP/JSON with a checked-in OpenAPI schema; a Unix domain socket (UDS) between router and runtime, TCP in engine mode; a generated Go client. |
| Naming | Domain `src/model-runtime/`, distribution and command `vllm-srun`, package `vllm_srun`, engine mode `vllm-sr serve <hf-model>`. The distribution ships in the router images and is not published to PyPI. Router mode keeps working and the router manages the runtime's lifecycle. |
| Numerics | The default path is byte-identical to the released packages' runtime. Shared context, cross-request batching and max speed are opt-in profiles, each with a measured accuracy impact. |
| Router | A `decision` signal and a `decision` selector, plus the router-managed lifecycle. Fail-open: a late or failed answer leaves the signal unknown and the selector falls back. |
| Hardware | ROCm (MI300 / MI325X) and CPU validated; CUDA implemented and unit-tested, unvalidated. |

### 2.2 Phases 2–4

| Area | Decision |
| --- | --- |
| Delivery | Phases 2, 3 and 4 in one PR ([#4496](https://github.com/vllm-project/semantic-router/issues/4496)), commits split by module. |
| Families | `decision1` (Decision 1.0), `task_heads` (Vela 1.0 and compatible HF encoder task models), `vela2` (Vela 2.0), `multimodal_embedding` (Vela 1.0 Omni). `decision2` is unchanged. |
| Surfaces | `POST /v1/classify`, `/v1/embeddings` (OpenAI-compatible), `/v1/rerank`, and `/v1/bundle`; Set and Span answers on `/v1/decisions` where a model declares them. The Phase 1 surfaces are unchanged. |
| Processes | One runtime process serves one or more models. Each model keeps its own worker, device and profile. The router groups its managed deployments into processes (by default one per device). |
| Bundling | The router sends all model work of one request stage for one runtime process as one `/v1/bundle` call, and every question the stage asks one model as one decisions task, about one state or several (`states`). Bundling is transport only: every task keeps the semantics of its own surface. |
| Engines | The native PyTorch engine first, for encoders and decoders, on every accelerator; every built-in model runs on it by default. An optional `onnxruntime` engine (the `onnx` extra, not in the router images) runs the ONNX graphs packages ship (Vela 1.0, Vela 2.0 0.3B) and prepared Omni bundles, for portability. |
| Hardware | Accelerators `cpu`, `cuda`, `rocm` (built in) plus `xpu` and `mps` (built-in plugin slots, unvalidated). |
| Router seam | The router's typed task bindings (`pkg/modelruntime/binding`, contracts `label_distribution.v1`, `label_scores.v1`, `token_spans.v1`, `embedding.v1`, `relevance_scores.v1`) are kept. The provider facade `pkg/modelruntime/native` (candle, ORT, OpenVINO) is replaced by `pkg/modelruntime/serving`, which implements every contract through the runtime. Consumers change their constructor, not their logic. |
| Algorithms | Keyword matching (BM25, n-gram) and the KNN, KMeans, SVM and MLP model selectors are algorithms, not models: they become pure Go with parity tests against the bindings they replace. |
| Config | The canonical layout stays. `provider: model_runtime` deployments may serve task bindings. The providers `candle`, `ort` and `openvino` and their execution fields are removed from the parser; `vllm-sr config migrate` rewrites them. |
| Retirements | The NLI model (hallucination explainer, response-cache polarity guard) and the OpenVINO provider retire, as decided in the runtime proposal. Every other legacy model migrates or has a documented replacement (section 16). |
| Exactness | Decision 2.0 `exact` stays byte-identical. Decision 1.0 `exact` targets bit-identity with the packages' bundled runtime on the same device class. Vela 1.0 `exact` is FP32 on every device, with parity records against the legacy path. |
| Images | Every router image ships the runtime (CPU PyTorch in CPU images, the ROCm or CUDA wheel in GPU images), so the managed lifecycle works everywhere. The Rust build stages are deleted. |

## 3. Overview

```text
                 +------------------------- vllm-sr router (Go, no cgo models) ------------------------+
 request ------> | signals (domain, guard, PII, ..., decision) --> decision rules --> selector --> backend|
                 | caches, memory, vector stores, RAG, tools, hallucination, model selection            |
                 |   pkg/modelruntime/serving: typed task bindings (label distribution, scores, spans,  |
                 |   embeddings, relevance) over pkg/modelservice                                       |
                 |   pkg/modelservice: generated client, request bundles, managed process groups,       |
                 |   attached endpoints, per-deployment readiness, fail-open                            |
                 +---------------------------------|---------------------------------------------------+
                                                   | HTTP/JSON over UDS (managed) or TCP (attached)
                                                   v
 +------------------------------ vllm_srun (one Python process, one or more models) ---------------+
 | api: /v1/decisions /v1/systemone /v1/classify /v1/embeddings /v1/rerank /v1/bundle /v1/models /health |
 | per model: scheduler (admission, deadlines, profile hook, micro-batches) and one worker on its device |
 | families: decision2 | decision1 | task_heads | vela2 | multimodal_embedding  (render, readout, answers)|
 | engines: native (PyTorch: Qwen3, Qwen3.5, ModernBERT, BERT, LoRA, Omni towers) | onnxruntime (optional)|
 | accelerators: cpu | cuda | rocm | xpu | mps  (detection, kernels, pure-torch fallbacks)               |
 | registry: pinned Hub revisions, cache, manifest / file-hash verification, licence and access policy  |
 | placement per model; supervision: golden readiness per model, health, metrics                         |
 +--------------------------------------------------------------------------------------------------------+
```

A runtime process is a fault domain: a GPU fault or an out-of-memory error
ends one process, never the router, and the router marks that process's
deployments unknown until the supervisor brings it back. Models that must not
share a fault domain go to separate processes (section 13.4).

## 4. Package layout

```text
src/model-runtime/
  pyproject.toml            # distribution vllm-srun; entry points for built-in plugins
  AGENTS.md, README.md
  Dockerfile                # CPU image (CI, E2E, Kubernetes)
  docs/
    design.md               # this document
    records/                # parity and performance records
  vllm_srun/
    __main__.py, cli.py     # `vllm-srun serve ...`; `vllm-sr serve <hf-model>` delegates here
    config.py               # ServeConfig (process options) and ModelConfig (one served model)
    runtime.py              # the process: models by served name, surface dispatch, bundles
    api/
      openapi.yaml          # THE contract (served at /openapi.yaml; the Go client is generated from it)
      app.py, server.py     # Starlette routes; uvicorn (uvloop, httptools) over TCP or UDS
    errors.py               # error codes shared by API, scheduler and families
    systemone.py            # System One request semantics shared by every decisions family
    text/                   # shared text handling: the package tokenizer, input policies and windows
    heads/                  # shared readout heads (candidate, sequence, scores, token, pooled, relevance, span)
    plugins/
      base.py               # ModelFamily, LoadedModel, Engine, EngineModel, Accelerator, Profile, specs
      decisions.py          # DecisionModel: the decisions surface, golden comparison and outcomes per question
      registry.py           # entry-point discovery and selection
    registry/
      resolve.py            # local directory or Hub repo at a pinned 40-hex revision, cache
      artifacts.py          # file inventories and hash verification for packages without a manifest
      builtin.py            # the first-party models: the tables the installed families name
      tables/               # the built-in families' tables: decision2.py, decision1.py, vela1.py, vela2.py, omni.py
      golden_answers*.json  # golden references per family, per device class
      kernel_choices.json   # pinned autotuned kernel choices (Decision 2.0)
      policy.py             # licence and access policy, token handling
    scheduler/              # planner, scheduler (one per model)
    placement.py            # device choice and memory budget
    devices.py              # `vllm-srun devices`: the host's devices and the one auto takes
    supervision/            # readiness (golden answers), metrics
    families/               # package formats, rendering and answer assembly only
      decision2/            # Decision 2.0 (Phase 1)
      decision1/            # Decision 1.0: vela-encoder and qwen3.5-decision runtimes
      task_heads/           # HF encoder task models (Vela 1.0 and compatible ModernBERT models)
      vela2/                # Vela 2.0 schema encoder (0.3B) and Qwen3.5 encoders (0.8B, 4B, 9B); Set and Span
      multimodal_embedding/ # Vela 1.0 Omni: the published package, readouts, processors; ONNX bundles
    engines/
      native/               # PyTorch: models/qwen3.py, qwen3_5.py, modernbert.py, bert.py, siglip.py,
                            #   whisper.py, clap.py, lora.py, tree.py
      onnxruntime/          # ONNX Runtime sessions, execution providers, graph outputs
    accel/                  # cpu, cuda, rocm, xpu, mps; kernels.py
    profiles/               # exact, shared_context, batching, max_speed
    testing/
      fixtures.py           # tiny random-weight packages through each family's writer, and a tiny tokenizer
      <family>.py           # the built-in families' fixture writers
  examples/third_party_plugin/  # a documented out-of-tree family, engine, accelerator and profile, installed by the tests
  tests/                    # unit, contract, registry, scheduler, server and integration tests
  tools/                    # golden answers, parity drivers (GPU, legacy), benchmarks
```

Code shared by more than one family lives in `systemone.py`, `text/`,
`heads/` and `engines/native/models/`; a family composes those blocks and
never imports another family.

## 5. Plugin layers

All plugin types are abstract base classes in `vllm_srun.plugins.base`.
Built-in plugins register through the same entry points as third-party ones,
so the runtime has one discovery path.

| Entry-point group | Base class | Built-in |
| --- | --- | --- |
| `vllm_srun.families` | `ModelFamily` | `decision2`, `decision1`, `task_heads`, `vela2`, `multimodal_embedding` |
| `vllm_srun.engines` | `Engine` | `native`, `onnxruntime` |
| `vllm_srun.accelerators` | `Accelerator` | `cpu`, `cuda`, `rocm`, `xpu`, `mps` |
| `vllm_srun.profiles` | `Profile` | `exact`, `shared_context`, `batching`, `max_speed` |

Every plugin class has a capability descriptor (`descriptor()`): a family's
surfaces and package formats, an engine's architectures and outputs, an
accelerator's validation status, a profile's numerics; engines and
accelerators also list their `auto_priority`. `/v1/models` lists each active
plugin with its distribution, version and descriptor, so a card shows why
`auto` chose its engine and device.

Everything else the runtime needs from a plugin is declared on its class, so
no central list names a built-in: a family names its table of pinned models
(`ModelFamily.builtin_table`, section 7.3) and the module that writes its tiny
test packages (`ModelFamily.fixture_writer`); an engine and an accelerator
declare where `auto` tries them (`auto_priority`). The runtime reads these
when it first needs them, at load time, never per request. A new built-in
family is its own package plus an entry point in `pyproject.toml`, exactly
like a third-party one.

### 5.1 ModelFamily

A family owns the task contract of a model format: how a package is
recognised and verified, which surfaces it serves, how a request becomes model
inputs, how model outputs become answers, and which limits apply.

```python
class ModelFamily(ABC):
    name: ClassVar[str]                    # "task_heads"
    surfaces: ClassVar[frozenset[str]]     # the surfaces any model of the family may serve

    def detect(self, package: PackageRef) -> bool: ...             # cheap: reads the pointer or config only
    def verify(self, package: PackageRef) -> VerifiedPackage: ...  # inventory, hashes, identity; no weights
    def fetch(self, package: PackageRef) -> PackageRef: ...         # download what it loads (no manifest)
    def describe(self, package: VerifiedPackage) -> ModelSpec: ...  # model options: RegistryOptions.model_options
    def load(self, package, spec, engine_model) -> LoadedModel: ...
    def golden(self, package) -> list[GoldenCase]: ...              # per-surface golden requests
    builtin_table: ClassVar[str | None]    # module whose MODELS lists the family's pinned models
    fixture_writer: ClassVar[str | None]   # module with write_fixture(output, variant, seed) and VARIANTS
```

A loaded model plans, runs and finishes requests per surface. The scheduler
only sees work items with token IDs, so every family uses the same admission,
deadlines and profiles:

```python
class LoadedModel(ABC, Generic[ItemT, ResultT]):  # ItemT satisfies WorkItem
    info: ModelInfo                        # identity, surfaces, heads, limits, licence
    fuse_bundled_jobs: ClassVar[bool]      # exact may run one bundle's jobs as one batch
    def plan_surface(self, surface: str, request: SurfaceRequest) -> SurfacePlan[ItemT]: ...  # validate + render
    def run(self, items: list[ItemT]) -> list[ResultT]: ...                         # one forward + readout
    def run_shared(self, items: list[ItemT], shared_prefix: int) -> list[ResultT]: ...  # decoders' shared-context batches
    def finish_surface(self, plan: SurfacePlan[ItemT], results: Results[ResultT]) -> dict: ...  # the response body
    def golden_values(self, surface: str, response: dict) -> dict: ...             # comparable values (default: numbers)
    def golden_compare(self, surface, values, reference, tolerance) -> tuple[int, int] | None: ...  # (checked, matched)
    def outcomes(self, surface: str, body: dict) -> Iterable[tuple[str, str]]: ...  # (type, outcome) per item
```

A work item (`WorkItem`) has `ids` (its token IDs) and may have `cost`, the
tokens it costs a forward when it has no token IDs (images, audio), and
`cache_key`, a content hash of everything its result depends on: the runtime
then answers a repeated item from the model's result cache without a forward.
`Results` are one value per item or `DEADLINE` (the `Expired` instance) for an
expired item or plan. Results are shared with the cache, so `finish_surface`
never mutates them. `mypy --strict` checks the plugin interfaces, the
scheduler, the profiles, the runtime core, the API, the registry, the shared
heads, supervision, the engines and the model families in
`make model-runtime-test`.

Decision families subclass `DecisionModel` (`plugins/decisions.py`), the
decisions mixin: it serves `/v1/decisions` through the family's `plan` (render
every question) and `answer` (one question's readout), compares golden
answers question by question and reports each question's type and outcome for
the metrics. `decision1`, `decision2` and `vela2` use it (`vela2` assembles a
request's answers in its own `finish_surface`); the generic base has no
decisions method. `ModelInfo` grows the descriptors the router needs before
it routes: `heads` (name, kind, labels, default threshold, overflow and window
policy), `embedding` (dimensions, layer exits, modalities, normalisation,
representation identity) and `rerank` (pair-scorer exits).

`ModelSpec` is the hand-off to an engine: the backbone architecture and
config, the weight files (or ONNX graphs), head and LoRA tensors, the dtype
policy and the token budget. The family never imports an engine; an engine
never parses a package format.

### 5.2 Engine

```python
class Engine(ABC):
    name: ClassVar[str]                     # "native" | "onnxruntime"
    def supports(self, spec: ModelSpec, device: DeviceInfo) -> str | None: ...
    def read(self, spec, accelerator, device, options) -> Callable[[], EngineModel]: ...  # host work; returns the device work
    def load(self, spec, accelerator, device, options) -> EngineModel: ...

class EngineModel(ABC):
    def forward(self, batch: ForwardBatch) -> ForwardOutput: ...   # decoders: gathered rows
    def encode(self, batch: EncoderBatch) -> EncoderOutput: ...    # encoders: hidden states or graph outputs
```

`ForwardBatch` carries padded token IDs, the attention mask, the gathered
positions (decision endpoints and query rows) and the shared-context prefix.
`EncoderBatch` carries padded token IDs, the mask, the layer exits the readout
needs and, for engines that run a graph, named graph inputs; `EncoderOutput`
returns hidden states by layer or named graph outputs. Readouts stay in the
family: an engine that returns hidden states (native) and one that returns
graph outputs with heads baked in (ONNX graphs as published) serve the same
family, and the family records which one it used.

A load is two steps. `read` does the host work and returns the device work
that finishes the load; the runtime runs the first before it takes the
device and the second as device work (section 10.2). The native engine reads
the checkpoint into host memory there, in the dtypes the device holds, and
leaves the copy to the device and the model's setup as device work. On the
CPU it reads as device work too: there the host copy is the model's weights,
and the CPU's torch work runs on its device thread (section 9). An engine
without its own `read` (`onnxruntime`, and third-party engines by default)
loads as device work.

The engine defaults to `auto`: a built-in model's preferred engine for the
placed device class (`BuiltinModel.engines`, set where an interleaved
measurement shows it faster), else the first registered engine whose
`supports()` accepts the model on the device, in the order of their
`auto_priority` (`native` 0), then by name for engines without one. A load
that no engine accepts fails with every engine's reason; cards, response meta
and metrics name the engine chosen. `--engine NAME` pins one.

### 5.3 Accelerator

An accelerator describes one device class and supplies kernels: detection,
devices, capabilities, a kernel registry with pure-torch references, autocast
and synchronisation. Every accelerated kernel registers the reference it must
reproduce and whether it is bit-exact; a kernel that is not bit-exact is only
selectable by an opt-in profile. `xpu` and `mps` detect `torch.xpu` and the
MPS backend and run the pure-torch references; they are marked unvalidated.

### 5.4 Profile

| Profile | Numerics | What it does |
| --- | --- | --- |
| `exact` (default) | exact | The released numerics: Decision 2.0 as in Phase 1; Decision 1.0 as its bundled runtime; encoders in FP32 on every device, one request's rows per batch, or the requests queued together on a batch-invariant model |
| `shared_context` | approximate | Runs a multi-question request's shared state once (decision families) |
| `batching` | approximate | Coalesces rows from concurrent requests into shared padded batches within a bounded window |
| `max_speed` | approximate | Numerics-changing kernels and dtypes on top of the above |

A request may select `exact` at any time and another profile only if the
server enabled it for that model.

When `max_speed` is a deployment's configured profile, an encoder also loads
a reduced-precision copy of its linear layers next to the FP32 weights, which
`exact` keeps using: BF16 on GPUs, with FP32 norms, softmax and heads (the
Decision 2.0 GPU policy), and dynamic int8 on CPU where that measures faster
than FP32. `max_speed` requests run the copy. A family consents to a copy
(`DtypePolicy.reduced_gpu` / `reduced_cpu`) only where its records show at
least 99% label agreement with `exact` (embeddings: cosine of at least
0.999); faster alone is not enough. Golden readiness always runs `exact`;
each family records the accuracy (label agreement, max |Δp| or embedding
cosine against `exact`) and the latency of its reduced path. `vllm-sr config
migrate` maps the legacy `precision: fp16` to `max_speed`.

### 5.5 Third-party plugins

A third party ships a normal Python distribution with entry points. The runtime
loads entry points at startup, refuses a name two distributions claim with
different targets, refuses a class that does not subclass its group's base or
carries another name, and lists every active plugin with its distribution,
version and descriptor in `/v1/models`. The built-in plugins are the runtime's
own entry points; run uninstalled from a source tree, it reads them from its
`pyproject.toml`. A family's pinned models and fixture writer, and an engine's
`auto` priority, come from the same class declarations for built-in and
third-party plugins; a test installs a temporary distribution whose family,
table, fixture writer and engine the runtime finds only through its entry
points. A profile builds itself from the process options
(`Profile.from_config`), and the CLI accepts any registered profile and
device. Plugins run in the runtime process; the runtime never executes code
from a model package. `examples/third_party_plugin` is a complete, documented
example (a keyword family, a counting engine, a host accelerator and a
one-by-one profile). The test suite builds its wheel with its own build
backend, installs it into a fresh directory, discovers it through the entry
points its `pyproject.toml` declares, serves it on every surface next to a
Decision 2.0 model, and serves it on the example accelerator and profile.

## 6. API

The contract is `vllm_srun/api/openapi.yaml` (OpenAPI 3.0.3), served at
`GET /openapi.yaml`, checked by contract tests on both sides, and the source of
the Go client in `pkg/modelservice/api`. Every surface takes an optional
`model` (the served model ID, required when a process serves several models)
and `options` with `deadline_ms` and `return_meta` (the runtime's `meta`:
revision, profile, numerics, engine, device and timings, only when asked;
a family's own fields such as `meta.representation` are always returned).
Every surface and bundle response, errors included, carries a
`Server-Timing` header with the request's phases (`parse`, `tokenize`,
`queue`, `forward`, `post`, `serialize`) and their `total` in milliseconds
(`vllm_srun/timing.py`); the router subtracts the total from its own time
for the exchange to record each call's transport.
A request-level error uses
an HTTP status with `{"error": {"code", "message"}}`: 400 `invalid_request`,
404 `model_not_found`, 413 `request_too_large`, 422 `unsupported_surface`
(the model does not serve this surface), 429 `overloaded`, 503 `not_ready`.
Item-level failures (`max_length_exceeded`, `invalid_input`,
`invalid_model_output`, `deadline_exceeded`, `unavailable`) are reported in
place and never fail sibling items.

### 6.1 `POST /v1/decisions`

Phase 1's strict superset of System One, with `POST /v1/systemone` as an
alias. Every family validates the System One fields through one reader
(`systemone.read_question`): text is never blank, keys are unique non-blank
strings, and a question takes only its type's fields plus those its family
reads itself (`over`, `head`, `threshold`, `preset`); anything else is
`invalid_question`, and the OpenAPI `Question` schema says so. Families add
only their own types, presets and defaults. Phases 2–4 add, for models that
declare them:

- **Set** (`type: set`): independent labels, `criteria` as `{label: description}`.
  The response adds `sets.<id> = {selected, probabilities}` and one Noul answer
  per label under `answers["<id>.<label>"]`.
- **Span** (`type: span`): labelled text spans, `criteria` as
  `{label: description}`. The response adds `spans.<id> = [{label, start, end,
  text, probability}]` (Unicode code-point offsets into the field the question
  reads) and a Noul answer `answers.<id>` = the highest word probability.
- **Typed state parts.** A JSON-object state is read as parts: `request` /
  `user` / `prompt` are the user part; `answer` / `response` the answer part;
  everything else the context part. A question's `over` names the field it
  reads (default the whole state; for a span, the answer if present, else the
  request).
- **Thresholds.** A Set or Span question may carry `threshold`; the response
  always reports the applied one in `thresholds.<id>`, including
  length-dependent rules.
- **Further states** (every decision family). `states` names more states, each
  with its own questions; the runtime runs each as a request of its own in the
  model's job group, as it runs a bundle's tasks, so a state's answers are
  those it gets alone, and answers it under the same name in the response's
  `states`. A typed-part state reads all its parts together, so texts a
  client wants read apart are states, not parts.

This follows the Vela 2.0 packages' own server, so the runtime is a drop-in
for it. Decision models without Set or Span answer such a question with
`invalid_question`.

### 6.2 `POST /v1/classify`

Fixed heads: label distributions, independent label scores, token spans and
text-pair inputs.

```json
{
  "model": "vela-pii",
  "input": ["Hi, I'm Tom Baker (tom.baker@example.com)."],
  "head": "default",
  "options": {"overflow": "window", "max_tokens": 32768, "window": {"tokens": 512, "overlap": 64},
              "threshold": 0.5, "deadline_ms": 80}
}
```

- `input`: a string, a list of strings, a list of `{text, text_pair}` pairs,
  or a list of grounded items `{context, question, answer}` (hallucination
  detection reads the answer against the context).
- `head`: the head to run (default: the package's primary head). A model with
  several heads lists them in `/v1/models`.
- `options.overflow`: `reject` (the default unless the head declares another),
  `truncate` (keep the beginning inside the tokenizer's special-token envelope)
  or `window` (overlapping windows reduced by the head's declared rule: `max`
  per label for distributions and scores, span union for tokens). Nothing is
  cut silently: every result reports `input` usage.
- Long inputs are tokenized only as far as the budget decides the answer
  (`text/bounds.py`), so their cost follows the budget, not their length. A
  prefix is cut before a word boundary (whitespace, punctuation, a symbol, a
  CJK, kana or Hangul character) and keeps its words' tokens, except the last
  word and words starting within the longest added token of the cut; in a run
  without a boundary it is cut before a character NFC never composes with its
  predecessor and keeps the tokens ending 1,024 characters before the cut.
  Those are the whole text's first tokens: a probe checks each kind of cut
  once per tokenizer, on probe texts joined with every kind of text after the
  cut, and a tokenizer reads whole texts where a cut fails. One that drops or
  folds characters (WordPiece) never cuts inside a run, and has no bound on
  the characters a token covers. With that bound (the longest vocabulary
  entry, times three under NFC), `reject` and `window` fail a text longer than
  the budget times the bound without tokenizing it. The usage of an input read
  in part counts the tokens read and sets `tokens_lower_bound`. `window` reads
  at most `max_tokens` (the scan budget) and fails a longer input with
  `scan_budget_exceeded`.

```json
{
  "model": "vela-pii", "head": "default", "kind": "token",
  "labels": ["O", "B-AGE", "I-AGE", "..."],
  "results": [
    {"index": 0,
     "spans": [{"label": "PERSON", "start": 8, "end": 17, "text": "Tom Baker", "probability": 0.998}],
     "input": {"tokens": 19, "processed_tokens": 19, "truncated": false, "windows": 1}}
  ],
  "usage": {"input_tokens": 19}, "meta": {"revision": "6d3300c4...", "engine": "native", "device": "cpu"}
}
```

| `kind` | Per-input result |
| --- | --- |
| `sequence` | `label` (arg-max), `probabilities` (softmax, in `labels` order) |
| `scores` | `scores` (independent sigmoid per label); `selected` when the head has an operating point (per-label thresholds) |
| `token` | `spans` (decoded from BIO labels or a binary head, code-point offsets, end exclusive) |

### 6.3 `POST /v1/embeddings`

OpenAI-compatible: `input` (a string, a list of strings, or content parts
`{type: text | image_url | input_audio}`), `dimensions`, `encoding_format`
(`float` or `base64`). Extensions: `layer` (an early exit the model declares),
`input_type` (`query` or `document`, for instructed embedders) and `options`
(`overflow`, `max_tokens`, `deadline_ms`). The response is the OpenAI list plus
`meta.representation`: model identity, layer, dimension and normalisation,
which caches and stores use to keep vector spaces apart.

### 6.4 `POST /v1/rerank`

```json
{"model": "vela-reranker", "query": "how do I reset my password",
 "documents": ["Open Settings > Security ...", "Our offices are closed ..."],
 "top_n": 2, "layer": 22, "dimensions": 768}
```

Returns `results: [{index, relevance_score, logit}]` ordered by score, plus
`usage` and `meta`. `layer` and `dimensions` select a declared pair-scorer exit
(the deployment usually pins one).

### 6.5 `POST /v1/bundle`

All of a request's model work for one runtime process in one call:

```json
{"tasks": [
   {"id": "domain", "classify": {"model": "vela-domain", "input": ["..."]}},
   {"id": "pii", "classify": {"model": "vela-pii", "input": ["..."], "options": {"overflow": "window"}}},
   {"id": "embedding", "embeddings": {"model": "vela-embedding", "input": ["..."], "dimensions": 768}},
   {"id": "decisions", "decisions": {"model": "kai", "state": "...", "questions": {"...": {}}}}
 ],
 "options": {"deadline_ms": 80}}
```

Each task names one surface field whose value is exactly that surface's
request body. The runtime plans every task, submits them to their models'
schedulers at once (so tasks for different models run in parallel and tasks for
one model share its batching), and returns `results` in task order:
`{id, status, <surface>: <response body>}` or `{id, status, error}` with the
status the task would have had alone. The bundle itself fails only when it is
malformed (400) or too large (413): more tasks than `--max-bundle-tasks` (64
by default) or a body over `--max-request-bytes`. `/v1/models` reports both as
`limits`, so a client splits larger work into several bundles. The bundle
deadline applies to every task unless a task's own deadline is earlier.

### 6.6 `GET /v1/models`, `GET /health`, `GET /metrics`

`/v1/models` lists every served model: ID, family, repository, revision,
`model_sha256`, `manifest_sha256`, surfaces, question types, heads (with
labels), embedding and rerank descriptors, limits, licence, profiles, engine,
accelerator, device, dtype, plugin versions, and per-model readiness and
golden status, plus the process's `limits` (`max_bundle_tasks`,
`max_request_bytes`). `/health` is readiness (`Health`): 200 only when every
model is ready, with the process state (`starting`, `loading`, `warming`,
`ready`, `degraded`, `failed`) and, when the process serves several models,
each model's state. `/health/live` is
liveness (`Liveness`): 200 with `status: alive` while the process serves
HTTP, whatever its models' states. All three report `api_version`, the
contract version (`info.version`); a client refuses a runtime of another
major version.
`/metrics` is Prometheus text with a `model` label on every model-scoped
series.

## 7. Model registry

### 7.1 Resolution

`vllm-sr serve <model>` accepts local package directories and Hub repository
IDs, one or several. A Hub model is always resolved to a 40-hex commit: with
`--revision` (or `MODEL@REVISION`) the runtime downloads that revision;
without it, a built-in model uses its pinned revision and any other repository
needs an explicit revision. Downloads go into the HF cache with an allow-list
derived from the family's inventory. A token comes from the environment or the
HF token file, never from argv. Gated and private repositories need a token
with access.

### 7.2 Verification

Before any model code runs, the family verifies the package:

- **Decision 2.0** (Phase 1): the root pointer, `MODEL_MANIFEST.json` with
  every file's SHA-256, header parameter counts, identity, the adapter base.
- **Decision 1.0, Vela 1.0, Vela 2.0, Omni:** these packages either have no
  manifest or one that does not cover every file. The built-in table records,
  per pinned revision, the SHA-256 of every file the family loads (weights,
  configs, tokenizer, label maps, operating points, graphs). The on-disk bytes
  must match; files the family does not load are never downloaded. A package's
  own manifest (Vela 2.0 `MODEL_MANIFEST.json`, Route `MANIFEST.json`) is
  checked too when present. A local package without a built-in entry reports
  `verification: local` with its computed identity.
- In every case links may resolve only inside the HF cache that holds the
  snapshot, loaded parameter counts are checked after loading, and
  `trust_remote_code` is never used: bundled Python (`modeling_*.py`,
  `decision1_*.py`, `vela2_inference.py`) is hashed but never imported.

### 7.3 First-party models

| Family | Models | Pinned revisions |
| --- | --- | --- |
| `decision2` | `vllm-sr/Decision-2.0-{Kai-0.6B, Eos-0.8B, Sol-2B, Nox-4B, Lux-9B, Vega-27B}` | Kai `cd49ea38`, Eos `3594047d`, Sol `64235bef`, Nox `25e8f67d`, Lux `78bf3c03`, Vega `7aec49ae` (runtime-only revisions of the Phase 1 pins, same weights and identity) |
| `decision1` | `vllm-sr/Decision-1.0-{Kai-0.6B, Lex-0.6B, Route-0.6B}` (Vela encoder runtime); `{Eos-0.8B, Sol-2B, Nox-4B, Lux-9B}` (Qwen3.5 runtime) | Kai `79263ba4`, Lex `a5ba6895`, Route `deed1f29`, Eos `2ca39a23`, Sol `5c698b1a`, Nox `7f65e1db`, Lux `2064c84d` |
| `task_heads` | `vllm-sr/Vela-1.0-Encoder-307M-{Domain, Guard, Safety, Shield, FactCheck, Feedback, Modality, Hazard, PII, Halu, Embedding, Reranker}`, `Qwen/Qwen3-Embedding-0.6B` | The revisions the router pinned (section 16.3), for example Domain `f6354f54`, PII `6d3300c4`, Halu `ca875312`, Embedding `1e57cebf`, Reranker `a388e41c` |
| `vela2` | `vllm-sr/Vela-2.0-{0.3B, 0.8B, 4B, 9B}` | 0.3B `a3209a50`, 0.8B `a778eb2a`, 4B `c1e64d4f`, 9B `bc876163` |
| `multimodal_embedding` | `vllm-sr/Vela-1.0-Omni-{Nano, Mini}` | Nano `2ff2d663`, Mini `801bae3a` (the published weights, section 8.5) |

Each family names its table (`ModelFamily.builtin_table`); the built-in
families keep theirs in `registry/tables/<family>.py`, with the full
revisions, file digests, the expected identity and parameter count, and
reference answers per device class in `registry/golden_answers*.json`.
`registry.builtin` reads the tables of the installed families once, in
family-name order. A table that fails to import pins nothing and is logged,
as a family that fails to load serves nothing. A repository two tables pin,
or a table entry of another family, is refused, and with it every model of
the process, built-ins included: no family may take over another's pin. A
bare model name resolves only while one organisation's model has it; a name
two tables use is logged, and only their repository IDs resolve. A new
revision is a new entry; the runtime never follows a moving branch.

### 7.4 Licence and access

`/v1/models` reports per-component licences. The policy refuses a component
marked non-commercial or research-only unless the operator passes
`--accept-licence <id>`. Vela 2.0 0.3B's tokenizer carries the Gemma terms;
the runtime reports them and serves it (they permit use). Gated and private
repositories need a token from the environment or token file.

## 8. Families

### 8.1 Decision 2.0 (`decision2`, Phase 1)

Unchanged from Phase 1. The renderer reproduces the scored prompt exactly
(`decision2-segmented-options-global-query-v1`): each segment is tokenized
separately, the last token of each option segment is that option's endpoint
and the last token overall is the query position. The candidate head reads
the endpoint and query rows in FP32. Answers follow the released runtime
(per-level Score offsets, temperature, softmax, arg-max with first-option
ties, P(true), expected level, normalised-entropy confidence). The native
engine implements the Qwen3 dense and Qwen3.5 hybrid backbones and Vega-27B's
unmerged LoRA; on GPUs the backbone runs under BF16 autocast with BF16-resident
Linear weights, FLA kernels with the released kernel choices pinned per model
(`registry/kernel_choices.json`, §11), so models that share a GPU process each
run their own. Byte-identity is defined against the released runtime on the
same device class and recorded on the four scored panels (10,653 prompts).

### 8.2 Decision 1.0 (`decision1`, Phase 2)

Decision 1.0 packages carry the pointer `{"decision_format": "vllm-sr-decision",
"format_version": 1, "runtime_family": ...}` and two runtimes:

| `runtime_family` | Models | Backbone and readout |
| --- | --- | --- |
| `vela-encoder` | Kai-0.6B, Lex-0.6B, Route-0.6B | The Vela 307M ModernBERT encoder plus separate choice and score encoders and K-slot decision heads (`native/decision_config.json`) |
| `qwen3.5-decision` | Eos-0.8B, Sol-2B, Nox-4B, Lux-9B | A Qwen3.5 backbone and an FP32 decision head, with the package's calibration temperature |

The family reimplements the bundled runtime (`decision1_vela.py`,
`decision1_qwen.py`, `decision1_system_one.py`) on the native engine's
ModernBERT and Qwen3.5 backbones: the same prompt rendering, slots, readout,
calibration and System One answer assembly. `exact` targets bit-identity with
the bundled runtime on the same device class; the parity record runs both on
the same requests. Route-0.6B also declares its fixed router questions
(`QUESTIONS.json`) as presets.

### 8.3 Encoder task heads (`task_heads`, Phase 3)

Serves HF-format encoder task models: every Vela 1.0 `Encoder-307M` task
model and compatible ModernBERT (mmBERT) classifiers.

- **Detection:** `config.json` with `model_type: modernbert` and a known
  architecture (`ModernBertForSequenceClassification`,
  `ModernBertForTokenClassification`, `ModernBertModel` with a
  sentence-transformers pooling config or reranker heads).
- **Backbone:** the native engine's ModernBERT (alternating global and
  128-token local attention, YaRN-extended rotary positions, GeGLU, pre-norm),
  in FP32, checked against the Transformers reference on fixtures.
- **Heads:**

  | Head kind | Readout | Vela 1.0 models |
  | --- | --- | --- |
  | `sequence` | CLS or mean pooling, prediction head, classifier, softmax | Domain, Guard, Safety, Shield, FactCheck, Feedback, Modality |
  | `scores` | Mean pooling, classifier, sigmoid per label; the package's `operating_point.json` thresholds | Hazard |
  | `token` | Per-token classifier, BIO decoding to spans | PII |
  | `grounded` | Token classifier over `context [SEP] question [SEP] answer`, spans on the answer, operating point | Halu |
  | `pooled` | Mean pooling at a layer exit, Matryoshka truncation, L2 normalisation | Embedding |
  | `relevance` | Pair input, per-exit classification heads (`classification_heads.safetensors`) | Reranker |

- **Long inputs:** `reject`, `truncate` and `window` exactly as the router
  applies them today (the window size, overlap and reductions of
  `pkg/modelruntime/native/window_contract.go` move into the family), so the
  router stops tokenizing. The family reads an input only as far as its budget
  decides (section 6.2). On the CPU, local layers attend in calls of 4,096
  query tokens (`BAND_CALL_TOKENS`), so the keys, values and masks a call
  copies stay small on a 32,768-token row; each query block is its own SDPA
  problem, so the result is the same, bit for bit
  (`docs/records/input-memory-cpu.md`).
- **Engines:** native (PyTorch FP32) by default; `onnxruntime` runs the
  packages' ONNX graphs where they exist. Both are parity-checked.

### 8.4 Vela 2.0 (`vela2`, Phase 3)

Vela 2.0 is a schema-conditioned decision model: options, labels and rubrics
arrive at request time, and one forward answers Choice, Noul, Score, Set and
Span questions over typed parts.

| Model | Backbone | Context | How questions are read |
| --- | --- | --- | --- |
| `Vela-2.0-0.3B` | Vela 307M ModernBERT encoder with eight marker tokens | 8,192 tokens; longer labelled parts windowed with 512-token overlap | One sequence `<bos> ([Q] question ([O] option)* [ABS]?)* ([E] label)* [SEP_SCHEMA] ([SEG_role] part)* <eos>`; one extra sequence per extra span question |
| `Vela-2.0-0.8B`, `Vela-2.0-4B`, `Vela-2.0-9B` | Qwen3.5 (the Decision 2.0 base) used as an encoder | 16,384 tokens; span targets over 2,048 tokens in 1,800-token windows | The state once, then one block per question that sees the state and itself (the native engine's shared-context tree) |

The family reimplements the packages' engine (`vela2_inference.py`): marker
insertion by token ID (user text can never create a marker), per-part pooling,
option logits (cosine plus MLP terms), softmax or sigmoid, word-level span
scores with code-point offsets, per-type temperatures, and the thresholds in
`calibration.json` (constant, per question, the PII length rule and its
sparse-document gate). It serves `/v1/decisions` with Set and Span, plus the
packages' presets (`pii`, `halu`, `toxic`, relevance) as named questions.

A question reads a part whole up to a scan budget (`limits.max_scan_tokens`):
four inputs on a CPU, 32 on a GPU, or the model option `max_scan_tokens`; a
request's `options.max_tokens` overrides it, never below one input. Each part
is tokenized only as far as the budgets need (`bounds.read`), so a longer part
costs its budget, not its length, and fails the questions that read it whole
with `scan_budget_exceeded`; the other questions read its first tokens, which
are all any truncation keeps of a part they do not read. A question with
`overflow: truncate` reads only a part's first `limits.truncate_tokens` (one
input on a CPU, its row one forward over what fits). The model's answer to a
question depends on the questions beside it in a sequence, so questions of
both kinds share their rows as before unless a row has to be read in windows;
only then does each kind get rows of its own. The router truncates its
routing questions and reads its safety questions whole. Span windows start at
word starts found by bisection, so planning a long target costs its windows
times the logarithm of its words.

### 8.5 Multimodal embeddings (`multimodal_embedding`, Phase 3)

Vela 1.0 Omni Nano and Mini embed text, images and audio into one space. The
family serves the published repositories as they are. The native engine builds
the model's four towers from `model.safetensors`, the text backbone plus three
named towers (`ModelSpec.towers`), and loads every tensor by its exact name:

| Tower | Nano | Mini |
| --- | --- | --- |
| Text | BERT (12 layers, width 384), CLS | Qwen3 0.6B, last token, Matryoshka 768 |
| Image | SigLIP B/16 at 512 px with its attention-pooling head | SigLIP SO400M/14 at 384 px |
| Speech | Whisper tiny encoder, mean over its 1,500 frames | Whisper medium encoder |
| CLAP | HTSAT Swin audio encoder and its projection | the same |

The family owns the published readouts (text pooling, the image and speech
projections, the frozen CLAP residual and the unit normalisations) and the
processors: the tokenizer, SigLIP's resize and normalisation, and the Whisper
and CLAP log-mel features, with the Slaney filter banks computed as
Transformers computes them. Verification checks the SHA-256 of every file it
reads against the pinned table, that the safetensors header holds exactly the
tensors the towers and readouts read, and the declared parameter count. The
towers equal the reference Transformers modules bit for bit on CPU, SigLIP
within 1.9e-6 (MKL's one-row products depend on buffer alignment).

Each image and audio item runs alone. Texts share a packed batch where the
load-time probe finds the model batch-invariant; on 8 or more CPU threads
oneDNN splits Nano's 1,536-to-384 product differently per row count, so Nano's
texts then run one at a time.

`tools/models/vela_omni` still prepares ONNX bundles (four verified graphs and
the exact processors) for the optional `onnxruntime` engine: a deployment
names the bundle directory and `engine: onnxruntime`, with the `onnx` extra
installed. The router images ship neither ONNX Runtime nor a bundle.

## 9. Scheduler and planner

```text
request -> validate (400) -> route to its model (404 / 422) -> admission (429)
        -> plan: render every item (per-item errors) -> that model's queue
        -> its worker: the profile forms batches -> shortest expected finish first -> readout -> response
bundle  -> plan every task -> submit all at once -> await each -> results in task order
```

Each model has its own scheduler and worker thread. Validation and rendering
happen on the request thread, so malformed requests never take queue
capacity. In `exact`, one request's rows form one padded batch, split only by
the forward token budget. A bundle's tasks for one model and profile are
queued at once as one job group, whatever their deadlines (each job keeps its
own, measured from when the HTTP request arrived). Decision models still run
each task as its own batch (their released numerics), and a family that sets
`fuse_bundled_jobs` (encoders) runs the group as one batch, so one forward
serves every head and consumer that reads the same input. On a batch-invariant model, `exact` runs the queued jobs
of concurrent requests in shared batches of at most 512 padded tokens, rows
with the same token IDs always together so the family computes them once. A
model that pads its rows (an Omni ONNX bundle) keeps one length class
(power-of-two band of padded length) per batch; a model that packs its rows
back to back (`packs_rows`: the native encoders of `task_heads` and Omni)
mixes lengths freely, since a packed row costs no padding, and closed-loop callers' short
texts of different lengths then still share forwards. Either way each row's
answer is the one it gets alone, which the load-time probe checks on a batch
that spans several length classes. Repeated cacheable
items are answered from the model's result cache (`--result-cache-entries`,
keyed by profile and content).

The worker runs planned batches in order of their jobs' expected finish: the
time a job was queued plus its tokens at the measured cost per token (a running
average of recent forwards). Short work goes first, and a long job still runs
once the work queued before its expected finish is done, so nothing starves.
Between two forwards the worker takes the jobs that arrived meanwhile (jobs of
coalescing profiles once they have waited one batching window), so a short request
waits for the forward in flight, not for the remaining windows of a long one;
each job is answered as soon as its own batches have run. The order of
forwards never changes a batch's contents, so answers are unchanged. Deadlines
travel with jobs; a job whose deadline passes before its next batch runs is
answered `deadline_exceeded` and its remaining batches are skipped. A client
that disconnects before its answer cancels its jobs the same way: they leave
the queue at once and stop counting against admission, and the request is
counted with status 499 in the metrics. Admission
bounds, per model, the jobs not yet answered (queued, planned or running) and
their tokens, and answers 429 at once, for a whole group or none of it. On
CPU, workers of different models share the process's intra-op thread pool
(`--threads`). On a GPU, each device call (load, warm-up, golden check, a
batch) holds the device's lock (`GPUAccelerator.execute`), so the models of
one GPU never launch work at once: a model captures a HIP / CUDA graph the
second time it sees a shape, and a capture fails when another thread
launches work on the device, whatever the capture mode.

A model whose batches need no device thread (`device_thread = False`, today
an Omni ONNX bundle) runs them without the CPU's device thread (on
a GPU they still hold the device's lock), and skips the hand-off to its worker
when it is idle: a lone request planned off the event loop runs on its
planning thread when nothing is queued or planned and the worker is not
running a batch. The
caller and the worker each own the model while they run batches, so one model
never runs two forwards at once, and the event loop never runs one. Once
another job is queued the caller stops after its forward in flight and the
worker runs the rest in expected-finish order with the new arrivals.

## 10. Placement and supervision

### 10.1 Placement

Each model is placed on its own device. `--device` names `auto` or any
installed accelerator plugin with an optional `:N` (`cpu`, `cuda`, `rocm`,
`xpu` and `mps` are built in); an explicit device is honoured or fails, and
one this host does not have fails before the model is resolved, so nothing is
downloaded.
`auto` tries the accelerators in the order of their `auto_priority` (ROCm,
CUDA, then the CPU; an accelerator without one, such as XPU, MPS or a
third-party plugin by default, serves only when named) and takes the first
available device with enough free memory. The memory estimate is the weight
bytes under the dtype policy plus the activation bound; `--memory-budget`
caps it per model. A model names the device capabilities it requires per
accelerator (`ModelSpec.requires`; a backbone architecture declares its own
once, in `BackboneSpec.requires`, and the families pass it on), and placement
refuses a device whose accelerator does not report one, before any weights
load. The Qwen3.5
decoders require `lapack` on the CPU, since their gated-delta kernel solves
triangular systems there: a PyTorch built without LAPACK (the ROCm image's)
refuses them with that cause and the fix (a PyTorch with LAPACK, such as the
CPU image, or a GPU device). When every device the model may use lacks a
capability, the failure is final instead of retried.

### 10.2 Readiness and health

Per model: `loading` (resolve, verify, load, identity) → `warming` (golden
answers) → `ready`, or `failed` with a reason. Readiness is gated on golden
answers per surface: every built-in revision carries golden requests with
reference answers per device class (1e-3 on CPU, 0.02 on GPUs; encoder
embeddings compare by cosine). Every golden request runs through its
surface the same way: the loaded model's `golden_values` and `golden_compare`
say what to compare and how, so decisions count questions and the other
surfaces count values. Without a reference the runtime checks
determinism and well-formed answers and reports `golden: unverified`. A
failed model does not stop the others; `/health` reports the process
degraded and the router treats only that model's deployments as unavailable.
A device fault during serving exits the process so the supervisor restarts it
with a clean device context. The accelerator decides what is one (on CUDA and
ROCm: a CUDA or HIP error, a device-side assert or an illegal memory access;
running out of memory is not). Any other error in a forward fails only the
requests whose jobs were in that batch, as `internal_error`, and so does an
answer holding NaN or infinity; the model keeps serving. A model that fails
to load releases its worker and device memory. A package or golden-answer
failure is final; any other load failure (no device with enough free memory,
a download, a busy device) is retried `--load-attempts` times in all (5),
after `--load-retry-seconds` (5 s) doubling up to 300 s, while the model
reports `loading` with the reason. An attempt, like the first load, reads the
weights before it takes the device (`Engine.read`, section 5.2), so the other
models of its GPU keep answering meanwhile; they wait only for its device
work (section 9): the copy to the device, the family's load and each golden
batch. On the CPU's device thread the native engine's read is device work
too, so the other CPU models of the process wait for it.

### 10.3 Router-managed lifecycle

Section 13.4.

## 11. Engines and hardware

| Engine | Runs | Devices |
| --- | --- | --- |
| `native` | Qwen3, Qwen3.5 (GDN hybrid), ModernBERT, BERT, unmerged LoRA; Omni's SigLIP, Whisper and CLAP towers; hidden states, gathered rows, shared-context trees | Every accelerator |
| `onnxruntime` (optional: the `onnx` extra, not in the router images) | Package ONNX graphs (heads baked in) and prepared Omni bundles | CPU EP (validated); CUDA EP, ROCm / MIGraphX EPs and the OpenVINO EP when the installed build provides them (unvalidated until recorded) |

| Accelerator | Status | Kernels |
| --- | --- | --- |
| `cpu` | validated | pure-torch references, FP32; on x86, encoder linears through oneDNN's pre-packed FP32 kernel (weights reordered once at load, batch-invariant, within 3.3e-6 of `F.linear`) |
| `rocm` | validated (MI300X, MI325X) | BF16 autocast, FLA gated delta, causal-conv1d, exact-shape HIP graphs, bit-exact fused Triton element-wise kernels (gfx942) |
| `cuda` | implemented, unit-tested, **unvalidated** | the same kernel slots; fused kernels only after a bit-exactness record |
| `xpu`, `mps` | built-in plugins, **unvalidated** | pure-torch references |

Intel hardware that used the OpenVINO provider runs on CPU, on `xpu`, or
through ONNX Runtime's OpenVINO execution provider.

**Pinned kernel choices, per model.** FLA's gated-delta kernels pick block
sizes and warps by timing them, so two processes could round the Qwen3.5
sizes differently. Each built-in Qwen3.5 model records the configurations its
release ran (`registry/kernel_choices.json`, per device class), and the
runtime runs those instead of timing: a recorded tuning key gets its
configuration, and any other key the first entry, in key-hash order, that
differs from it only in numbers, else the kernel's first entry. That is how
FLA's `FLA_CACHE_MODE=full` resolves the same entries written as config files.
The released models record different configurations for the same keys (47
of 55 pairs of built-in GPU models disagree somewhere), and the wrong ones
change answers (`decision1-parity.md`), while FLA's autotuners are
process-wide. So the choices are per model (`accel/autotune.KernelChoices`):
each FLA autotuner they name gets a cache that routes lookups, per thread, to
the configurations of the model whose scope the thread is in, and the runtime
runs every device call of a pinned model (load, warm-up, golden check, graph
capture, the worker's batches, inline batches) inside that scope. Other
threads, and models without choices, use the autotuner's own cache. The
routing takes no lock of its own (a GPU's device calls are serialized for
graph captures anyway, §9); a captured graph keeps the configurations it was
captured with. A model whose choices
can't run here (another FLA version, or a recorded kernel FLA no longer has)
still loads, with golden `unverified` and the reason in its health and card.
`--autotune-cache` still persists Triton's compiled kernels and the tuning of
models without choices.

The ROCm router image serves with the PyTorch of vLLM's ROCm image, pinned by
digest (2.12.0 built on ROCm 7.2.3, with AOTriton 0.13.50), and that image's
ROCm 7.2.3 libraries. The released decoders were recorded on that build: the
official rocm7.2 wheel bundles AOTriton 0.11.2, whose attention rounds
differently and moves about 5% of Vela 2.0 4B / 9B span sets, while this build
answers their panels byte for byte. The official wheel still provides torch's
Python dependencies and Triton. causal-conv1d 1.7.0 is built against it with
ROCm 7.2.3's compiler, the one the released wheel was built with, so its gfx942
machine code is the released build's. It carries
code for gfx90a, gfx942 and gfx950 (Instinct MI200, MI300 and MI350;
`CAUSAL_CONV1D_ARCHS` builds others), since a decoder fails on a GPU it has no
code for rather than falling back. Triton's compiled kernels and
MIOpen's databases live in the model volume, since the charts run the router
on a read-only root without a home directory.

The oneDNN linear is a kernel variant a family opts into
(`ModelSpec.kernel_variants`): Vela 1.0 task heads use it on `exact`, since
their reference is the legacy path's recorded agreement, not a bit pattern;
Decision 1.0 keeps `F.linear` to stay byte-identical to its bundled runtime.
Tests that compare against Transformers bit for bit run the reference
kernels. A model whose whole CPU forward is batch-invariant (each row the
same alone or in any batch, shown by a test) sets
`LoadedModel.batch_invariant`, and `exact` then batches the requests queued
together, which lifts throughput under load without changing an answer.

## 12. CLI

Engine mode runs a runtime process in the current Python environment:

```bash
pip install ./src/vllm-sr ./src/model-runtime   # after PyTorch; or the vllm-sr image, or src/model-runtime/Dockerfile
vllm-srun serve vllm-sr/Decision-2.0-Kai-0.6B --device cpu --port 8100
vllm-srun serve vllm-sr/Vela-1.0-Encoder-307M-Domain vllm-sr/Vela-1.0-Encoder-307M-PII --device cpu
vllm-srun serve vllm-sr/Decision-2.0-Lux-9B --device rocm:0 --profile shared_context
vllm-srun serve --models models.yaml --uds /run/vllm-sr/runtime.sock
```

Several `MODEL` arguments share the process options; `MODEL@REVISION` pins a
revision per model; `--models FILE` lists models with their own name,
revision, device, profile, engine and family options (the router writes this
file in managed mode). These are worker-level `vllm-srun` options. The managed
`vllm-sr serve ARTIFACT --engine` command instead exposes the frontend and
Dashboard, with named models and listeners configured in YAML. Every invocation
of `vllm-sr serve` without `--engine` starts Router mode. Every serve option's default
is the `ServeConfig` / `ModelConfig` field default, so the CLI and an
embedding host start alike. `vllm-srun fixture OUTPUT --family F
--variant V` writes a tiny random-weight package of any installed family
that names a writer (`ModelFamily.fixture_writer`; the built-in families'
writers are `testing/<family>.py`). `vllm-srun devices` prints, as JSON,
the devices of the available accelerators (`devices`) and the one
`--device auto` tries first (`auto`: the first device of the first available
accelerator in `auto_priority` order); the router reads `auto` before it
groups its processes (section 13.4).

Importing `vllm_srun` sets these environment defaults:
`THP_MEM_ALLOC_ENABLE=1`, `ONEDNN_PRIMITIVE_CACHE_CAPACITY`, on x86_64
`OPENBLAS_NUM_THREADS=1` (OpenBLAS's idle threads took the cores of the CPU
device's OpenMP team after the runtime's small NumPy products), and MIOpen's
`MIOPEN_FIND_MODE=FAST` with `MIOPEN_LOG_LEVEL=3`. MIOpen's
default find mode times each new convolution shape's solvers, so cold
processes picked different ones and answered differently; FAST never times.
`THP_MEM_ALLOC_ENABLE` puts PyTorch's CPU allocations of 2 MiB or more on
transparent huge pages, where the kernel's `madvise` mode allows: on 4 KiB
pages the physical placement of a model's copied weights varied by process,
and some processes ran the same CPU forward about a quarter slower
(`decision1-performance.md`). They take effect only when
PyTorch loads, and the plugin base layer imports PyTorch, so they cannot wait
for `main`. A value the caller set is kept.

libgomp's spin count depends on what a process serves, so `serve` chooses
it from its models before anything imports PyTorch (`vllm-sr serve` runs the
same `main`), and keeps a value the caller set. PyTorch's idle OpenMP threads
spin that long after a native forward. Native CPU forwards run fastest on
libgomp's default (300,000), while an ONNX Runtime run that follows on the
same cores needs them asleep sooner, so a process with a model that names
`engine: onnxruntime` on a `cpu` or `auto` device gets `GOMP_SPINCOUNT=10000`
(`decision1-performance.md`, `embed-performance.md`). `engine: auto` counts
as native: it tries `native` first, which runs every built-in model. A
process that still serves ONNX Runtime beside other CPU models on the
default (a third-party package only ONNX Runtime runs, or a runtime embedded
in another program) logs a warning that names the fix.

## 13. Router integration

### 13.1 Configuration

The canonical layout is unchanged. A deployment with `provider: model_runtime`
names a runtime model; bindings connect consumers to deployments with the
existing contracts.

```yaml
global:
  model_catalog:
    deployments:
      vela-domain:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Domain
        device: cpu                  # cpu | auto | cuda:N | rocm:N | xpu:N | mps
        input: {max_tokens: 512, overflow: reject}
      decision-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        device: auto
        process: decisions           # optional: deployments with the same name share a process
        # endpoint: http://runtime:8100   # attach to a runtime the router does not manage
    bindings:
      domain_classifier: {deployment: vela-domain, contract: label_distribution.v1, adapter: modernbert}
```

- `artifact` is a Hub repository (pinned by the built-in table, or with
  `revision`) or an absolute package path. `revision`, `device`, `profile`,
  `input` and `endpoint` keep their Phase 1 meaning; `process` is new.
- Built-in module defaults (`global.model_catalog.system.*`, the embedding
  models) resolve to implicit `model_runtime` deployments of the same
  artifacts, on CPU unless `use_cpu: false` (then `auto`).
- The providers `candle`, `ort` and `openvino`, and `precision`,
  `custom_ops_profile`, `compilation_cache_dir` and graph-file `head` values
  are rejected with a pointer to `vllm-sr config migrate`.

### 13.2 Typed task bindings

`pkg/modelruntime/serving` replaces `pkg/modelruntime/native` with the same
facade, so consumers keep their logic:

| Facade method | Contract | Runtime call |
| --- | --- | --- |
| `Sequence`, `SequenceWindows` | `label_distribution.v1` | `/v1/classify`, `kind: sequence` |
| `Scores`, `ScoreWindows`, `OperatingPoint` | `label_scores.v1` | `/v1/classify`, `kind: scores` |
| `Tokens`, `TokenWindows` | `token_spans.v1` | `/v1/classify`, `kind: token` |
| `Grounded` | `token_spans.v1` (grounded) | `/v1/classify`, grounded input |
| `Embedding` | `embedding.v1` | `/v1/embeddings` (text, image, audio; dimension and layer views) |
| `RemoteEmbedding` | `embedding.v1` | external OpenAI-compatible endpoint (unchanged) |
| `Relevance` | `relevance_scores.v1` | `/v1/rerank` |
| `Diagnose*` | all | the same calls, for the router's diagnostics API |

Preparation reads the deployment's `/v1/models` card and checks the binding
against it (surface, head kind, labels in mapping order, embedding dimensions
and exits, input budget) before the generation is published, as the native
facade checked loaded artifacts. Span offsets arrive in code points and are
converted to UTF-8 byte offsets once. The inventory the dashboard shows is fed
from the same cards.

### 13.3 Request flow and bundles

The signal stage evaluates signal families in parallel goroutines, as before.
It opens a request bundle in the context (`modelservice.WithBundle`) and every
goroutine joins it (`Join`); a runtime call made through `modelservice`
(`Decide`, `Classify`, `Embed`, `Rerank`) parks in the bundle, and the bundle
flushes one `/v1/bundle` call per runtime process when every joined goroutine
has parked or finished (or 2 ms after the first call parked), then hands each
caller its own result. Work that fans out inside one signal (one call per
piece or per rule) joins through `Fan`, so it parks as several participants.
Only model-backed signals join; keyword and heuristic signals never delay a
flush. Calls outside a bundle go directly to their surface. The response
stage (hallucination, response guard) opens its own bundle. Each call carries
the smallest consumer timeout as its context deadline and `options.deadline_ms`.

A flush fuses the classify calls that differ only in their inputs and deadline
(one per PII or jailbreak chunk, one per message a history rule reads) into
one task per model, head and options, up to the card's `limits.max_inputs`,
and hands each caller its own results; a fused task carries the latest of its
callers' deadlines, and each caller still stops at its own. So a long prompt
is one runtime job, not one per chunk. The flush then splits each process's
tasks into bundles of at most the process's `limits.max_bundle_tasks` (64
until `/v1/models` has been read); the router starts managed runtimes with
`--max-bundle-tasks 1024` and `--max-request-bytes` 64 MiB, so a stage
normally stays one call.

A bundle is per runtime process, not per request. The default process groups
(section 13.4) give each CPU model its own process, up to the process cap, and
one process to each GPU device. So a request whose signals read several CPU
models makes one bundle call per model process, and those calls run in
parallel. One CPU process runs one forward at a time, so a single shared
process would serialize the stage: on 16 cores it served 11.1 requests/s
against 20.3 for the split (`router-latency-cpu.md`). Deployments that name
the same `process` share one process and one bundle.

### 13.4 Lifecycle: process groups and attached endpoints

- **Managed (default).** The router groups its `model_runtime` deployments
  without an `endpoint` into processes: by `process` when set; else each GPU
  device gets one process, and each CPU model its own process (`cpu-0`,
  `cpu-1`, ...; at most half the cores, capped by
  `VLLM_SRUN_CPU_PROCESSES`; `1` folds them into one). A deployment on
  `device: auto` is grouped by the device `auto` takes on the host, which the
  router asks the runtime once (`vllm-srun devices`, section 12) and
  keeps: on the CPU it is planned as a `cpu` deployment, so on a host without
  a GPU each `auto` model gets a CPU process and thread share of its own; on a
  GPU it joins that device's process (`rocm:0`, ...), and the runtime still
  places it by free memory; one it then places on the CPU runs in that
  process without a thread share. If the runtime cannot answer, the `auto`
  deployments share one process, `auto`, until the router restarts, and the
  router logs why: a later reload does not ask again, so a slow query (bounded
  at two minutes) holds up only the first plan, and plans do not change under
  a running router. A
  CPU process runs `ceil(cores / processes)` threads, unpinned, so a busy model can use
  the cores an idle one leaves: on 16 cores and five task models, one shared
  process served 11.1 requests/s, pinned disjoint shares 15.6, unpinned
  shares 20.3 (`docs/records/router-latency-cpu.md`). For each group it
  writes a models file and starts `vllm-srun serve --models <file>
  --uds <path>` when the configuration loads, and stops it when the group
  disappears or the router exits (SIGTERM, then SIGKILL after a 10 s grace
  period, logged as `runtime_process_killed`). Preparing a binding waits for the deployment's card while its model
  is `loading` (up to `VLLM_SRUN_READY_TIMEOUT`, default 10 minutes), and building a
  router generation waits the same way for every managed deployment it leases, so the
  decision models of decision signals and of the decision selector are ready before
  the generation serves; `/ready` and `/startup-status` name the ones startup waits
  for, and a reload keeps the previous generation serving meanwhile. The supervisor
  polls `/health` and `/v1/models`, restarts a dead or failed process with
  exponential back-off (1 s to 60 s), and marks only the affected deployments
  unavailable. Sockets live in a private 0700 directory.
- **Attached.** With an `endpoint` (`unix:///path` or `http(s)://host:port`)
  the router uses a runtime it does not manage, such as a Kubernetes sidecar or
  a shared GPU runtime started with `vllm-sr serve <hf-model> ...`; the
  deployment name or `served_name` selects the model on it.
- `VLLM_SRUN_COMMAND`, `VLLM_SRUN_DIR` and
  `VLLM_SRUN_CACHE_DIR` keep their Phase 1 meaning. Every router image
  ships the runtime, so managed deployments work out of the box on CPU, and on
  GPUs with the GPU images.
- **Fail-open** is unchanged once a generation serves: while a deployment is
  not ready (a restarting process, an attached runtime that is not up) its
  calls fail at once, late answers are unknown at the consumer's timeout, and
  every consumer follows its existing `on_error` / `on_unknown` policy.

### 13.5 Client, supervisor, metrics

`pkg/modelservice` owns the generated client, transports for UDS and TCP with
keep-alive pools, request bundles, process groups, per-deployment state
(`starting`, `ready`, `unavailable`), the supervisor and metrics
(`vsr_model_runtime_*`: requests and latency per deployment and surface,
result-cache outcomes, bundle sizes and waits, unknown answers by reason,
readiness and restarts). A per-deployment LRU of classify and decision
results (`VLLM_SRUN_RESULT_CACHE` entries, default 4,096, cleared when
readiness changes) answers repeated requests without a round trip; the
runtime's own item cache serves every router and deduplicates items inside a
bundle.

`pkg/modelruntime/serving` is the facade the router's consumers use: typed
readouts (`Sequence`, `Scores`, `Tokens`, `Grounded` and their windowed
forms), operating points from the model card (thresholds, window, `max`
reduction), code-point spans converted to byte offsets once, and responses
checked against the prepared head's labels.

### 13.6 Configuration migration

`vllm-sr config migrate` (the existing migration command) rewrites legacy
layouts; the router parser and `vllm-sr validate` accept only the canonical
layout and refuse every path below with a pointer to `config migrate`.

| Legacy | Migrated to |
| --- | --- |
| `provider: candle \| ort \| openvino` | `provider: model_runtime`; device `cpu` → `cpu`, `cuda:N` → `cuda:N`, `rocm:N` / `migraphx:N` → `rocm:N`, `metal:0` → `mps`, OpenVINO devices → `cpu` |
| `precision: fp16` | `profile: max_speed`; `native` and `fp32` are dropped (exact is FP32) |
| `custom_ops_profile`, `compilation_cache_dir`, ONNX graph `head` paths | removed (the runtime selects graphs) |
| `embedding_config.backend: candle \| openvino` | `backend: model_runtime`, the default (local embeddings are served by the runtime) |
| Module `variant`, `model_type`, `use_modernbert`, `use_mmbert_32k`; `detector.backend: candle` | removed (the served package defines its architecture and windows) |
| Label mapping files of built-in models | optional: a local consumer takes its labels from the served model card; a file stays a rename override |
| Legacy model paths and aliases (section 16.3) | their Vela 1.0 replacements, with label remaps where labels differ; refused with guidance where no safe remap exists |
| `hallucination_explainer`, `nli_model`, NLI filtering, `use_nli`, the response-cache polarity guard | removed with a warning (retired) |
| `nli_contradiction_penalty` (looper fusion) | `contradiction_penalty`, read from the Halu grounded head |

## 14. Security

- No package code is executed and `trust_remote_code` is never used; model
  code is built into the runtime or installed plugins. ONNX graphs are data
  loaded by ONNX Runtime without custom operators.
- Every loaded byte is verified against the manifest or the built-in file
  digests before load; built-in models are pinned by revision and identity.
- Managed runtimes listen only on a UDS inside a 0700 directory; engine mode
  binds `127.0.0.1` unless `--host` says otherwise.
- Tokens come from the environment or token files, never argv, logs or
  responses. Request text is never logged; metrics carry no request content.

## 15. Testing and CI

| Layer | Where | What |
| --- | --- | --- |
| Unit (CPU, tiny fixtures) | `src/model-runtime/tests` | every family's renderer, readout and answers; ModernBERT, Qwen3 and Qwen3.5 backbones against the Transformers reference; heads, windows, spans; engines and fallbacks; scheduler, profiles, placement, plugins |
| Registry | same | manifests and file digests, tampered, extra, missing and linked files, identity, offline cache |
| API contract | same | every response validated against `openapi.yaml`; System One cases; bundles; error codes; UDS and TCP; multi-model processes |
| Integration | same | `vllm-srun serve` on generated fixture packages of every family, readiness gating, deadlines, overload |
| Router | `pkg/modelservice`, `pkg/modelruntime/serving`, consumers | generated client against an httptest runtime, bundles, process groups with a fake runtime binary, every contract, fail-open, config validation and migration |
| Ports | `pkg/classification`, `pkg/modelselection` | BM25, n-gram, KNN, KMeans, SVM and MLP against fixtures recorded from the bindings they replace |
| E2E (CPU, Kind) | `e2e/profiles/*` | managed and attached runtimes, every migrated signal and feature, decision signals and the selector, fail-open; tiny fixtures |
| Real models (opt-in) | `e2e/profiles/model-runtime-real` | real Kai-0.6B and Vela models on CPU with explicit inputs |
| GPU and parity | records under `docs/records/` | ROCm parity and performance per model; legacy parity per migrated model |

## 16. Inventory and migration (Phases 2–4)

### 16.1 Legacy bindings

| Binding | Implementation | What it provides | Disposition |
| --- | --- | --- | --- |
| `candle-binding/` (214 files, ~111K lines) | Rust candle, cgo | BERT, ModernBERT, mmBERT, DeBERTa sequence and token classifiers; the BERT LoRA trio and Qwen3 multi-LoRA; Qwen3Guard; embeddings (Qwen3, Gemma, mmBERT 2D Matryoshka, BERT, multi-modal-embed-small); NLI; LettuceDetect and Vela Halu grounding; the Vela Reranker pair scorer; owned typed instances; the MLP selector; tokenization and similarity helpers | Models → runtime families; MLP → Go; the rest deleted |
| `onnx-binding/` (191 files, ~49K lines) | Rust `ort`, cgo; CK flash-attention custom op; export scripts | ORT instances for sequence, scores, token, grounded, relevance and embedding graphs, Vela Omni, operating points; CPU, ROCm and MIGraphX providers | Graphs → `onnxruntime` engine; Omni → `multimodal_embedding`; the binding, the CK op and the MIGraphX path are deleted |
| `openvino-binding/` (43 files, ~8K lines) | C++, cgo (build tag `openvino`) | OpenVINO embeddings and sequence classifiers | Retired (decided) |
| `ml-binding/` (15 files, ~3.8K lines) | Rust `linfa`, cgo | KNN, KMeans and SVM selectors loaded from JSON | Pure Go in `pkg/modelselection` |
| `nlp-binding/` (12 files, ~2.4K lines) | Rust, cgo | BM25 and n-gram keyword classifiers | Pure Go in `pkg/classification` |

Router entry points into them: `pkg/modelruntime/native` (61 files: the
owned-model facade for every contract), the legacy singletons in
`pkg/classification` (`classifier_{category,jailbreak,pii}_init.go`,
`unified_classifier*`, `native_task_results.go`, `keyword_classifier*`),
`pkg/modelselection` (`selector.go`, `persistence.go`), test helpers in
`pkg/cache` and `pkg/extproc`, and, outside the router,
`src/training/model_classifier/*_verifier.go`,
`src/training/model_selection/ml_model_selection/validate.go`,
`tools/dev/examples/{redis,valkey}`, `tools/modelcompat` and `perf/`. 204 router
files carry `!windows && cgo` build tags only because of these imports.

### 16.2 Router consumers

| Consumer | Contract | Model today (default) | Backend today | Target |
| --- | --- | --- | --- | --- |
| Domain signal (`domain_classifier`) | `label_distribution.v1` | Vela Domain | candle CPU; ORT MIGraphX (AMD recipe) | `task_heads` sequence, `/v1/classify` |
| Prompt guard / jailbreak signal (`prompt_guard`) | `label_distribution.v1`, windowed | Vela Guard | candle; ORT ROCm graph | `task_heads` sequence, windows |
| PII signal and plugin (`pii_classifier`) | `token_spans.v1`, windowed | Vela PII | candle; ORT | `task_heads` token, windows |
| Fact-check signal (`fact_check_classifier`) | `label_distribution.v1` | Vela FactCheck | candle; ORT | `task_heads` sequence |
| User-feedback signal (`feedback_detector`) | `label_distribution.v1` | Vela Feedback | candle; ORT | `task_heads` sequence |
| Modality signal (`modality_detector`) | `label_distribution.v1` | Vela Modality | candle; ORT | `task_heads` sequence |
| Safety rules (`safety.<rule>`) | `label_distribution.v1` / `label_scores.v1` | Vela Safety, Shield, Hazard | candle; ORT | `task_heads` sequence / scores with operating point and windows |
| Generic classifier rules (`classifier.<rule>`) | `label_distribution.v1` / `label_scores.v1` | any local sequence classifier | candle; ORT; OpenVINO | `task_heads` |
| Hallucination detector (`hallucination_detector`) | `token_spans.v1`, grounded | Vela Halu (LettuceDetect legacy) | candle; ORT | `task_heads` grounded |
| Hallucination explainer, response-cache polarity guard | `text_pair_distribution.v1` | ModernBERT-base-nli | candle | **Retired** |
| Embedding signal, KB signal, complexity candidates, contrastive jailbreak and preference, reask | `embedding.v1` | Vela Embedding (mmBERT), Omni Nano for images | candle; ORT; OpenVINO | `task_heads` pooled; `multimodal_embedding` |
| Semantic cache, memory, vector stores, RAG retrieval, tool selection | `embedding.v1` | Vela Embedding; Qwen3, Gemma, BERT when configured | candle; ORT; OpenVINO; OpenAI-compatible | runtime `/v1/embeddings` (or the external endpoint, unchanged) |
| RAG reranker (`rag.reranker`) | `relevance_scores.v1` | Vela Reranker | candle; ORT (CK flash attention) | `task_heads` relevance, `/v1/rerank` |
| Complexity (`complexity`) | `score.v1` | remote only | HTTP | unchanged |
| Decision signal and selector | `/v1/decisions` | Decision 2.0 | runtime (Phase 1) | unchanged, plus Decision 1.0 and Vela 2.0 |
| Keyword signal | — | BM25, n-gram, fuzzy, regex | nlp-binding; Go | pure Go |
| Model selection (`knn`, `kmeans`, `svm`, `mlp`) | — | trained JSON selectors on query embeddings | ml-binding; candle MLP | pure Go; embeddings from the runtime |
| Diagnostics and classification APIs (`/api/v1/diagnostics/...`) | all | the bound models | native facade | `serving` facade |
| Language, structure, context, conversation, event, authz, metadata signals | — | heuristics, `lingua-go` | Go | unchanged |

### 16.3 Models and dispositions

| Model (router alias) | Used for | Disposition |
| --- | --- | --- |
| Vela 1.0 Domain `f6354f54`, Guard `087f9e40`, Safety `6e70e725`, Shield `a981a99e`, FactCheck `99ede1ab`, Feedback `47434a7f`, Modality `5384b899`, Hazard `5dd25f2c`, PII `6d3300c4`, Halu `ca875312`, Embedding `1e57cebf`, Reranker `a388e41c` | the defaults | **Migrate** to `task_heads` at the same pins, with parity records |
| Vela 1.0 Omni Nano `2ff2d663`, Mini `801bae3a` | multimodal embeddings, image routing | **Migrate** to `multimodal_embedding` |
| Vela 1.0 Encoder-307M `fe9ccc07` | training parent (not served) | Kept in the registry for adaptation; not loaded |
| BERT LoRA trio (`lora_intent`, `lora_pii`, `lora_jailbreak`), ModernBERT category / jailbreak / PII classifiers, mmBERT-32K merged and LoRA classifiers (intent, fact-check, jailbreak, feedback, PII, modality), `halugate-sentinel`, `feedback-detector` | earlier defaults and aliases | **Replace** with the Vela 1.0 task model of the same purpose; `config migrate` remaps labels where they differ (feedback gains `NO_FEEDBACK`; modality keeps `AR` / `DIFFUSION` / `BOTH`) and refuses rules it cannot remap safely |
| LettuceDetect v1 / v2 | hallucination detector | **Replace** with Vela Halu (same grounded input); LettuceDetect v2 mmBERT also loads as a third-party `task_heads` grounded model |
| ModernBERT-base-nli | explainer, polarity guard | **Retire** (decided) |
| `mmbert-embed-32k-2d-matryoshka` | embeddings | **Replace** with Vela Embedding (its successor; re-embedding required) |
| Qwen3-Embedding-0.6B | embeddings (`model_type: qwen3`) | **Migrate** as a `task_heads` pooled model on the native Qwen3 backbone (last-token pooling), or use an external OpenAI-compatible endpoint |
| EmbeddingGemma-300m, all-MiniLM-L12-v2 | embeddings (`gemma`, `bert`) | **Replace**: Vela Embedding by default, or an external OpenAI-compatible endpoint (`vllm serve`); re-embedding required |
| multi-modal-embed-small / large | multimodal embeddings | **Replace** with Omni Nano / Mini (re-embedding required) |
| Qwen3Guard via `http_chat` | prompt guard | **Keep external** (unchanged) |
| candle Qwen3 multi-LoRA, candle Qwen3Guard, DeBERTa `auto` | nothing wired | **Retire** |
| Decision 1.0 (seven packages) | not served by the router today (proposal #4086) | **Add** via `decision1` |
| Vela 2.0 0.3B, 0.8B, 4B, 9B | new | **Add** via `vela2` |

### 16.4 Algorithms that leave the bindings

| Algorithm | Today | Target | Evidence |
| --- | --- | --- | --- |
| BM25 and n-gram keyword scoring | `nlp-binding` | Go in `pkg/classification` | Scores and matches equal on fixtures recorded from the binding |
| KNN, KMeans, SVM (linear and RBF) inference from trained JSON | `ml-binding` | Go in `pkg/modelselection` | Selections equal and scores within 1e-9 on the pretrained test models |
| MLP selector inference | candle | Go in `pkg/modelselection` | Selections equal and scores within 1e-6 on the pretrained test models |
| Regex provider | Go inside `candle-binding` (unused) | deleted | — |

### 16.5 Platform surfaces

| Surface | Today | Target |
| --- | --- | --- |
| Images | `tools/docker/Dockerfile.extproc{,-rocm}`, `src/vllm-sr/Dockerfile{,.cuda,.rocm}`, `deploy/operator/Dockerfile`, `onnx-binding/Dockerfile.rocm`, `openvino-binding/Dockerfile` build Rust bindings, ORT, MIGraphX or OpenVINO | Pure-Go router; the runtime with CPU, ROCm or CUDA PyTorch, without ONNX Runtime; no prepared Omni bundle (the runtime downloads Omni like every model); binding images deleted. One `tools/docker/Dockerfile.extproc` builds all five images (`ACCELERATOR`, targets `extproc` and `vllm-sr`); sizes, build times and startup in `docs/records/removal-footprint.md` |
| Make | `rust.mk`, `openvino.mk`, `build-run-test.mk` binding targets, `models.mk` native model tests, `common.mk` library paths | Deleted or pointed at the runtime |
| Workflows | `test-native.yml`, `build-native.yml`, `publish-crate.yml`, the native lanes of `ci.yml`, `performance-test.yml`, Rust hooks in `pre-commit.yml` | Deleted, or replaced by the model-runtime lanes |
| Helm and operator | model download init containers, native library environment, provider settings | The runtime ships in the router image; optional attached runtime sidecar values |
| E2E profiles | `vela-halu`, `vela-shield`, `vela-omni`, `multimodal-routing`, `local-classifier-backend`, `ml-model-selection`, `hallucination`, and every profile that loads default models through the router image | Run against the runtime; new runtime profiles (section 15) |
| Harness | `tools/agent/domains.yaml` domains `native-bindings`, `openvino`, `riscv-runtime`, `onnx-*`, `ck-flash-attn-rewriter`, `published-model-tests`, native verifications | Removed; `model-runtime` domain covers the families, engines and parity tools |
| Docs | runtime, installation, AMD, OpenVINO and Vela pages describing candle, ORT and OpenVINO | Task-oriented runtime docs and a migration guide |
| RISC-V | candle on riscv64 under QEMU, a riscv64 router build and its CI lane | **Removed** (maintainer decision): no riscv64 build, lane, image or build tags |

### 16.6 Retirements

- **NLI** (`tasksource/ModernBERT-base-nli`): the hallucination explainer,
  `enable_nli_filtering`, and the response-cache polarity guard are removed.
  Hallucination detection keeps Vela Halu spans.
- **OpenVINO provider** and `openvino-binding`.
- **ONNX Runtime binding paths specific to the router** (MIGraphX shape
  buckets, the CK flash-attention operator and its rewriter): the runtime's
  ROCm path is native PyTorch; ONNX Runtime remains as an optional portable
  engine.
- **Unwired legacy models** (candle Qwen3 multi-LoRA, Qwen3Guard, DeBERTa).
- **RISC-V** (maintainer decision): the riscv64 router build, its QEMU lane,
  build tags and fallbacks that existed only for riscv64.

## 17. Parity and evidence

| Model group | Reference | Device classes | Pass thresholds |
| --- | --- | --- | --- |
| Decision 2.0 | released package runtime | CPU, ROCm | Bit-identical answers (unchanged) |
| Decision 1.0 | the packages' bundled runtime, same requests | CPU, ROCm | Bit-identical on the same device class, else every answer within 1e-6 and recorded |
| Vela 1.0 sequence and scores heads | the legacy router path (candle CPU; ORT on the AMD recipe), same inputs | CPU (CI fixtures and real models), ROCm | CPU: label agreement 100% outside near ties (top-two margin under 1e-3), max \|Δp\| ≤ 1e-3; ROCm: label agreement ≥ 99.5%, max \|Δp\| ≤ 0.02 |
| Vela 1.0 token and grounded heads | the legacy router path | CPU, ROCm | CPU: identical span sets on ≥ 99.5% of inputs, max \|Δp\| ≤ 1e-3; ROCm: identical span sets on ≥ 98% |
| Vela 1.0 embeddings and reranker | the legacy router path | CPU, ROCm | CPU: cosine ≥ 0.99999 per vector, max \|Δ\| ≤ 1e-4, identical rerank order outside ties; ROCm: cosine ≥ 0.9995 |
| Omni | the published reference implementation's outputs on its reference inputs, and the legacy ORT path | CPU, ROCm | CPU: cosine ≥ 0.99999; ROCm: cosine ≥ 0.9995 |
| Vela 2.0 | the packages' engine (`vela2_inference.py`), same requests | CPU, ROCm | Decisions and span sets identical; max \|Δp\| ≤ 1e-4 on CPU, ≤ 0.02 on ROCm |
| Ported algorithms | the binding they replace | CPU | Section 16.4 |

The legacy driver calls the legacy router's diagnostics API
(`/api/v1/diagnostics/...`), built from `main` before this change, so the
reference is exactly what the router served. Each record states the commit,
image, device, inputs and every disagreement. CUDA stays implemented,
unit-tested and unvalidated.

## 18. Performance

Every migrated feature must match or beat the legacy binding's latency (p50,
p95) and throughput on CPU and on ROCm, measured on the same inputs and
hardware, with the records in `docs/records/<workstream>-*`. The techniques:

| Where | Technique | Records |
| --- | --- | --- |
| Decision models | exact-shape HIP / CUDA graphs with host-built masks, bit-exact fused Triton kernels (FP32 and BF16 streams on gfx942), lean LoRA, shared-context trees (prefix plus GDN state hand-off), cross-request batching, pinned kernel choices, the 2 GiB guard; Decision 1.0 encoders keep one graph set and one reduced copy per layer stack | `rocm-mi325x-*`, `decision1-performance.md`, `vela2-performance.md` |
| Runtime core | one bundled call per request and process; a bundle's tasks for one model as one job group (one forward for every head reading the same input); per-model content-hash result cache; shortest-expected-finish scheduling that answers each job when its own batches ran and lets short requests run between the windows of a long one (section 9); package files verified in parallel | `router-latency-cpu.md` |
| Encoders | packed (varlen) attention in length groups; banded local attention for long rows (from 1,024 tokens on CPU, 2,048 on GPU); oneDNN pre-packed FP32 linears on CPU `exact`, batch-invariant as probed at load, so `exact` batches concurrent requests; dynamic cross-request batching (`batching`); encoder graphs per shape bucket on GPU, replayed only when padding stays small; a fused gfx942 rotary kernel; reduced-precision copies under `max_speed` only where the records show at least 99% agreement (CPU `float32-packed` for Decision 1.0 Kai, Lex and Route and the Vela 2.0 0.3B; BF16 and int8 measured and refused elsewhere); Omni's four towers on the CPU's one OpenMP team, with NumPy's OpenBLAS on one thread; per-hardware kernel registry | `vela1-performance.md`, `embed-performance.md`, `decision1-performance.md`, `vela2-performance.md` |
| Router | parallel signal goroutines with one `/v1/bundle` per request stage and runtime process; a per-deployment result cache; one CPU process per model with thread shares; deadlines and fail-open; pure-Go keyword scoring and model selectors (AVX2 / FMA and NEON dot kernels with a pure-Go fallback) | `router-latency-cpu.md`, `router-latency-rocm.md`, `stores-algorithms.md`, `stores-consumers.md` |
| Router to runtime | HTTP/JSON on a Unix socket, timed per call from the runtime's `Server-Timing`: on CPU the transport is 0.6–2.2% of a call, so there is no binary fast path | `runtime-transport-cpu.md` |

## 19. Phase 1 follow-ups and later work

Done: ROCm golden answers and pinned kernel choices for all six Decision 2.0
models on gfx942, and the graph-cap and batch-shape bucket measurements
(`docs/records/rocm-mi325x-repeatability.md`).

Later: a serving option for the graph cap (1,024 is 1–5% faster under
Index-like traffic); kernel choices for other device classes; more `max_speed`
kernels; a `vllm` engine (pooling runner plus the family readouts) and a
`llamacpp` engine; CUDA validation on CUDA hardware.

Router `decision` signals ask Choice, Noul and Score questions, which every
decision family answers, Vela 2.0 included. Set and Span answers are served
on `/v1/decisions` to any client; signal rules over them (a label of a Set,
a span label present) are later work, together with Vela 2.0 presets as
drop-in task bindings (`pii`, `halu`, `relevance`).

## 20. Risks

| Risk | Mitigation |
| --- | --- |
| A ported backbone or head drifts from the scored numerics | Fixtures against the Transformers reference in CI; parity records against the legacy path and the bundled runtimes on real models |
| CPU latency of the encoders regresses against candle | Measured per model; the `onnxruntime` engine runs the packages' graphs where it is faster; intra-op threads and inter-model parallelism are tunable |
| Bundles add latency when one task is slow | Tasks run in parallel per model; a bundle waits only for its own tasks; consumers keep their own timeouts |
| A shared process couples fault domains | `process` groups isolate deployments; a crash restarts one group; fail-open |
| Removing the bindings breaks unmigrated configs | The parser points to `vllm-sr config migrate`; the migration guide lists every change |
| Re-embedding after an embedding-model change | Representation identity in every cache and store key; old vectors are never silently reused |
| Image size grows with PyTorch | CPU wheels in CPU images; GPU wheels only in GPU images; the Rust toolchain stages go away |
| Large models outgrow a device | Placement refuses to overcommit; Vega-27B needs one ≥ 64 GB device |
