# vLLM Semantic Router Model Runtime

Design for the built-in model runtime of vLLM Semantic Router (vllm-sr). The
runtime serves every model the router runs: Decision 3.0, 2.0 and 1.0, Vela 1.0 and
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
2. **Throughput first, checked against the release.** Like vLLM, the runtime
   batches concurrent requests by default. A request's reference forwards
   (alone, in the released split) reproduce the runtime inside the released
   Decision 2.0 packages bit for bit, and the golden check compares against
   them at startup. Migrated models (Decision 1.0, Vela 1.0) carry parity
   records against the path that served them before. Any other faster path
   that changes numerics is on by default only where its accuracy record shows
   no loss; elsewhere it is an engine argument (section 5.4). Engine arguments
   replaced the original design's opt-in numerics profiles.
3. **Pluggable.** Model families, execution engines and hardware accelerators
   are separate plugin layers. Third parties add a family, an engine or an
   accelerator through Python entry points without patching the runtime.
4. **One contract.** The router and every client talk to the runtime through a
   checked-in OpenAPI schema with five surfaces: decisions, classify,
   embeddings, rerank and bundle. The Go client is generated from it.
5. **One call per request.** A runtime process can serve several models, and
   the router sends all of a request's model work for one runtime as one
   bundled call.
6. **Universal hardware, honestly labelled.** CPU, ROCm, and CUDA are
   validated. CUDA measurements are under `docs/records/cuda-*.md` (H200,
   A100, Tesla T4). Vela 2.0 0.3B at 32 tokens is a known p50 regression on
   H200. Fused kernels without bit-exact evidence stay excluded from the
   exact profile; that is a valid outcome of item 4.
   `CUDAAccelerator.validated` is true. XPU and MPS are plugins marked
   unvalidated.
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
| Numerics | A request's reference path is byte-identical to the released packages' runtime. Cross-request batching is the default (section 5.4); shared context and faster kernels each have a measured accuracy record and are on by default only where it shows no loss. |
| Router | A `decision` signal and a `decision` selector, plus the router-managed lifecycle. Fail-open: a late or failed answer leaves the signal unknown and the selector falls back. |
| Hardware | ROCm (MI300 / MI325X), CPU, and CUDA validated. CUDA records are under `docs/records/cuda-*.md`. |

### 2.2 Phases 2–4

| Area | Decision |
| --- | --- |
| Delivery | Phases 2, 3 and 4 in one PR ([#4496](https://github.com/vllm-project/semantic-router/issues/4496)), commits split by module. |
| Families | `decision1` (Decision 1.0), `task_heads` (Vela 1.0 and compatible HF encoder task models), `vela2` (Vela 2.0), `multimodal_embedding` (Vela 1.0 Omni). `decision2` is unchanged. |
| Surfaces | `POST /v1/classify`, `/v1/embeddings` (OpenAI-compatible), `/v1/rerank`, and `/v1/bundle`; Set and Span answers on `/v1/decisions` where a model declares them. The Phase 1 surfaces are unchanged. |
| Processes | One runtime process serves one or more models. Each model keeps its own worker, device and engine arguments. The router groups its managed deployments into processes (by default one per device). |
| Bundling | The router sends all model work of one request stage for one runtime process as one `/v1/bundle` call, and every question the stage asks one model as one decisions task, about one state or several (`states`). Bundling is transport only: every task keeps the semantics of its own surface. |
| Engines | The native PyTorch engine first, for encoders and decoders, on every accelerator; every built-in model runs on it by default. An optional `onnxruntime` engine (the `onnx` extra, not in the router images) runs the ONNX graphs packages ship (Vela 1.0, Vela 2.0 0.3B) and prepared Omni bundles, for portability. |
| Hardware | Accelerators `cpu`, `cuda`, `rocm` (built in) plus `xpu` and `mps` (built-in plugin slots, unvalidated). |
| Router seam | The router's typed task bindings (`pkg/modelruntime/binding`, contracts `label_distribution.v1`, `label_scores.v1`, `token_spans.v1`, `embedding.v1`, `relevance_scores.v1`) are kept. The provider facade `pkg/modelruntime/native` (candle, ORT, OpenVINO) is replaced by `pkg/modelruntime/serving`, which implements every contract through the runtime. Consumers change their constructor, not their logic. |
| Algorithms | Keyword matching (BM25, n-gram) and the KNN, KMeans, SVM and MLP model selectors are algorithms, not models: they become pure Go with parity tests against the bindings they replace. |
| Config | The canonical layout stays. `provider: model_runtime` deployments may serve task bindings. The providers `candle`, `ort` and `openvino` and their execution fields are removed from the parser; `vllm-sr config migrate` rewrites them. |
| Retirements | The NLI model (hallucination explainer, response-cache polarity guard) and the OpenVINO provider retire, as decided in the runtime proposal. Every other legacy model migrates or has a documented replacement (section 16). |
| Reference path | Decision 2.0's stays byte-identical. Decision 1.0's targets bit-identity with the packages' bundled runtime on the same device class. Vela 1.0's is FP32 on every device, with parity records against the legacy path. |
| Images | Every router image ships the runtime (CPU PyTorch in CPU images, the ROCm or CUDA wheel in GPU images), so the managed lifecycle works everywhere. Its other dependencies install at the versions in `requirements-lock.txt`, so every build of a release gets the same set. Model-backed validation of the image runs in development CI: recipe conformance serves the maintained recipes' models through the CI-built image, and Published Models serves the published Vela models through the runtime from source. The release workflow rebuilds the tagged source without running tests (#4273), so the published digest itself is not model-run. The Rust build stages are deleted. |

## 3. Overview

```text
                 +------------------------- vllm-sr router (Go, no cgo models) ------------------------+
 request ------> | signals (domain, guard, PII, ..., decision) --> decision rules --> selector --> backend|
                 | caches, memory, vector stores, RAG, tools, hallucination, model selection            |
                 |   pkg/modelruntime/serving: typed task bindings (label distribution, scores, spans,  |
                 |   embeddings, relevance) over pkg/modelservice                                       |
                 |   pkg/modelservice: generated client, request bundles, managed workers,              |
                 |   attached endpoints, per-deployment readiness, fail-open                            |
                 +---------------------------------|---------------------------------------------------+
                                                   | HTTP/JSON over UDS (managed) or TCP (attached)
                                                   v
 +------------------------------ vllm_srun (one Python process, one or more models) ---------------+
 | api: /v1/decisions /v1/systemone /v1/classify /v1/embeddings /v1/rerank /v1/bundle /v1/models /health |
 | per model: scheduler (admission, deadlines, batching, micro-batches) and one worker on its device     |
 | families: decision3 | decision2 | decision1 | task_heads | vela2 | multimodal_embedding (render, readout)|
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
  requirements-lock.txt     # dependency versions every image installs, PyTorch's excepted
  docs/
    design.md               # this document
    records/                # parity and performance records
  vllm_srun/
    __main__.py, cli.py     # `vllm-srun serve ...`; `vllm-sr serve <hf-model>` delegates here
    config.py               # ServeConfig (process options), ModelConfig (one served model), EngineArgs
    dtypes.py               # --dtype: the published precision, and the policy an explicit dtype runs
    runtime.py              # the process: models by served name, surface dispatch, bundles
    api/
      openapi.yaml          # THE contract (served at /openapi.yaml; the Go client is generated from it)
      app.py, server.py     # Starlette routes; uvicorn (uvloop, httptools) over TCP or UDS
    errors.py               # error codes shared by API, scheduler and families
    systemone.py            # System One request semantics shared by every decisions family
    text/                   # shared text handling: the package tokenizer, input policies and windows
    heads/                  # shared readout heads (candidate, sequence, scores, token, pooled, relevance, span)
    plugins/
      base.py               # ModelFamily, LoadedModel, Engine, EngineModel, Accelerator, specs
      decisions.py          # DecisionModel: the decisions surface, golden comparison and outcomes per question
      registry.py           # entry-point discovery and selection
    registry/
      resolve.py            # local directory or Hub repo at a pinned 40-hex revision, cache
      artifacts.py          # file inventories and hash verification for packages without a manifest
      builtin.py            # the first-party models: the tables the installed families name
      tables/               # the built-in families' tables: decision3.py, decision2.py, decision1.py, vela1.py, vela2.py, omni.py
      golden_answers*.json  # golden references per family, per device class
      kernel_choices.json   # pinned autotuned kernel choices (Qwen3.5 decoders)
      policy.py             # licence and access policy, token handling
    scheduler/              # planner, batching (the batcher the engine arguments set), scheduler (one per model)
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
    testing/
      fixtures.py           # tiny random-weight packages through each family's writer, and a tiny tokenizer
      <family>.py           # the built-in families' fixture writers
  examples/third_party_plugin/  # a documented out-of-tree family, engine and accelerator, installed by the tests
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
| `vllm_srun.families` | `ModelFamily` | `decision3`, `decision2`, `decision1`, `task_heads`, `vela2`, `multimodal_embedding` |
| `vllm_srun.engines` | `Engine` | `native`, `onnxruntime` |
| `vllm_srun.accelerators` | `Accelerator` | `cpu`, `cuda`, `rocm`, `xpu`, `mps` |

Every plugin class has a capability descriptor (`descriptor()`): a family's
surfaces and package formats, an engine's architectures and outputs, an
accelerator's validation status; engines also list the `--dtype` values
they run, and engines and accelerators their `auto_priority`. `/v1/models` lists each active
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
deadlines and batching:

```python
class LoadedModel(ABC, Generic[ItemT, ResultT]):  # ItemT satisfies WorkItem
    info: ModelInfo                        # identity, surfaces, heads, limits, licence
    batch_invariant: bool                  # set at load: each row answers the same alone or in any batch
    shared_context_default: ClassVar[bool] # shared context when the engine arguments leave it unset
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
scheduler and its batcher, the runtime core, the API, the registry, the shared
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
positions (decision endpoints and query rows) and the shared-context prefix;
with `packed`, an engine that `supports_packed` runs the rows back to back in
one row instead (`engines/native/models/packed.py`: variable-length attention,
the chunked gated delta rule over `cu_seqlens`, each causal convolution from a
zero history), which only merged forwards ask for.
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
reproduce and whether it is bit-exact; a kernel that is not bit-exact runs
only for a family that consents to approximate kernels
(`DtypePolicy.approximate_kernels`), at `--dtype auto`. `xpu` and `mps` detect `torch.xpu` and the
MPS backend and run the pure-torch references; they are marked unvalidated.

`Accelerator.golden_reference(device)` names an optional hardware-qualified
golden reference. Its default is `accelerator:arch`; CPU adds PyTorch's selected
instruction set and native or emulated BF16 capability, for example
`cpu:x86_64:avx2:bf16_emulated`. Device inventory architecture names stay unchanged.

`tools/golden_answers.py --record` writes both the generic device-class baseline
and any qualified reference. The first generic baseline is preserved within a
revision; later hardware runs update only their qualified entries. A revision
change clears all old references. Record the new revision on the intended
baseline device before adding other hardware classes.

### 5.4 Engine arguments

A served model batches and computes as its engine arguments say (`EngineArgs`,
`config.py`), named after vLLM's:

| Argument | Default | What it does |
| --- | --- | --- |
| `max_num_seqs` | by device | Most rows one forward holds, across requests: one per question of a decision request, one per input or window otherwise; 1 runs every row alone |
| `max_num_batched_tokens` | by device, calibrated | Tokens one forward may hold: the sum of its rows (each rounded up to 8) for a model that packs merged rows (`packs_merged_rows`), else its rows times its longest row |
| `batch_window_ms` | 2.0 | How long the scheduler collects requests after the first, before it plans the queue |
| `dtype` | `auto` | The precision the backbone computes in; `auto` is the one the model was published in |
| `enable_shared_context` | the family's | Shared-context trees for multi-question decision requests; on by default only where the family's records show no accuracy loss (Decision 2.0, and state-once Decision 3.0 packages, whose requests always branch from their state: there it only decides whether whole requests share forwards) |

Limits left unset take the device's memory class (`scheduler/calibration.py`:
32,768 tokens and 256 rows at 160 GB and more, 16,384 / 128 from 64 GB,
8,192 / 64 from 32 GB, 4,096 / 32 below; the CPU and a device that reports no
memory, such as MPS or a plugin's accelerator, keep 65,536 / 8,192). On a
GPU the runtime then calibrates the token budget after the golden check, from
merged forwards of the model's calibration rows (`calibration_items`; a
state-once package's rows each branch from a state of their own, as a burst of
one-question requests does) at doubling budgets: the largest budget whose cost
per token is within 2% of the cheapest timed (on a flat curve the smallest
would follow timing noise), but never more than one forward does in 0.3 s
(interpolated between the timed forwards) nor than 85% of the free device
memory holds. It logs the chosen values and their source, and `engine_args`
reports them.

The batcher (`scheduler/batching.py`) plans the rows of every queued request
together (`coalesced`), one lane at a time (`LoadedModel.lane`; Decision 3.0:
text rows and rows that read media), with a token budget per lane
(`coalesced_token_budget(lane)`). Padded forwards take rows longest first; a
model that packs merged rows fills forwards in arrival order and counts their
summed length. The n-th batches of the lanes share one turn (`Batch.sub`): the
scheduler runs each as its own forward, the one expected to take least first,
and answers its rows as soon as it ran, so a planning round adds as many turns
as its busiest lane needs. A row's padding, neighbours and GEMM shapes depend
on the rows beside it, so an answer can move by rounding. A request here is one
forward, with no later step where it could join others, so the batching window
is what lets a burst share it. A
batch-invariant model (`LoadedModel.batch_invariant`, set at load from a test
that every row answers the same alone or in any batch) merges the requests
already queued without waiting and keeps forwards within 512 padded tokens
(`merged`), and its answers never change. The golden check plans its requests
as references: each alone, in the released split (`reference`), so readiness
compares like with like. A card and response meta report `numerics`: `exact`
for a batch-invariant model at `--dtype auto` without shared context or
approximate kernels, `approximate` otherwise.

At `--dtype auto` an encoder also loads the reduced copy of its linear layers
its family consents to, next to the weights the reference forwards keep
using, and batches that merge requests run it (`run_approximate`). A family
consents (`DtypePolicy.reduced_gpu` / `reduced_cpu`) only where its records
show no decision changed; faster alone is not enough. Today that is the
oneDNN-packed FP32 copy on x86 CPUs (Decision 1.0 Kai, Lex and Route, Vela
2.0 0.3B), which keeps the published precision; the BF16 and int8 kinds
(`engines/native/reduced.py`) were measured and refused. Approximate kernels
follow the same rule per question type (`DtypePolicy.approximate_kernels`):
Decision 2.0 consents to CUDA's approximate fused kernels, and Vela 2.0 does
not, because they change 7% of its span answers. Another `--dtype` holds every
backbone weight in that dtype and runs the reference kernels without copies;
the golden references apply only to `auto`, so its golden status is
`unverified` (`dtypes.py`). Each family records the accuracy (label
agreement, max |Δp| or embedding cosine against the reference path) and the
latency of its reduced path. `vllm-sr config migrate` drops the legacy
`precision: fp16` (the model runs in its published precision) and maps a
former `profile` onto these arguments.

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
points. The CLI accepts any registered device. Plugins run in the runtime process; the runtime never executes code
from a model package. `examples/third_party_plugin` is a complete, documented
example (a keyword family, a counting engine and a host accelerator). The test suite builds its wheel with its own build
backend, installs it into a fresh directory, discovers it through the entry
points its `pyproject.toml` declares, serves it on every surface next to a
Decision 2.0 model, and serves it on the example accelerator.

## 6. API

The contract is `vllm_srun/api/openapi.yaml` (OpenAPI 3.0.3), served at
`GET /openapi.yaml`, checked by contract tests on both sides, and the source of
the Go client in `pkg/modelservice/api`. Every surface takes an optional
`model` (the served model ID, required when a process serves several models)
and `options` with `deadline_ms` and `return_meta` (the runtime's `meta`:
revision, dtype, numerics, engine, device and timings, only when asked;
a family's own fields such as `meta.representation` are always returned).
Every surface and bundle response, errors included, carries a
`Server-Timing` header with the request's phases (`parse`, `tokenize`,
`queue`, `forward`, `post`, `serialize`) and their `total` in milliseconds
(`vllm_srun/timing.py`); the router subtracts the total from its own time
for the exchange to record each call's transport.
A request-level error uses
an HTTP status with `{"error": {"code", "message"}}`: 400 `invalid_request`,
404 `model_not_found`, 413 `request_too_large`, 415 `unsupported_media_type`,
422 `unsupported_surface` (the model does not serve this surface), 429
`overloaded`, 503 `not_ready`. A forward out of device memory is 413
`request_too_large` for a request that had it alone and 429 `overloaded` for
one that shared it (`scheduler.py`). Item-level failures
(`max_length_exceeded`, `invalid_input`, `invalid_model_output`,
`deadline_exceeded`, `unavailable`) are reported in place with a fixed
`reason` and never fail sibling items, except on `POST /v1/systemone`, where
the first one answers the request (`question_failed`).

### 6.1 `POST /v1/decisions`

Phase 1's System One questions, extended. `POST /v1/systemone` takes the same
requests and answers a request with a failed question as a whole, as System
One does; the reference page's System One compatibility section documents its
bodies, assets, Box and long context. Every family validates the System One fields through one reader
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
labels), embedding and rerank descriptors, limits, licence, engine
arguments, numerics, engine, accelerator, device, dtype, plugin versions, and per-model readiness and
golden status, plus the process's `limits` (`max_bundle_tasks`,
`max_request_bytes`). `/health` is readiness (`Health`): 200 only when every
model is ready, with the process state (`starting`, `loading`, `warming`,
`ready`, `degraded`, `failed`) and, when the process serves several models,
each model's state. `/health/live` is
liveness (`Liveness`): 200 with `status: alive` while the process serves
HTTP, whatever its models' states. All three report `api_version`, the
contract version (`info.version`). A router and its runtime come from one
release, so the router accepts only a runtime of exactly its contract version.
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
| `decision3` | `vllm-sr/{d3, d3-flash, d3-mini, d3-nano, d3-lite, d3-edge}` | d3 `cbff3a3c`, flash `521525ef`, mini `a475367d`, nano `3aeb1c22`, lite `a8d25a91`, edge `18d61df7` (only the files the family loads are fetched) |
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
ties, P(true), expected level), with System One's published Choice and Score
confidence, as every family reports it. The native
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
calibration and System One answer assembly. Its reference path targets
bit-identity with the bundled runtime on the same device class; the parity record runs both on
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

### 8.6 Decision 3.0 (`decision3`)

Decision 3.0 packages (`d3-package-manifest/1`) are a Transformers
`Qwen3_5Model` checkpoint with its vision tower, a 255-way answer-code readout
and `decision_config.json`. The family verifies every file it loads against
`MODEL_MANIFEST.json` and recomputes the scored identity (the weight, readout,
tokenizer and chat-template digests plus the inference fields of
`decision_config.json`); the bundled `d3_*.py` runtime is never fetched for a
built-in model and never imported.

One question is one prompt: the chat-templated system and user turns, with the
options listed under single-token answer codes, and the generation prompt with
thinking off. The family renders the chat format itself for the templates it
knows by digest, and builds the tokenizer as Transformers 5.17 builds
`Qwen2Tokenizer` (the Qwen2 pre-tokenizer pattern and every special token of
`tokenizer_config.json`), which the released runtime tokenized with. The
backbone holds BF16 parameters on every device and runs without autocast; its
full-attention layers see the whole left-padded prompt (`attention_mode:
noncausal_full_attention`) while the Gated DeltaNet layers stay causal. The
layers follow the package's `layer_types`, so a pruned backbone (d3-edge keeps
12 of d3-lite's 24 layers, whose `full_attention_interval` no longer describes
them) loads as it is. The FP32 readout scores the question's codes at the last
position, and the answer is the softmax over them at the package temperature.
A request's questions run in request order, eight per forward.

A request's `images` (base64 PNG, JPEG or WebP data URLs) are decoded by PIL
and processed as the Transformers 5.17 torchvision `Qwen2VLImageProcessor`
processes them: `smart_resize` to 65,536 to 1,638,400 pixels, a uint8 bicubic
antialiased resize, one FP32 normalization and the patch layout. Its `videos`
(base64 MP4, WebM, QuickTime or Matroska data URLs) are decoded as the d3
runtime decodes them, by OpenCV's FFmpeg backend from a temporary file: 2
frames per second, at least 4 and at most 32 spread evenly over the whole
video, converted to RGB, an odd count padded with the last frame. The
Transformers 5.17 `Qwen3VLVideoProcessor` path follows with the d3 runtime's
budget: at most 200,704 pixels per frame, the same resize and normalization,
and the patch layout over each pair of frames. The videos of a request take at
most 16,384 input tokens.

Images and videos go in front of the text of every question, images first. In
an image request, each question of a forward carries its own copy of the
images through the native Qwen3.5 vision tower (the matrix-product patch
embedding, bilinearly resampled learned positions, axial rotary attention per
image and the patch merger). In a request with videos, the tower reads the
request's images in one call and its videos in another, once per request, and
every question reuses those features, as the d3 runtime does. Each video's
placeholder expands to its frame pairs, each after its timestamp (`<0.2
seconds>`); a frame pair is an image of one temporal patch for the
multimodal rotary positions, and image features replace the image placeholders
before video features replace the video placeholders. On gfx942 the depthwise
convolution is the `fp64_naive` variant (MIOpen's naive kernel, which the
released runtime ran) and the fused decoder layers run on the BF16 stream.
Each model pins the FLA kernel choices of its release
(`registry/kernel_choices.json`, from the autotune cache its release runs
used), so every process rounds as the release did (§11).

Repeated media are read once, whatever the engine arguments, and without
changing an answer: image and video data URLs are keyed by the SHA-256 of their text,
and a hit reuses the processed patch rows; the tower's features of a
request's images are kept by the images' digests and the number of copies
one tower call holds, which is what a pass of that many rows runs, so later
passes of that size reuse them. Batched, text rows and media rows of
concurrent requests are planned in lanes of their own, each sized to about 0.3
seconds of device time at its own measured cost per token
([record](records/decision3-batching.md)). Where the engine packs rows, the
tower reads the images of every request of a media forward in one call with
one variable-length attention per block (features cached apart from the exact
path's), and a merged pass runs its rows packed when left-padding them would
add more than 10% to its tokens ([record](records/decision3-lanes-packing.md)).
Request images and videos are decoded in the process's media worker processes
(`vllm_srun/media.py`, `--media-workers`).

A package may hold its projections quantized, declared by a compressed-tensors
`quantization_config` in `config.json` (`quantization.py`): FP8 weights with
one scale per output channel and FP8 activations per token (`float-quantized`,
FP8_DYNAMIC), or 4-bit integer weights in groups, symmetric (GPTQ) or with
zero points (AWQ), and BF16 activations (`pack-quantized`), or NVFP4 weights
and activations: FP4 in groups of 16 with FP8 block scales under a global
scale, activations quantized per request (`nvfp4-pack-quantized`). Any other
scheme, including every other FP4 variant, is refused. The targeted layers
become `Float8Linear`, `Int4Linear` or `NVFP4Linear` on the meta backbone and
load the stored tensors as they are (`engines/native/quantized.py`): FP8 runs
`torch._scaled_mm` after one Triton kernel quantizes the activations per row,
and needs FP8 matrix units; int4 unpacks its weights to BF16 with one Triton
kernel per call (plain torch on a CPU) and multiplies in BF16, which keeps the
memory but not the time; NVFP4 quantizes the activations with one Triton kernel
that writes their scales in cuBLAS's layout and multiplies in
`torch.nn.functional.scaled_mm` with 1x16 block scales, which needs FP4 tensor
cores (a Blackwell GPU). At load, NVFP4 projections of one input (q / k / v,
gate / up) whose input and weight global scales agree are merged into one
layer (`merge_siblings`): one quantization and one product per group, each
part keeping views of the merged weight so device memory stays the same, and
its outputs reaching the fused kernels as column-slice views. Projections that
share only the input global scale (`in_proj_qkv` / `in_proj_z`) share its
quantization (`project`). Each column is the one its separate product gives, so
answers do not change. With approximate kernels on compute capability 12.x,
graph captures of up to 512 rows multiply in FlashInfer's CUTLASS FP4 GEMM
instead, where FlashInfer is installed and can tune its tiles (it compiles a
delay kernel for its tuner with a CUDA 12.9 or later toolkit), each weight
shape checked against cuBLAS first (`accel/flashinfer_nvfp4.py`,
`use_flashinfer`); the receipt's `nvfp4_product` says which runs and why. A
quantized package is its own model: its manifest's parameter counts are the
logical ones, placement counts its files, and its answers are compared with
the BF16 package's in `docs/records/decision3-quantized.md`.

**State-once packages.** A package may declare an `attention_mode` under
which no state token attends to a question (`StateOnceModel`,
`families/decision3/state_once.py`):

| `attention_mode` | Full-attention layers |
| --- | --- |
| `prefix_bidir` | The state attends within itself, both ways; a question block attends to the state and within itself, both ways |
| `prefix_causal` | As `prefix_bidir`, with the state causal within itself |
| `causal` | Every token attends to itself and the tokens before it |

The Gated DeltaNet layers stay causal and positions run 0..L-1 over the row,
as the checkpoint expects. The state block is the leading tokens of the rendered prompt
that end within the state text: the system turn, the user turn's image and
video placeholders, `State:` and the state. Each question block holds the
rest: the blank line after the state (with any token that merges the state's
last characters into it), the question, its coded options and the generation
prompt. Because the text after the state is always `\n\nQuestion:`, the split
depends on the state alone, and the reference implementation splits its rows the same way. A
request's state block is tokenized and encoded once, its images and videos go
through the tower once as part of it, and every question block branches from
it (`EngineModel.branch`, `BranchBatch`). The native engine packs the states
and blocks of a batch in one row (`models/branch.py`): a state attends within
itself, and a block takes one softmax over its state's keys followed by its
own (one variable-length call for all states, one for all blocks); a Gated
DeltaNet block starts from its state's final recurrent state, its
convolution window seeded with the state's last inputs, and the chunked rule
runs every state and every block over `cu_seqlens`. On gfx942 the fused
layers run on the packed row, the gated-delta preparation on the rows laid out
with each block after its state's last inputs. The branched forward is the
model's reference: the golden check, shared-context batches and batches that
merge requests all run it, and whole requests share forwards up to the
coalesced token budget (`merges_shared_batches`). A lone text question
shares nothing, so it runs as its row under the same mask while the dense mask
stays small; merged batches branch, since their dense masks cost more than the
variable-length kernels save. `run_rows` runs each
question as its own row under the same mask (`ForwardBatch.states`); the two
differ only by rounding ([record](records/state-once.md)). A question costs its
block and its share of the state, so `usage.input_tokens` counts the state once
and each question's block.

**Prompt layout and temperature per request.** Two optional
`decision_config.json` fields, both covered by the identity, change how a
package reads a request. `prompt_layout` `header-v1` puts a text-only
request's question list in front of `State:`: every valid question in request
order, numbered, Choice and Score questions with their option texts
(`prompt.header_block`). The list is part of the state block, so a state-once
request still encodes it once and branches every question from it; requests
with images, videos, assets or Box and track questions keep the d3 layout.
`temperature_by_questions` gives the temperature of a request from its number
of Choice, Noul and Score questions (the questions it renders, or, for a
request with assets or Box and track questions, every question of those
types). Each item carries its request's temperature, so a forward that merges
requests divides each row by its own, one scalar division per temperature as a
request's own forward divides; Box and track readouts keep the package
temperature. A package without the fields answers as before
([record](records/d3-ultra.md)).

## 9. Scheduler and planner

```text
request -> validate (400) -> route to its model (404 / 422) -> admission (429)
        -> plan: render every item (per-item errors) -> that model's queue
        -> its worker: the batcher forms batches -> shortest expected finish first -> readout -> response
bundle  -> plan every task -> submit all at once -> await each -> results in task order
```

Each model has its own scheduler and worker thread. Validation and rendering
happen on the request thread, so malformed requests never take queue
capacity. The batcher plans the rows of every queued job together (section
5.4). A bundle's tasks for one model are queued at once as one job group,
whatever their deadlines (each job keeps its own, measured from when the HTTP
request arrived), so one forward serves every head and consumer that reads
the same input. On a batch-invariant model the batcher runs the queued jobs
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
keyed by content).

The worker runs planned batches in order of their jobs' expected finish: the
time a job was queued plus its tokens at the measured cost per token of their
lane (a running average of that lane's recent forwards). Short work goes first, and a long job still runs
once the work queued before its expected finish is done, so nothing starves.
Between two forwards the worker takes the jobs that arrived meanwhile (on a
model whose batcher coalesces, once they have waited one batching window), so a short request
waits for the forward in flight, not for the remaining windows of a long one.
It plans them together with the jobs of planned batches that have not started,
so requests that wait for the device share forwards up to the budget instead of
keeping the batches their arrival formed; a started batch, a job split over
forwards once one of them ran, and reference forwards keep their plan. Each job
is answered as soon as its own batches have run. The order of
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
available device with enough free memory. The memory estimate is the weights
in the dtype the accelerator holds them in (FP32 as an upper bound, or BF16
plus 10% for a family whose `gpu_weights` / `cpu_weights` name it and, on GPUs,
for the BF16-stored Linear weights of a `bf16_resident` family) plus a 2 GiB
activation reserve, matched to the peaks in
[the Decision 3.0 device memory record](records/decision3-device-memory.md) and
[the BF16-resident one](records/bf16-resident-device-memory.md);
`--memory-budget` caps it per model. A refusal names, per device, what the
model needs and how much is free; on a host with a GPU the GPU's reasons come
first and the CPU's last. A model names the device capabilities it requires per
accelerator (`ModelSpec.requires`; a backbone architecture declares its own
once, in `BackboneSpec.requires`, and the families pass it on), and placement
refuses a device whose accelerator does not report one, before any weights
load. The Qwen3.5
decoders require `lapack` on the CPU, since their gated-delta kernel solves
triangular systems there: a PyTorch built without LAPACK (the ROCm image's)
refuses them with that cause and the fix (a PyTorch with LAPACK, such as the
CPU image, or a GPU device). A package with FP8 weights requires `float8` on
every accelerator, which a GPU reports when it has FP8 matrix units (an NVIDIA
GPU of compute capability 8.9 or later, or an MI300 or MI350 series GPU), so
any other device refuses it before its weights load. A package with NVFP4
weights likewise requires `nvfp4`, which a GPU of compute capability 10.x or
12.x (Blackwell) reports. When every device the
model may use lacks a capability, the failure is final instead of retried.

### 10.2 Readiness and health

Per model: `loading` (resolve, verify, load, identity) → `warming` (golden
answers) → `ready`, or `failed` with a reason. Readiness is gated on golden
answers per surface: every built-in revision carries golden requests with
reference answers per device class (1e-3 on CPU, 0.02 on GPUs; encoder
embeddings compare by cosine). A recorded hardware-qualified key takes
precedence over the device-class reference; otherwise the existing device-class
reference is used. A selected reference that fails is never replaced by another
candidate. The tolerance is selected by device class, not reference key.
Every golden request runs through its
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
| `native` | Qwen3, Qwen3.5 (GDN hybrid), ModernBERT, BERT, unmerged LoRA; Omni's SigLIP, Whisper and CLAP towers; hidden states, gathered rows, shared-context trees, state-once branches | Every accelerator |
| `onnxruntime` (optional: the `onnx` extra, not in the router images) | Package ONNX graphs (heads baked in) and prepared Omni bundles | CPU EP (validated); CUDA EP, ROCm / MIGraphX EPs and the OpenVINO EP when the installed build provides them (unvalidated until recorded) |

| Accelerator | Status | Kernels |
| --- | --- | --- |
| `cpu` | validated | pure-torch references, FP32; on x86, encoder linears through oneDNN's pre-packed FP32 kernel (weights reordered once at load, batch-invariant, within 3.3e-6 of `F.linear`) |
| `rocm` | validated (MI300X, MI325X) | BF16 autocast, FLA gated delta, causal-conv1d, exact-shape HIP graphs, bit-exact fused Triton element-wise kernels (gfx942) |
| `cuda` | validated (H200, A100, Tesla T4; Vela 2.0 0.3B@32 p50 known regression on H200; `docs/records/cuda-*.md`) | the same kernel slots; fused kernels without bit-exact evidence stay off the exact profile (`cuda-fused-kernels.md`); on Ampere and newer with Triton, a consenting family may use them under `max_speed` (Decision 2.0; `cuda-fused-approximate.md`) |
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
The choices are recorded at the published precision, so a model served at
another `--dtype` runs without them and its kernels tune their own: a BF16
configuration can need more shared memory than an FP32 kernel has (d3 at
`float32` on gfx942).
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
code for gfx90a, gfx942, gfx950 and gfx1151 (Instinct MI200, MI300 and
MI350, and Ryzen AI Max; `CAUSAL_CONV1D_ARCHS` builds others), since a decoder fails on a GPU it has no
code for rather than falling back. Triton's compiled kernels and
MIOpen's databases live in the model volume, since the charts run the router
on a read-only root without a home directory.

The oneDNN linear is a kernel variant a family opts into
(`ModelSpec.kernel_variants`): Vela 1.0 task heads use it on their reference
path, since their reference is the legacy path's recorded agreement, not a bit pattern;
Decision 1.0 keeps `F.linear` to stay byte-identical to its bundled runtime.
Tests that compare against Transformers bit for bit run the reference
kernels. A model whose whole CPU forward is batch-invariant (each row the
same alone or in any batch, shown by a test) sets
`LoadedModel.batch_invariant`, and the batcher then merges the requests
queued together without a window, which lifts throughput under load without
changing an answer.

## 12. CLI

Engine mode runs a runtime process in the current Python environment:

```bash
pip install ./src/vllm-sr ./src/model-runtime   # after PyTorch; or the vllm-sr image, or src/model-runtime/Dockerfile
vllm-srun serve vllm-sr/Decision-2.0-Kai-0.6B --device cpu --port 8100
vllm-srun serve vllm-sr/Vela-1.0-Encoder-307M-Domain vllm-sr/Vela-1.0-Encoder-307M-PII --device cpu
vllm-srun serve vllm-sr/Decision-2.0-Lux-9B --device rocm:0 --max-num-seqs 256
vllm-srun serve --models models.yaml --uds /run/vllm-sr/runtime.sock
```

Several `MODEL` arguments share the process options; `MODEL@REVISION` pins a
revision per model; `--models FILE` lists models with their own name,
revision, device, engine arguments, engine and family options (the router writes this
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
        # endpoint: http://runtime:8100   # attach to a runtime the router does not manage
    bindings:
      domain_classifier: {deployment: vela-domain, contract: label_distribution.v1, adapter: modernbert}
```

- `artifact` is a Hub repository (pinned by the built-in table, or with
  `revision`) or an absolute package path. `revision`, `device`, `input` and
  `endpoint` keep their Phase 1 meaning; the engine arguments (`max_num_seqs`,
  `max_num_batched_tokens`, `batch_window_ms`, `dtype`,
  `enable_shared_context`) replace the former `profile`, which the parser
  refuses with a pointer to `vllm-sr config migrate`. `replicas` declares
  independent worker placements for one logical deployment.
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

A request stage bundles calls by runtime process. Each managed deployment or
replica owns a separate worker (section 13.4), so calls to different workers
can run in parallel. Attached runtimes may serve several models in one process;
calls to that shared endpoint can use one bundle. Adding or removing a routing
consumer does not regroup workers.

### 13.4 Lifecycle: managed workers and attached endpoints

- **Managed (default).** Every logical `model_runtime` deployment without an
  `endpoint` owns a worker; an explicit `replicas` list creates one worker per
  placement. Workers remain separate even when they use the same GPU. Their
  identity includes the logical deployment, placement, artifact, runtime command,
  cache directory and CPU thread setting, so unchanged workers survive Router
  mode changes and configuration generations. A new generation leases compatible
  workers; retired workers stop after their last reference drains.

  For `device: auto`, the router asks `vllm-srun devices` once and keeps the
  answer for the manager's lifetime. An `auto` placement resolved to CPU is
  planned as `cpu` and receives the managed CPU thread budget. If detection
  fails, the router logs it and each independent worker resolves its own
  placement at startup. The query is bounded at two minutes and is not repeated
  during later reloads.

  Each managed CPU worker receives `--threads max(1, min(16, cores / 2))` by
  default, using integer division.
  `cores` is the router's `GOMAXPROCS`, normally based on affinity and the
  container's CPU quota. `VLLM_SRUN_CPU_THREADS` overrides this with a positive
  integer capped at the full `cores` budget; invalid or nonpositive values use
  the default. This leaves room for peers in the default thread count but does
  not cap the total threads of several concurrent workers.
  This setting controls threads per worker, never worker count, and is
  independent of the number of models or active consumers. GPU workers and
  attached endpoints do not receive this CPU setting. The historical process
  grouping and thread-share measurements remain in
  [`router-latency-cpu.md`](records/router-latency-cpu.md).

  For each worker the router writes a models file and starts
  `vllm-srun serve --models <file> --uds <path>`. It stops the worker after its
  leases drain or the router exits (SIGTERM, then SIGKILL after a 10 s grace
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
  a shared GPU runtime started with `vllm-srun serve <hf-model> ...`; the
  deployment name or `served_name` selects the model on it. The attached owner
  controls process layout and `--threads`; the router does not start or regroup
  that runtime.
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
keep-alive pools, request bundles, worker leases, per-deployment state
(`starting`, `ready`, `unavailable`), the supervisor and metrics
(`vsr_model_runtime_*`: requests and latency per deployment and surface,
result-cache outcomes, bundle sizes and waits, unknown answers by reason,
readiness and restarts). A single-worker deployment uses a frontend LRU of
classify and decision results (`VLLM_SRUN_RESULT_CACHE` entries, default 4,096,
cleared when readiness changes). Multi-worker pools bypass that cache; each
worker keeps its own item cache and deduplicates items inside a bundle.

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
| `precision: fp16` | removed: the model runs in its published precision (`dtype: float16` asks for half precision on a GPU); `native` and `fp32` are dropped too |
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
| Unit (CPU, tiny fixtures) | `src/model-runtime/tests` | every family's renderer, readout and answers; ModernBERT, Qwen3 and Qwen3.5 backbones against the Transformers reference; heads, windows, spans; engines and fallbacks; scheduler, batching and engine arguments, placement, plugins |
| Registry | same | manifests and file digests, tampered, extra, missing and linked files, identity, offline cache |
| API contract | same | every response validated against `openapi.yaml`; System One cases; bundles; error codes; UDS and TCP; multi-model processes |
| Integration | same | `vllm-srun serve` on generated fixture packages of every family, readiness gating, deadlines, overload |
| Router | `pkg/modelservice`, `pkg/modelruntime/serving`, consumers | generated client against an httptest runtime, bundles, worker leases and replicas with a fake runtime binary, every contract, fail-open, config validation and migration |
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
| Decision 3.0 | the packages' own server (`d3_server.py`), same requests, text, images and videos | CPU, ROCm | Bit-identical answers ([`decision3-parity.md`](records/decision3-parity.md)) |
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
image, device, inputs and every disagreement. CUDA records are
`docs/records/cuda-*.md`; the accelerator `validated` flag is true.

## 18. Performance

Every migrated feature must match or beat the legacy binding's latency (p50,
p95) and throughput on CPU and on ROCm, measured on the same inputs and
hardware, with the records in `docs/records/<workstream>-*`. The techniques:

| Where | Technique | Records |
| --- | --- | --- |
| Decision models | exact-shape HIP / CUDA graphs with host-built masks (and, on CUDA only, of Vela 2.0's forest forward), bit-exact fused Triton kernels (FP32 and BF16 streams on gfx942), lean LoRA, shared-context trees (prefix plus GDN state hand-off), cross-request batching, pinned kernel choices, the 2 GiB guard; Decision 1.0 encoders keep one graph set and one reduced copy per layer stack | `rocm-mi325x-*`, `rocm-mi300x-fused.md` (each fast path on and off), `cuda-vela-forest-graphs.md`, `decision1-performance.md`, `vela2-performance.md`, `decision3-graphs.md` |
| Runtime core | one bundled call per request and process; a bundle's tasks for one model as one job group (one forward for every head reading the same input); per-model content-hash result cache; shortest-expected-finish scheduling that answers each job when its own batches ran and lets short requests run between the windows of a long one (section 9); package files verified in parallel | `router-latency-cpu.md` |
| Encoders | packed (varlen) attention in length groups; banded local attention for long rows (from 1,024 tokens on CPU, 2,048 on GPU); oneDNN pre-packed FP32 linears on the CPU reference path, batch-invariant as probed at load, so concurrent requests merge without a window; dynamic cross-request batching (the default); encoder graphs per shape bucket on GPU, replayed only when padding stays small; a fused gfx942 rotary kernel; reduced copies in merged forwards only where the records show no decision changed (CPU `float32-packed` for Decision 1.0 Kai, Lex and Route and the Vela 2.0 0.3B; BF16 and int8 measured and refused elsewhere); Omni's four towers on the CPU's one OpenMP team, with NumPy's OpenBLAS on one thread; per-hardware kernel registry | `vela1-performance.md`, `embed-performance.md`, `decision1-performance.md`, `vela2-performance.md` |
| Router | parallel signal goroutines with one `/v1/bundle` per request stage and runtime process; a per-deployment result cache; independent deployment/replica workers with a stable CPU thread budget; deadlines and fail-open; pure-Go keyword scoring and model selectors (AVX2 / FMA and NEON dot kernels with a pure-Go fallback) | `router-latency-cpu.md` (historical process layout), `router-latency-rocm.md`, `stores-algorithms.md`, `stores-consumers.md` |
| Router to runtime | HTTP/JSON on a Unix socket, timed per call from the runtime's `Server-Timing`: on CPU the transport is 0.6–2.2% of a call, so there is no binary fast path | `runtime-transport-cpu.md` |

## 19. Phase 1 follow-ups and later work

Done: ROCm golden answers and pinned kernel choices for all six Decision 2.0
models on gfx942, and the graph-cap and batch-shape bucket measurements
(`docs/records/rocm-mi325x-repeatability.md`).

Later: a serving option for the graph cap (1,024 is 1–5% faster under
Index-like traffic); kernel choices for other device classes; more
approximate kernels with their accuracy records; an opt-in deterministic mode
(every row the same answer in any batch); a `vllm` engine (pooling runner plus the family readouts) and a
`llamacpp` engine.

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
| A worker crash interrupts inference | Independent deployment/replica workers isolate failures; a crash restarts that worker and affected consumers follow their error policy |
| Removing the bindings breaks unmigrated configs | The parser points to `vllm-sr config migrate`; the migration guide lists every change |
| Re-embedding after an embedding-model change | Representation identity in every cache and store key; old vectors are never silently reused |
| Image size grows with PyTorch | CPU wheels in CPU images; GPU wheels only in GPU images; the Rust toolchain stages go away |
| Large models outgrow a device | Placement refuses to overcommit; Vega-27B needs one ≥ 64 GB device |
