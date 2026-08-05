# Routing Inference Runtime and Day-0 Model Ecosystem

**Status:** Proposal<br />
**Target:** vLLM Semantic Router H2 roadmap<br />
**Baseline date:** 2026-08-05<br />
**Repository baseline:** `vllm-project/semantic-router@a0d75fd0`<br />
**Related issues:** [#2587](https://github.com/vllm-project/semantic-router/issues/2587), [#2396](https://github.com/vllm-project/semantic-router/issues/2396), [#2395](https://github.com/vllm-project/semantic-router/issues/2395), [#2394](https://github.com/vllm-project/semantic-router/issues/2394), [#2382](https://github.com/vllm-project/semantic-router/issues/2382), [#2360](https://github.com/vllm-project/semantic-router/issues/2360), [#2247](https://github.com/vllm-project/semantic-router/issues/2247), [#2250](https://github.com/vllm-project/semantic-router/issues/2250), [#2252](https://github.com/vllm-project/semantic-router/issues/2252), [#2760](https://github.com/vllm-project/semantic-router/issues/2760)<br />
**Related pull requests:** [semantic-router #2759](https://github.com/vllm-project/semantic-router/pull/2759), [vLLM #42094](https://github.com/vllm-project/vllm/pull/42094)

## Executive summary

vLLM Semantic Router should support the best models for routing, safety, scoring,
retrieval, and policy enforcement regardless of who trained them. Project-owned
models remain useful defaults and research assets, but they should participate in
the same compatibility program as models from Liquid AI, Qwen, IBM, NVIDIA,
KRLabs, Meta, Google, Alibaba, BAAI, OpenAI, Perplexity, and other model teams.

This proposal introduces a **Routing Inference Runtime**: a router-owned control
and compatibility layer over multiple execution drivers. It is deliberately not
a new monolithic tensor engine. vLLM Semantic Router owns the routing-specific
contracts that existing inference engines do not own:

- task and input/output codecs;
- immutable model, adapter, and deployment identity;
- capability and device negotiation;
- model session lifecycle, warmup, hot swap, draining, and rollback;
- signal-to-deployment binding;
- reference conformance, performance receipts, and support levels;
- a repeatable vendor Day-0 onboarding and release pipeline.
- a progressive user experience that makes a verified model easy to discover,
  plan, enable, diagnose, and replace without understanding every backend detail.

Execution remains specialized:

- `sr_native` supplies an in-process, low-overhead path for edge and small
  router-specialized models;
- ONNX Runtime and OpenVINO supply portable and graph-optimized encoder paths;
- vLLM supplies the primary cloud GPU path for decoder, pooling, classification,
  reranking, and compatible LoRA workloads;
- TEI supplies a standardized remote encoder/reranker path;
- an isolated Hugging Face reference driver supplies a safe bring-up path for
  custom-code models before a production driver lands;
- SGLang or another generative server may be added when its protocol or streaming
  behavior is materially better for a model.

The first public milestone targets **36 verified deployment identities at support
level L2**, followed by 21 stretch candidates. A repository name or registry row
is not counted as support. A model counts only when an immutable revision, task
codec, driver, precision, device profile, conformance result, and license/access
status are captured in a reproducible receipt.

## Motivation

### The ecosystem problem

The current router model story is centered on models trained or packaged by the
vLLM Semantic Router project. That made early development fast, but it creates
three long-term constraints:

1. Model vendors cannot integrate a new router-specialized model without adapting
   to signal-specific code and native binding details.
2. Users cannot choose the best model for their language, device, latency budget,
   safety policy, or license without replacing substantial configuration and
   runtime logic.
3. The project risks becoming a model family with a router attached instead of an
   open routing platform.

Recent models show why a broader contract is needed:

- [Liquid LFM2.5 Encoders](https://www.liquid.ai/blog/lfm2-5-encoders) include a
  prompt router that consumes free-form candidate route descriptions and a policy
  linter that scores free-form rules at token level.
- [Qwen3Guard](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B)
  includes both generative and incremental streaming guard forms.
- [KRLabs LettuceDetect v2](https://huggingface.co/KRLabsOrg/lettucedect-v2-qwen-2b)
  returns structured hallucination spans and taxonomy labels from a decoder model,
  while its mmBERT variant is a token classifier.
- [OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter) is a
  long-context token classifier with a custom sparse architecture and structured
  PII decoding.
- [IBM Granitelib Guardian](https://huggingface.co/ibm-granite/granitelib-guardian-r1.0)
  is a signed base-plus-adapter library rather than one standalone checkpoint.

These are all useful router models, but they do not share one architecture, output
shape, artifact layout, license, or execution path.

### User scenarios

The runtime must serve the following scenarios without introducing a new
signal-specific backend enum for each one:

1. An edge gateway runs a 22M to 350M encoder on ARM or x86 CPU with strict memory
   and tail-latency limits.
2. A CPU server runs several sequence and token classifiers through ONNX Runtime
   or OpenVINO.
3. A single AMD MI300X serves a mixture of encoder, reranker, generative guard,
   and base-plus-LoRA deployments.
4. A model vendor supplies an unreleased revision and reference tests so vLLM
   Semantic Router can publish compatibility on release day.
5. An operator chooses a remote TEI or vLLM deployment instead of loading the
   model in the router process.
6. A recipe binds the same deployment to different local policies without sharing
   recipe-owned thresholds or decisions.
7. A new model emits ranked routes, multiple heads, spans, embeddings, pair scores,
   structured JSON, or incremental safety state without changing core routing
   logic.

## Current-state audit

### Registry is not runtime support

`src/semantic-router/pkg/config/registry.go` currently contains 24 static model
rows. Nineteen point to project-owned repositories and five point to external
repositories. Each row records a local path, repository, purpose, approximate
size, context length, and selected attributes.

The current registry inventory is:

| Purpose | Registered repositories or artifact families |
| --- | --- |
| Domain/intent | Project BERT LoRA; mmBERT-32K intent LoRA and merged variants |
| PII | Project BERT LoRA; mmBERT and mmBERT-32K token-classifier LoRA/merged variants |
| Jailbreak | Project ModernBERT; mmBERT-32K LoRA and merged variants |
| Hallucination/fact-check | Project HaluGate sentinel; KRLabs LettuceDetect ModernBERT; tasksource ModernBERT NLI; mmBERT-32K fact-check LoRA/merged variants |
| Feedback | Project feedback detector; mmBERT-32K LoRA and merged variants |
| Modality | Project mmBERT-32K modality router |
| Embedding/similarity | Qwen3 Embedding 0.6B; EmbeddingGemma 300M; MiniLM-L12; project mmBERT 2D-Matryoshka and multimodal embeddings |

This is a useful statement of intended assets, not an execution matrix. A row may
be downloadable but unavailable in the selected build, incomplete for the active
binding, incompatible with its configured head/mapping, or untested on the target
device.

This registry is useful inventory, but it does not prove that a model can execute:

- it does not identify an execution driver or driver version;
- it does not define the exact input/output contract;
- it does not pin a revision or artifact digest;
- it does not capture tokenizer, pooling, head, label, or adapter compatibility;
- it does not report a tested device, precision, numerical tolerance, or license
  decision;
- several rows represent LoRA artifacts while others represent merged models or
  complete checkpoints.

The model downloader compounds this ambiguity. `BuildModelSpecs` currently assigns
the mutable revision `main`; a changed repository can therefore produce different
runtime behavior without a config change.

### Generic classification is still globally constrained

The generic classifier signal supports `type: local` and `type: llm`. The current
local path:

- requires exactly two labels;
- allows one local classifier per routing profile;
- requires all recipes to use an identical process-global model, label list, and
  device;
- initializes a singleton through the Candle binding;
- requires a process restart if the model or labels change.

The external `llm` form calls a chat model with instructions. It is not the same
contract as a sequence-classification endpoint that returns a full probability
distribution. PR [#2759](https://github.com/vllm-project/semantic-router/pull/2759)
is a valuable vertical experiment because it introduces a sequence-classifier
interface, explicit positive labels, and an HTTP classification path for prompt
guard. It should be treated as evidence for the common runtime contract, not copied
into every signal as another backend enum.

Issue [#2587](https://github.com/vllm-project/semantic-router/issues/2587) is the
concrete user-facing failure behind this design. It exposed three independent
concerns that the current surface blends together: the quality of one default
checkpoint, a `risk_score` that represented the winning class confidence rather
than the named risk probability, and hard-coded positive labels that could map a
valid custom model to the wrong policy decision. The issue discussion converged
on a universal sequence/scoring/token classifier mechanism rather than a
jailbreak-only model switch. This proposal preserves that direction while
separating model choice, typed result semantics, quality evidence, and
recipe-local policy.

### Native bindings expose capabilities through different seams

The current native paths have valuable implementations but incompatible ownership
models:

- Candle covers BERT, ModernBERT, mmBERT/mmBERT-32K, Qwen/Gemma embeddings,
  specialized multi-LoRA classification, guards, modality, and multimodal paths.
  Much of its model state remains process-global.
- ONNX Runtime covers named mmBERT sequence/token classifiers, embeddings, 2D
  Matryoshka, and CPU/CUDA/ROCm/MIGraphX provider logic. Some parity methods are
  stubs rather than supported capabilities.
- OpenVINO has a separate ModernBERT classification, token-classification, and
  embedding surface.
- `src/semantic-router/go.onnx.mod` replaces the entire Candle module with the
  ONNX module at build time. This makes package compatibility convenient, but it
  prevents Candle and ONNX from being independently selected drivers in the same
  binary.

Issue [#2396](https://github.com/vllm-project/semantic-router/issues/2396)
correctly identifies this binding-neutral runtime gap. This proposal expands that
scope from native bindings to all local and remote inference drivers.

### Startup orchestration is not a model-session lifecycle

`pkg/modelruntime` provides parallel startup and warmup tasks. That is useful, but
the request path still receives task-specific singleton services rather than
immutable deployment sessions. Current reload behavior can update config and
service pointers independently, and native singleton state cannot always be
replaced without a restart.

The target lifecycle needs two-phase preparation and activation:

1. resolve and validate the complete candidate configuration;
2. materialize and verify artifacts;
3. create new sessions without changing active traffic;
4. probe and warm the new sessions;
5. atomically publish one immutable runtime snapshot;
6. drain in-flight references to the previous snapshot;
7. close old sessions or roll back if activation fails.

### Existing public surfaces duplicate model semantics

Model choices appear in Go config, Python CLI schemas, Dashboard forms, operator
types, model inventory, startup status, downloader metadata, and individual signal
modules. Adding a model family can therefore require edits across unrelated
callers. The new contract must normalize once and project the same capability and
lifecycle state to all consumers.

## Goals and non-goals

### Goals

- Support encoder, decoder, encoder-decoder, pairwise, late-interaction, and
  incremental streaming router workloads.
- Support full checkpoints, merged adapters, base plus adapter, multiple adapters,
  task heads, quantized variants, and remote deployments.
- Select drivers by typed capability and deployment policy rather than vendor name.
- Cover edge CPU through single-node NVIDIA and AMD GPU serving.
- Keep recipes isolated while sharing explicitly global model deployments.
- Make model compatibility reproducible and visible before traffic is served.
- Give vendors a stable intake, conformance, release, and deprecation process.
- Preserve project-owned models without giving them a privileged execution path.
- Make the common path require only a model, task, and device profile while
  keeping every resolved revision, driver, codec, license, and receipt inspectable.

### Non-goals

- Reimplement vLLM, ONNX Runtime, OpenVINO, TEI, or their kernels inside the Go
  router.
- Make one backend support every model or every task.
- Run arbitrary Hugging Face `trust_remote_code` inside the router process.
- Count a Hugging Face repository, registry entry, unverified adapter variant, or
  model card claim as production support.
- Bundle gated, non-commercial, missing-license, or incompatible-license artifacts
  in release images.
- Change default models before reproducible quality, calibration, performance, and
  rollback evidence exists.
- Expose router-internal inference deployments through OpenAI `/v1/models` as if
  they were request-facing generation models.

## Design principles

1. **Model behavior, not vendor, defines the core API.** Vendor-specific loading
   stays at artifact and driver boundaries.
2. **A deployment is immutable.** Mutable aliases resolve to pinned revisions and
   content digests before activation.
3. **Capability is executable evidence.** Unsupported combinations fail before
   serving, with a reason.
4. **Reference and production execution are separate.** A flexible reference
   driver accelerates Day-0 bring-up without becoming the default production path.
5. **Policy remains recipe-local.** Deployments may be global resources, but
   thresholds, positive labels, projections, and decisions belong to the recipe
   that consumes them.
6. **Fallback is explicit.** No driver silently changes task semantics, label
   mappings, pooling, truncation, or precision.
7. **Optimization follows workload shape.** Encoder, decoder, streaming, and
   adapter workloads do not share one scheduler policy.
8. **Upstream first, temporary plugin second.** General architecture support goes
   to the relevant engine upstream; a version-paired companion plugin provides a
   bounded Day-0 bridge.

## Domain model

The public and runtime contracts use the following distinct objects.

| Object | Ownership | Meaning |
| --- | --- | --- |
| `ModelManifest` | Catalog | Immutable semantic and provenance metadata for one model revision |
| `Artifact` | Artifact store | One content-addressed file set: weights, tokenizer, config, head, adapter, or auxiliary mapping |
| `TaskKind` | Runtime | Stable behavior requested by a consumer, independent of architecture |
| `Codec` | Runtime | Versioned typed input and output semantics for a task |
| `Driver` | Runtime | An execution implementation such as `sr_native`, `onnxruntime`, or `vllm` |
| `Deployment` | Global config | A manifest plus artifacts, codec, driver policy, device profile, and limits |
| `Session` | Runtime | A loaded, immutable, concurrency-safe deployment instance |
| `Binding` | Recipe/config | Connects a signal or shared consumer to a deployment and its policy-local interpretation |
| `RuntimeSnapshot` | Runtime | One atomically published set of bindings and sessions |
| `Receipt` | Compatibility pipeline | Signed evidence that a deployment identity passed conformance on a driver/device |

The support identity is:

```text
manifest revision
+ task codec version
+ driver and driver version
+ precision/quantization
+ device profile
+ relevant adapter/head revisions
```

Changing any component creates a new identity that must be checked or explicitly
covered by an existing compatibility rule.

## Proposed architecture

import ZoomableMermaid from '@site/src/components/ZoomableMermaid';

<ZoomableMermaid title="Routing Inference Runtime" defaultZoom={4.5}>
{`flowchart LR
    Config[Canonical config and model manifests] --> Resolver[Deployment resolver]
    Resolver --> Artifacts[Content-addressed artifact store]
    Resolver --> Drivers[Driver registry and capability negotiation]

    Drivers --> Native[sr_native]
    Drivers --> ORT[ONNX Runtime]
    Drivers --> OV[OpenVINO]
    Drivers --> VLLM[vLLM companion driver]
    Drivers --> TEI[TEI driver]
    Drivers --> HF[Isolated HF reference driver]
    Drivers --> SGLang[Optional SGLang driver]

    Native --> Sessions[Typed model sessions]
    ORT --> Sessions
    OV --> Sessions
    VLLM --> Sessions
    TEI --> Sessions
    HF --> Sessions
    SGLang --> Sessions

    Sessions --> Snapshot[Immutable runtime snapshot]
    Snapshot --> Bindings[Recipe-local bindings]
    Bindings --> Signals[Signals, plugins, retrieval, and model selection]

    Receipts[Conformance and performance receipts] --> Resolver
    Vendors[Vendor Day-0 intake] --> Manifests[Reviewed manifests and golden cases]
    Manifests --> Config

    style Resolver fill:#dbeafe
    style Snapshot fill:#dcfce7
    style Receipts fill:#fef3c7
    style Vendors fill:#fce7f3`}
</ZoomableMermaid>

### Control plane and execution plane

The Go router remains the control plane. It resolves config, validates
capabilities, manages lifecycle, publishes snapshots, enforces deadlines, and
normalizes results into signal inputs.

Drivers form the execution plane. An in-process driver calls a native session.
A remote driver calls a supervised local sidecar or operator-managed endpoint.
Both expose the same router-owned task contract and lifecycle state.

The separation lets vLLM Semantic Router call itself a routing inference runtime
without claiming ownership of every tensor kernel.

## Task and codec contracts

`TaskKind` is intentionally smaller than the set of signals. Domain, jailbreak,
PII, hallucination, complexity, feedback, retrieval, and tool routing are
consumers of reusable tasks.

| Task kind | Initial codec | Input | Output | Example consumers |
| --- | --- | --- | --- | --- |
| `sequence_classify` | `label_distribution.v1` | Text or normalized messages | Ordered labels and full probabilities | Domain, NLI, jailbreak, feedback |
| `candidate_route` | `candidate_ranking.v1` | Text plus route names/descriptions | Ranked candidates, scores, abstention | Dynamic route and model selection |
| `multi_head_classify` | `named_heads.v1` | Text | Named categorical and scalar heads | Task plus complexity in one forward |
| `token_classify` | `token_spans.v1` | Text | Typed spans with scores and offsets | PII, hallucination, policy matching |
| `rule_token_score` | `rule_token_matrix.v1` | Text plus free-form rules | Per-rule token scores and spans | Policy linting and configurable guardrails |
| `score` | `scalar_score.v1` | Text or typed fields | Calibrated scalar plus provenance | Complexity and quality estimation |
| `pair_score` | `pair_score.v1` | Query/document or chosen/rejected pair | Scalar or pair preference | Reranking and preference routing |
| `dense_embed` | `dense_embedding.v1` | Text batch | Dense vectors and dimensions | Semantic cache, model/tool selection |
| `contextual_embed` | `document_context_embedding.v1` | Ordered chunks grouped by document | Contextual vectors preserving boundaries | Long-document retrieval and routing |
| `late_interaction_embed` | `late_interaction.v1` | Query or documents | Token vectors and MaxSim-compatible metadata | ColBERT-style shortlist/rerank |
| `generate_structured` | `structured_generation.v1` | Messages plus JSON schema or choice set | Validated typed object and token provenance | Generative guards, router judges |
| `stream_guard` | `incremental_guard.v1` | Session ID plus token/text deltas | Incremental risk state and final result | Streaming output/input safety |

### Common codec rules

Every codec specifies:

- normalization from text and chat messages;
- tokenizer and chat-template ownership;
- maximum input units and truncation side;
- batch ordering and deterministic result correlation;
- label names, aliases, positive-label interpretation, and calibration revision;
- byte, Unicode code-point, and token offset semantics for spans;
- score range, normalization, and abstention behavior;
- timeout, cancellation, partial-result, and invalid-output behavior;
- a schema version and backward-compatibility policy.

Drivers return the complete model result. A driver must not silently collapse an
N-class result to a boolean, convert a rank to an uncalibrated probability, or
invent a score for malformed generative output. Recipe-local projections decide
how the typed result affects routing.

### User-defined label signals

A user-defined label signal is a first-class consumer of these codecs, not a new
backend type. The user declares a signal name, deployment, label taxonomy, and
recipe-local projection. The label set may be binary or multi-class and is not
restricted to built-in concepts such as jailbreak, domain, or hallucination.

An encoder with a sequence-classification head may produce
`label_distribution.v1` directly. A decoder may implement the same semantic
contract through constrained label generation or a logits adapter. Those
implementations are interchangeable only when their receipts cover the same
codec, label mapping, calibration behavior, and failure cases. Architecture names
such as BERT or GPT never appear in the signal decision path.

Custom signal extractors may be supplied through the existing plugin boundary,
but they consume a typed result and declare the codec versions they accept. They
do not receive a raw vendor response or gain ownership of model loading. This
keeps custom signals extensible without recreating inference, lifecycle, and
label-mapping code inside every plugin.

## Runtime modules

### Manifest and artifact resolver

The resolver turns a user declaration into immutable runtime material:

1. Resolve repository aliases to an exact revision SHA.
2. Read model config, tokenizer config, model card metadata, and artifact index.
3. Verify expected files, sizes, digests, architecture, task, license state, and
   gated access.
4. Resolve base, adapter, head, and mapping relationships.
5. Select only drivers whose declared capability can satisfy the deployment.
6. Emit a resolved deployment or an actionable validation error.

Artifacts are content-addressed. The configured repository and revision remain
provenance, while local cache identity uses verified digests. An offline deployment
must resolve entirely from the cache and its receipt.

### Driver registry

The driver registry is keyed by `DriverKind`, not model family or vendor. A driver
reports structured capabilities and limitations:

```go
type Driver interface {
    Kind() DriverKind
    Probe(context.Context, ResolvedDeployment) (CapabilityReport, error)
    Load(context.Context, ResolvedDeployment) (Session, error)
}

type Session interface {
    Identity() DeploymentIdentity
    Capabilities() CapabilitySet
    Warm(context.Context, WarmupPlan) error
    BeginDrain()
    Close(context.Context) error
}

type SequenceClassifier interface {
    Session
    Classify(context.Context, SequenceClassifyRequest) (LabelDistribution, error)
}

type TokenClassifier interface {
    Session
    ClassifyTokens(context.Context, TokenClassifyRequest) (TokenSpans, error)
}
```

Other task interfaces follow the same pattern. Binding construction checks the
specific task interface once; request handlers do not switch on driver or vendor.

### Session manager

The session manager owns:

- concurrent load limits and device memory budgets;
- deduplication of identical resolved deployments;
- request reference counts;
- per-session queues and batching hints;
- health, warmup, and last-error state;
- graceful drain and bounded close;
- optional sleep/offload for remote or GPU drivers;
- restart policy for supervised sidecars.

Process-global native state is migrated behind one adapter at a time. A driver may
temporarily report `max_sessions: 1`, but that limitation becomes discoverable and
cannot masquerade as general multi-model support.

### Immutable runtime snapshots

A snapshot contains the resolved recipe bindings and references to ready sessions.
Reload builds a complete candidate snapshot before one atomic pointer swap. Each
request retains the snapshot for its lifetime. The previous snapshot drains only
after no request holds it.

This avoids mixed states such as a new config pointer with an old classification
service, or a new label mapping over an old model session.

### Binding resolver

Deployments are global shared resources. Bindings remain policy-local. A recipe
binding may declare:

- deployment reference;
- expected task and codec;
- positive labels or taxonomy mapping;
- thresholds and calibration reference;
- fail-open, fail-closed, unknown, or fallback behavior;
- shadow-only mode;
- privacy permission for remote execution.

Two recipes may share the same session while using different thresholds. They
cannot observe each other's decisions, projections, or state.

## Driver strategy

### Capability matrix

`Current` below describes an engine capability available at the proposal baseline;
it does not mean that vLLM Semantic Router already exposes the common driver
contract. `Target` describes work proposed here.

| Capability | `sr_native` | ONNX Runtime | OpenVINO | vLLM | TEI | HF reference | SGLang |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sequence classification | Current, limited families | Current, limited named models | Current ModernBERT | Current upstream | Current `/predict` families | Current | Not primary |
| Token classification | Current, signal-specific | Current, limited | Current ModernBERT | Current upstream | No common token-span API | Current | Not primary |
| Multi-head classification | Specialized only | Export-dependent | Export-dependent | Target plugin/upstream | No | Current | No |
| Dense embedding | Current | Current | Current | Current upstream | Current | Current | Optional |
| Contextual embedding | Target | Export-dependent | No | Target/custom architecture | No | Current | Optional |
| Late interaction | Target | Export-dependent | No | Current/expanding | Limited | Current | Optional |
| Pair reranking | Target | Target | Target | Current upstream | Current `/rerank` | Current | Optional |
| Structured generation | Limited guard path | No | No | Current | No | Current | Current |
| Incremental stream guard | Target | No | No | Target plugin/upstream | No | Reference | Target |
| Base plus LoRA | Specialized current path | Merged artifact | Merged artifact | Current where architecture supports it | Deployment-specific | Current | Current |
| Concurrent multi-LoRA | Specialized and constrained | No | No | Capability-dependent | No | Reference only | Capability-dependent |

### `sr_native`

`sr_native` is the renamed and refactored router-owned native path. Its purpose is
not breadth at any cost. It should provide the lowest-overhead execution for models
that materially benefit routing:

- small BERT, ModernBERT, and mmBERT encoders;
- sequence and token heads;
- dense and Matryoshka embeddings;
- shared-backbone multi-head inference;
- selected base-plus-adapter workloads;
- CPU-first quantization and bounded GPU support when the native library supports
  the device.

The initial adapter wraps Candle compatibility functions. It must replace
role-specific singleton state with a model registry or truthfully advertise the
remaining single-session limits. New architecture work is accepted only with a
router-specific latency, memory, portability, or fused-task advantage over the
external drivers.

### ONNX Runtime

ONNX Runtime is the primary portable graph path for encoder workloads:

- CPU and ARM where supported by the runtime build;
- CUDA for portable NVIDIA execution;
- MIGraphX-first evaluation on AMD, with explicit fallback diagnostics;
- static or bounded dynamic shapes;
- INT8/FP16 graph variants;
- merged adapters and exported task heads.

Issue [#2395](https://github.com/vllm-project/semantic-router/issues/2395)
owns the evidence-based AMD provider order. The runtime must not assume that
MIGraphX owns an entire graph or is always faster. Provider ownership, compile
cache, fallback, logit drift, cold start, and memory are receipt fields.

### OpenVINO

OpenVINO remains the optimized Intel path. It uses the same driver/session contract
and explicitly reports supported model families, device selection, static/dynamic
shape limits, tokenization requirements, and missing features. Configuration that
selects OpenVINO must never silently run Candle.

### vLLM

vLLM is the primary cloud GPU driver. The proposal baseline
`vllm-project/vllm@166f4e2d` already contains broad pooling support, including
BERT, RoBERTa, XLM-R, ModernBERT, Qwen3 embedding/reranking, GTE, BGE-M3, Jina,
Voyage, EmbeddingGemma, OpenAI Privacy Filter, and LFM2 ColBERT architecture paths.
The current [pooling model documentation](https://docs.vllm.ai/en/latest/models/pooling_models/)
is broader than vLLM Semantic Router's native model integration.

The default integration is a supervised or operator-managed vLLM process. The Go
driver calls typed vLLM endpoints for classification, pooling, scoring, reranking,
or generation and verifies response semantics through the selected codec.

General model architecture gaps should be contributed to vLLM. For example,
DeBERTa sequence classification already has an open upstream implementation in
[vLLM #42094](https://github.com/vllm-project/vllm/pull/42094); vLLM Semantic
Router should contribute relevant Prompt Guard and ProtectAI receipts rather than
open a duplicate implementation.

### TEI

[Text Embeddings Inference](https://huggingface.co/docs/text-embeddings-inference/en/supported_models)
is a standardized remote path for supported embedding, prediction, and reranking
models. The driver uses explicit protocol kinds such as `tei_embed`, `tei_predict`,
and `tei_rerank`; it does not label every `POST {"inputs": ...}` endpoint as one
generic Hugging Face protocol.

At the proposal baseline, official TEI hardware documentation covers x86/ARM CPU
and NVIDIA variants, but does not establish AMD support. AMD plans therefore use
vLLM ROCm or ONNX Runtime MIGraphX unless TEI publishes and vLLM Semantic Router
verifies a compatible AMD build.

### Hugging Face reference driver

The reference driver is a separate Python process with:

- no router credentials except the artifact token it needs;
- an allowlisted immutable revision;
- network disabled after artifact acquisition where possible;
- bounded CPU/GPU, memory, request size, concurrency, and timeouts;
- model-specific code isolated from the Go router;
- a version-pinned Transformers environment;
- no L2 production claim until reference parity and deployment policy pass.

It accelerates Day-0 support for custom-code models and generates golden fixtures.
It is not a silent fallback for an unavailable production driver.

### Optional SGLang driver

SGLang is considered only when it supplies a measured advantage for constrained
generation, streaming state, or a model not yet supported by vLLM. It follows the
same sidecar and receipt rules and does not become another unconditional runtime
dependency.

## Why not embed or fork vLLM?

vLLM-Omni is useful prior art. At
[`vllm-project/vllm-omni@78c144f3`](https://github.com/vllm-project/vllm-omni/tree/78c144f3a8f1e4fb3e9d9e0c38bc0a0e635c7c98),
it:

- registers model implementations through the
  [`vllm.general_plugins` entry point](https://github.com/vllm-project/vllm-omni/blob/78c144f3a8f1e4fb3e9d9e0c38bc0a0e635c7c98/pyproject.toml);
- versions releases against the matching vLLM major/minor;
- reuses vLLM EngineCore, schedulers, executors, workers, model runners, and
  registries;
- adds multi-stage orchestration, modality-specific workers, device packages,
  recipes, and CI;
- also subclasses or patches vLLM internal types, as documented in
  [`patch.py`](https://github.com/vllm-project/vllm-omni/blob/78c144f3a8f1e4fb3e9d9e0c38bc0a0e635c7c98/vllm_omni/patch.py).

The reusable lessons are the companion-package model, explicit version pairing,
plugin registration, per-model pipeline, device-specific packaging, and upstream
follow-up discipline. The tight Python internal-API coupling is unsuitable for a
Go request router that must remain stable across several drivers.

The selected design is therefore:

1. normal integrations use a process boundary and supported serving protocols;
2. architecture gaps may use an optional `vllm-sr-runtime` Python companion
   package with `vllm.general_plugins`;
3. the companion package is version-paired to a vLLM minor and tested as a unit;
4. every temporary plugin records an upstream issue/PR and removal condition;
5. vLLM Semantic Router does not fork vLLM or patch its Python internals in the Go
   process.

## Workload-specific optimization

### Encoder path

Encoder workloads perform one bidirectional forward pass and usually return dense
vectors, logits, or token labels. They do not need a KV cache or iterative decode.
The runtime should optimize:

- length-aware dynamic batching and padding reduction;
- separate short, medium, and long-context queues;
- graph compilation and operator fusion;
- CPU INT8 and GPU FP16/BF16 variants with numerical receipts;
- tokenizer parallelism without unbounded goroutines;
- early exit and Matryoshka layer/dimension selection;
- one shared backbone with multiple task heads where tasks share the same input;
- bounded token-span post-processing outside the hot tensor path;
- small-session residency and predictable memory.

Large maximum context should not force every request through a long-context bucket.
Receipts measure realistic distributions and hard long-range cases separately.

### Decoder path

Router decoder workloads differ from open-ended chat. They typically emit a short
label, JSON object, span set, or score. The runtime should optimize:

- high-throughput prefill and short decode;
- continuous batching separated from long answer generation queues;
- exact stop conditions and small `max_tokens`;
- constrained choices or JSON schema decoding;
- prefix caching for shared policy, taxonomy, and route descriptions;
- logits-only classification or first-decision-token termination when the model
  contract permits it;
- calibrated invalid/ambiguous output handling;
- KV-cache sizing based on short outputs rather than chat defaults.

Speculative decoding is not assumed to help. It is enabled only when measured
benefit exceeds coordination cost for the deployment's short output distribution.

### Incremental streaming guard path

Streaming guards add state absent from ordinary classification:

- a typed session key and maximum lifetime;
- token or text delta ordering and deduplication;
- incremental model state/KV reuse;
- monotonic or explicitly reversible risk aggregation;
- checkpoint frequency and bounded retained context;
- cancellation when the upstream stream ends;
- a final result that is comparable to full-context reference inference.

Qwen3Guard Stream is the first conformance target. A stateless endpoint that
reclassifies the full accumulated text is a valid L1 reference implementation but
not an optimized L3 streaming implementation.

## Base, adapter, and multi-head semantics

These mechanisms solve different problems and must not be conflated.

| Form | Memory/package effect | Compute effect | Primary use |
| --- | --- | --- | --- |
| Full or merged checkpoint | Duplicates base per task | One ordinary forward | Simplest portable deployment |
| Load-time adapter merge | Smaller distribution before load | Same as merged after load | One adapter per process/session |
| Runtime base plus LoRA | Shares base weights | Usually separate adapter-influenced forwards | Many task or tenant variants |
| Concurrent multi-LoRA | Shares base and batches adapter work | Backend-dependent | High concurrency across adapters |
| Shared backbone plus heads | Shares base and one backbone forward | Can reduce compute materially | Several signals over identical input |

LoRA inside attention or MLP layers does not automatically allow one backbone
forward to serve several adapters. The base-plus-LoRA roadmap must measure memory,
package size, cold start, throughput, and latency separately. Issue
[#2394](https://github.com/vllm-project/semantic-router/issues/2394) owns this
evaluation for current mmBERT assets.

A resolved adapter deployment records:

- base repository, revision, architecture, tokenizer, and digest;
- adapter repository/revision, PEFT type, target modules, rank, alpha, and digest;
- task head and label/taxonomy artifacts;
- whether the adapter is merged, load-time merged, or dynamically selected;
- driver compatibility and maximum resident adapters;
- calibration and conformance receipt;
- signature information when supplied, as in Granitelib Guardian.

Adapter variants count as separate L2 deployments only after separate receipts.
The project cannot claim dozens of supported models by expanding an adapter matrix
that was never executed.

## Device profiles and backend policy

| Profile | Typical constraint | Preferred drivers | Model guidance |
| --- | --- | --- | --- |
| `edge_arm_cpu` | Low memory, low concurrency, no accelerator | `sr_native`, ONNX Runtime | 20M-350M encoder, quantized, bounded context |
| `edge_x86_cpu` | Tail latency and package size | `sr_native`, ONNX Runtime, OpenVINO | MiniLM/BERT/ModernBERT/mmBERT small/base |
| `server_cpu` | Throughput across several models | ONNX Runtime, OpenVINO, TEI | Batched encoder/reranker, INT8/BF16 where validated |
| `nvidia_single_gpu` | Mixed workloads and VRAM budget | vLLM, TEI, ONNX Runtime, selected native | 100M-8B encoder/decoder, pooling, LoRA |
| `amd_mi300x` | ROCm compatibility and mixed GPU workloads | vLLM ROCm, ONNX Runtime MIGraphX | Decoder/pooling in vLLM; graph encoders in ORT when faster |
| `remote_managed` | Network/privacy/SLO dependency | TEI, vLLM, HF reference for bring-up | Any size with explicit auth, privacy, and timeout policy |

Driver selection is deterministic and validated at startup. `auto` selection may
rank compatible drivers using measured receipts, but the resolved choice and
reason are persisted in the runtime snapshot and diagnostics.

For MI300X, one process does not need to own every model. A common deployment may
run a vLLM pool for decoder and supported pooling models plus an ORT MIGraphX
sidecar for high-throughput encoders. The router runtime owns the common deployment
and fallback contract.

## Proposed configuration contract

The proposal introduces an additive v0.4 model-runtime surface. Exact field names
may change during implementation review, but the separation of manifest,
deployment, and binding is normative.

```yaml
version: v0.4

global:
  model_catalog:
    manifests:
      liquid-prompt-router@35ca4a0:
        source:
          kind: huggingface
          repo_id: LiquidAI/LFM2.5-Encoder-350M-Prompt-Router
          revision: 35ca4a0469f180f1cf05a630df8842fa17ac18e3
        architecture: Lfm2BidirForSequenceRouting
        license:
          id: LFM-1.0
          distribution: reference-only
        artifacts:
          - kind: weights
            format: safetensors
            digest: sha256:<reviewed-digest>
          - kind: tokenizer
            digest: sha256:<reviewed-digest>

    deployments:
      liquid-prompt-router-cpu:
        manifest_ref: liquid-prompt-router@35ca4a0
        task: candidate_route
        codec: candidate_ranking.v1
        driver:
          kind: hf_reference
          version: "transformers:<pinned-version>"
        device_profile: edge_x86_cpu
        precision: fp32
        limits:
          max_input_tokens: 8192
          max_batch_size: 16
          timeout: 100ms
        support:
          required_level: L2
          receipt_ref: liquid-prompt-router-cpu@<receipt-digest>

      granite-factuality:
        manifest_ref: granite-base@<revision>
        adapter_refs:
          - granitelib-factuality-detection@<revision>
        task: generate_structured
        codec: structured_generation.v1
        driver:
          kind: vllm
          endpoint_ref: router-models-mi300x
        device_profile: amd_mi300x
        precision: bf16

routing:
  signals:
    classifiers:
      - name: dynamic-domain
        deployment_ref: liquid-prompt-router-cpu
        candidates:
          - name: code
            description: Programming, debugging, and software engineering
          - name: general
            description: General knowledge and everyday assistance

    pii:
      - name: privacy
        deployment_ref: openai-privacy-filter
        threshold_ref: privacy-filter-calibration-v1
```

### Migration from v0.3

- Current model paths and module-specific settings normalize into implicit
  manifests, deployments, and bindings.
- Existing project-owned defaults behave unchanged during the migration window.
- A generated migration report shows every implicit deployment, mutable revision,
  missing receipt, and backend limitation.
- New v0.4 declarations cannot silently fall back to a legacy singleton path.
- The Python CLI, Dashboard, Operator, Helm, DSL projections, canonical config,
  and OpenAPI surfaces must preserve the same declarations before v0.4 is stable.
- Legacy fields receive a documented deprecation version and removal issue; they
  do not remain a permanent second model system.

## User experience and progressive disclosure

The complete manifest/deployment contract is necessary for reproducibility, but it
must not become the minimum knowledge required to use a model. The product exposes
three layers of detail.

### Level 1: choose a reviewed catalog model

A user who selects a receipt-backed catalog entry supplies only the intended task
and deployment profile. The CLI resolves the reviewed manifest, recommended
driver, immutable revision, and codec:

```bash
vllm-sr model plan openai/privacy-filter \
  --task token_classify \
  --profile amd_mi300x

vllm-sr model enable openai/privacy-filter \
  --name privacy-filter \
  --task token_classify \
  --profile amd_mi300x
```

`model plan` is read-only. Before anything is downloaded or configured it shows:

- resolved repository and revision;
- task and codec;
- selected driver and why it was selected;
- required artifact size and estimated resident memory;
- device/precision and known unsupported alternatives;
- license, gated-access, remote-code, and distribution status;
- current receipt level and the exact identity covered;
- config diff and any credential or endpoint requirements.

`model enable` applies the same plan through canonical config validation and
ETag-protected mutation. It does not maintain a separate CLI-only model registry.

### Level 2: bring an arbitrary Hugging Face model

A user can ask the planner to inspect an unregistered model:

```bash
vllm-sr model inspect LiquidAI/LFM2.5-Encoder-350M-Prompt-Router

vllm-sr model plan LiquidAI/LFM2.5-Encoder-350M-Prompt-Router \
  --task candidate_route \
  --profile edge_x86_cpu \
  --reference-first
```

The planner may return one of four outcomes:

- `ready`: an L2 identity matches exactly;
- `compatible`: a known driver can run it but a local receipt is needed;
- `reference_only`: the isolated reference driver can establish L1;
- `needs_contract`: task, codec, custom code, adapter/head, or license information
  is insufficient.

It never guesses a task solely from a repository name. When metadata is ambiguous,
the diagnostic names the missing decision and provides the smallest manifest
fragment needed to continue.

### Level 3: author an advanced deployment

Experts and vendors can provide the full manifest, adapter graph, driver policy,
limits, fallback, receipt requirement, and binding. The same object is editable in
YAML, validated by the CLI/API, preserved by the Operator, and displayed by the
Dashboard.

### CLI workflow

The proposed CLI surface is:

| Command | Purpose |
| --- | --- |
| `vllm-sr model search` | Browse reviewed catalog entries by task, language, device, license, size, and support level |
| `vllm-sr model inspect` | Resolve metadata and explain candidate compatibility without mutation |
| `vllm-sr model plan` | Select a driver/profile, estimate resources, and render the canonical config diff |
| `vllm-sr model enable` | Apply an approved plan through the management API or local config workflow |
| `vllm-sr model verify` | Run the relevant conformance receipt or compare with an existing receipt |
| `vllm-sr model list` | Show configured/resolved deployments, state, support level, and consumers |
| `vllm-sr model doctor` | Explain download, access, license, driver, provider, memory, codec, and health failures |
| `vllm-sr model disable` | Detach or drain a deployment with reference-safety checks |

All commands support `--json` for automation. Human output leads with the action
and remediation rather than dumping raw backend exceptions.

### Dashboard workflow

The Dashboard uses the same APIs and provides:

1. a catalog browser filtered by task, device, language, parameter size, support
   level, license/access, and local/remote execution;
2. a plan view comparing compatible drivers, resource estimates, receipts, and
   limitations;
3. a binding step that attaches the deployment to a recipe-local signal or shared
   retrieval consumer;
4. visible download, load, warmup, ready, drain, and failure progress;
5. a doctor view with actionable fixes and no secret exposure;
6. an advanced YAML/object view without silently dropping fields the current UI
   does not understand.

### Operator and Helm workflow

Kubernetes users declare the same manifest/deployment/binding objects. Admission
and reconciliation report typed conditions such as `ArtifactsResolved`,
`DriverCompatible`, `ReceiptSatisfied`, `SessionReady`, and `BindingActive`.
Unknown fields are preserved or rejected explicitly; they are never pruned into a
deployment with different semantics.

### UX acceptance criteria

- A reviewed L2 catalog model can be planned and enabled without manually entering
  a revision, driver endpoint, tokenizer, output mapping, or backend-specific flag.
- Planning is non-mutating and explains every default before activation.
- The CLI, Dashboard, API, and Operator render the same resolved identity and
  capability reasons.
- A failed model produces a specific next action such as accepting gated terms,
  providing a token, choosing a compatible profile, adding a codec, pinning an
  adapter head, or running verification.
- Simple workflows generate canonical config; they never create hidden runtime
  state that cannot be reproduced from config and receipts.
- Advanced users can override a recommendation, but the override is validated and
  visible in diagnostics.

## Public API and diagnostics

The existing request-facing `/v1/models` remains a list of routing entrypoints and
explicit backend model IDs according to its public contract. Internal signal model
deployments are exposed through a router-specific inventory API, for example:

```text
GET /api/v1/model-runtime/deployments
GET /api/v1/model-runtime/deployments/{id}
GET /api/v1/model-runtime/receipts/{digest}
```

The inventory reports:

- configured and resolved identity;
- task and codec;
- architecture and artifact form;
- driver, driver version, endpoint, device, precision, and provider selection;
- `configured`, `resolving`, `loading`, `warming`, `ready`, `draining`, `failed`,
  and `closed` states;
- supported and missing capabilities with reasons;
- immutable revision and digests;
- access/license classification without leaking credentials;
- receipt level and last conformance time;
- queue, batching, request, error, timeout, and latency summaries.

Config mutation continues to use whole-document validation, ETag/`If-Match`,
atomic persistence, activation, and rollback semantics.

## Support levels and receipts

### Support levels

| Level | Name | Required evidence | Public wording |
| --- | --- | --- | --- |
| L0 | Candidate | Repository and metadata discovered | Candidate, not supported |
| L1 | Compatible | Reference driver loads; codec smoke and deterministic fixture pass | Experimental compatibility |
| L2 | Verified | Production-intended driver/device passes conformance, failure, and bounded-load receipts | Supported on the named deployment identity |
| L3 | Optimized | L2 plus quality, calibration, latency, throughput, memory, and regression baselines | Optimized on the named profile |

### Receipt schema

Each receipt is machine-readable and content-addressed:

```yaml
receipt_version: v1
deployment_id: openai-privacy-filter-vllm-mi300x-bf16
manifest:
  repo_id: openai/privacy-filter
  revision: 7ffa9a043d54d1be65afb281eddf0ffbe629385b
  artifact_digests: [sha256:<digest>]
task: token_classify
codec: token_spans.v1
driver:
  kind: vllm
  version: 0.26.0
device:
  profile: amd_mi300x
  runtime: rocm
precision: bf16
conformance:
  reference: transformers
  fixtures_passed: 128
  max_logit_error: <measured-value>
  span_exact_match: <measured-value>
performance:
  batch_sizes: [1, 8, 32]
  input_buckets: [128, 512, 4096]
  p50_ms: <measured-value>
  p95_ms: <measured-value>
  throughput_rps: <measured-value>
  peak_memory_bytes: <measured-value>
license:
  id: Apache-2.0
  access: public
generated_at: <timestamp>
toolchain_digests: [sha256:<image-or-lock-digest>]
```

Receipts never store production prompts or private partner fixtures. Public
fixtures are synthetic or vendor-approved; private fixture hashes may prove the
test set used without publishing its contents.

### Required conformance behavior

- numerical parity against the authoritative reference implementation;
- exact label and taxonomy mapping;
- Unicode and span-offset parity;
- deterministic ordering for equal scores;
- maximum-length, empty, malformed, oversized, timeout, cancellation, and
  unavailable-backend cases;
- batch/single equivalence within declared tolerance;
- precision and quantization drift;
- driver restart and hot-swap behavior;
- concurrency and bounded resource use;
- explicit invalid generative output and abstention tests.

## Initial 36-model L2 plan

The following are **targets**, not claims about current vLLM Semantic Router
support. `First driver` is the shortest credible path to L2. Gated and
non-commercial models can receive an L2 receipt for authorized users but are not
bundled or selected as unrestricted defaults.

| # | Model repository or deployment family | Task/codec | First driver | Primary profile | License/access note |
| ---: | --- | --- | --- | --- | --- |
| 1 | [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) | `token_classify` | vLLM | GPU | Apache-2.0 |
| 2 | [meta-llama/Llama-Guard-3-1B](https://huggingface.co/meta-llama/Llama-Guard-3-1B) | `generate_structured` | vLLM | GPU | Llama 3.2, manually gated |
| 3 | [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | `dense_embed` | vLLM | GPU | Apache-2.0 |
| 4 | [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | `pair_score` | vLLM | GPU | Apache-2.0 |
| 5 | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | `dense_embed` | vLLM | CPU/GPU | MIT |
| 6 | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | `pair_score` | vLLM/TEI | CPU/GPU | Apache-2.0 |
| 7 | [Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | `dense_embed` | vLLM/TEI | CPU/GPU | Apache-2.0 |
| 8 | [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) | `dense_embed` | vLLM/TEI | CPU/GPU | Apache-2.0, custom architecture |
| 9 | [Alibaba-NLP/gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) | `pair_score` | vLLM/TEI | CPU/GPU | Apache-2.0 |
| 10 | [Alibaba-NLP/gte-multilingual-reranker-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base) | `pair_score` | vLLM/TEI | CPU/GPU | Apache-2.0, custom architecture |
| 11 | [jinaai/jina-embeddings-v5-text-nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano) | `dense_embed` plus task adapter | vLLM | CPU/GPU | CC-BY-NC-4.0; not a commercial default |
| 12 | [voyageai/voyage-4-nano](https://huggingface.co/voyageai/voyage-4-nano) | `dense_embed` | vLLM | GPU | Apache-2.0 |
| 13 | [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) | `dense_embed` | vLLM/TEI | CPU/GPU | Gemma license, manually gated |
| 14 | [sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) | `dense_embed` | vLLM/TEI | Edge/CPU | Apache-2.0 |
| 15 | [protectai/deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) | `sequence_classify` | TEI/HF reference, then vLLM | CPU/GPU | Apache-2.0 |
| 16 | [meta-llama/Llama-Prompt-Guard-2-22M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M) | `sequence_classify` | TEI/HF reference, then vLLM | Edge/CPU | Meta license, manually gated |
| 17 | [meta-llama/Llama-Prompt-Guard-2-86M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M) | `sequence_classify` | TEI/HF reference, then vLLM | Edge/CPU | Meta license, manually gated |
| 18 | [ibm-granite/granite-guardian-hap-38m](https://huggingface.co/ibm-granite/granite-guardian-hap-38m) | `sequence_classify` | vLLM/TEI | Edge/CPU | Apache-2.0 |
| 19 | [KRLabsOrg/lettucedect-v2-mmbert-base](https://huggingface.co/KRLabsOrg/lettucedect-v2-mmbert-base) | `token_classify` | vLLM/HF reference | CPU/GPU | Apache-2.0 |
| 20 | [patronus-studio/wolf-defender-prompt-injection-small](https://huggingface.co/patronus-studio/wolf-defender-prompt-injection-small) | `sequence_classify` | vLLM/TEI | Edge/CPU | Apache-2.0 |
| 21 | [tasksource/ModernBERT-base-nli](https://huggingface.co/tasksource/ModernBERT-base-nli) | `sequence_classify` | vLLM/TEI | CPU/GPU | Apache-2.0 |
| 22 | [vectara/hallucination_evaluation_model](https://huggingface.co/vectara/hallucination_evaluation_model) | `sequence_classify` | HF reference, then native/vLLM plugin | CPU/GPU | Apache-2.0, custom architecture |
| 23 | [lightonai/modernbert-embed-large](https://huggingface.co/lightonai/modernbert-embed-large) | `dense_embed` | vLLM/TEI | CPU/GPU | Apache-2.0 |
| 24 | [Snowflake/snowflake-arctic-embed-s](https://huggingface.co/Snowflake/snowflake-arctic-embed-s) | `dense_embed` | vLLM/TEI | Edge/CPU | Apache-2.0 |
| 25 | [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) | `pair_score` | vLLM/TEI | Edge/CPU | Apache-2.0 |
| 26 | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | `dense_embed` | vLLM/TEI | Edge/CPU | MIT |
| 27 | [LiquidAI/LFM2.5-Encoder-350M-Prompt-Router](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Prompt-Router) | `candidate_route` | HF reference, then vLLM plugin/upstream | CPU/GPU | LFM-1.0/other |
| 28 | [LiquidAI/LFM2.5-Encoder-350M-Policy-Linter](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Policy-Linter) | `rule_token_score` | HF reference, then vLLM plugin/upstream | CPU/GPU | LFM-1.0/other |
| 29 | [nvidia/prompt-task-and-complexity-classifier](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier) | `multi_head_classify` | HF reference, then native/vLLM plugin | CPU/GPU | NVIDIA Open Model License |
| 30 | [Qwen/Qwen3Guard-Gen-0.6B](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B) | `generate_structured` | vLLM | GPU | Apache-2.0 |
| 31 | [Qwen/Qwen3Guard-Stream-0.6B](https://huggingface.co/Qwen/Qwen3Guard-Stream-0.6B) | `stream_guard` | HF reference, then vLLM/SGLang plugin | GPU | Apache-2.0, custom guard architecture |
| 32 | [KRLabsOrg/lettucedect-v2-qwen-2b](https://huggingface.co/KRLabsOrg/lettucedect-v2-qwen-2b) | `generate_structured` | vLLM | GPU | Apache-2.0 |
| 33 | [perplexity-ai/pplx-embed-context-v1-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-context-v1-0.6b) | `contextual_embed` | HF reference, then vLLM plugin/upstream | GPU | MIT, custom architecture |
| 34 | [katanemo/Arch-Router-1.5B](https://huggingface.co/katanemo/Arch-Router-1.5B) | `generate_structured`/`candidate_route` | vLLM | GPU | Other; review terms before distribution |
| 35 | [LiquidAI/LFM2.5-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M) | `late_interaction_embed` | HF/PyLate, then vLLM | CPU/GPU | LFM-1.0/other |
| 36 | [ibm-granite/granitelib-guardian-r1.0](https://huggingface.co/ibm-granite/granitelib-guardian-r1.0), one pinned base+adapter deployment | `generate_structured` | vLLM | GPU | Apache-2.0; signed adapter family |

Fourteen entries in this table already appear by exact repository name in the
current vLLM source tests or pooling documentation. That makes them integration
fast paths, not automatic L2 receipts: gated access, codec mapping, device, and
failure behavior still require vLLM Semantic Router validation.

## Stretch and partnership backlog

These 21 repositories extend architecture, scale, language, or licensing coverage.
Base-only models become L2 deployments only when paired with a declared task head
or adapter.

| # | Candidate | Why it matters | Initial status |
| ---: | --- | --- | --- |
| 37 | [LiquidAI/LFM2.5-Encoder-230M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M) | Compact bidirectional base for router heads | L1 base; partner head required for L2 |
| 38 | [LiquidAI/LFM2.5-Encoder-350M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M) | Larger LFM2.5 encoder base | L1 base; partner head required for L2 |
| 39 | [Qwen/Qwen3Guard-Gen-4B](https://huggingface.co/Qwen/Qwen3Guard-Gen-4B) | Higher-capacity generative guard | L1 then GPU L2 |
| 40 | [Qwen/Qwen3Guard-Stream-4B](https://huggingface.co/Qwen/Qwen3Guard-Stream-4B) | Higher-capacity incremental guard | Custom streaming driver work |
| 41 | [KRLabsOrg/lettucedect-v2-taxonomy-head](https://huggingface.co/KRLabsOrg/lettucedect-v2-taxonomy-head) | Span-to-taxonomy cascade and head artifact | Adapter/head composition receipt |
| 42 | [fastino/gliguard-LLMGuardrails-300M](https://huggingface.co/fastino/gliguard-LLMGuardrails-300M) | Free-form rule guard through GLiNER2 | L1 isolated custom library |
| 43 | [ibm-granite/granite-guardian-4.1-8b](https://huggingface.co/ibm-granite/granite-guardian-4.1-8b) | Current Granite generative guardian | vLLM GPU candidate |
| 44 | [google/shieldgemma-2b](https://huggingface.co/google/shieldgemma-2b) | Gemma-family safety judge | Gated, vLLM GPU candidate |
| 45 | [allenai/wildguard](https://huggingface.co/allenai/wildguard) | Open guard benchmark/model family | Gated, large decoder |
| 46 | [bespokelabs/Bespoke-MiniCheck-7B](https://huggingface.co/bespokelabs/Bespoke-MiniCheck-7B) | Generative factuality checking | Production hold until license is explicit |
| 47 | [PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct](https://huggingface.co/PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct) | Hallucination/factuality judge | CC-BY-NC-4.0, evaluation only by default |
| 48 | [llm-blender/PairRM](https://huggingface.co/llm-blender/PairRM) | Pairwise preference routing | Custom PairRM codec/architecture |
| 49 | [jinaai/jina-reranker-v3](https://huggingface.co/jinaai/jina-reranker-v3) | Compact current reranker | CC-BY-NC-4.0 |
| 50 | [jinaai/jina-reranker-v2-base-multilingual](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual) | Multilingual XLM-R reranking | CC-BY-NC-4.0 |
| 51 | [mixedbread-ai/mxbai-rerank-base-v2](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v2) | Qwen2-based compact reranker | Apache-2.0, decoder/pooling comparison |
| 52 | [nvidia/llama-nemotron-embed-1b-v2](https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2) | Bidirectional Llama embedding | NVIDIA license, custom architecture |
| 53 | [nvidia/llama-nemotron-rerank-1b-v2](https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2) | Bidirectional Llama reranking | NVIDIA license, custom architecture |
| 54 | [lightonai/GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) | ModernBERT late interaction | Apache-2.0, PyLate reference |
| 55 | [colbert-ir/colbertv2.0](https://huggingface.co/colbert-ir/colbertv2.0) | Classic ColBERT compatibility baseline | MIT, custom ColBERT architecture |
| 56 | [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | Popular Matryoshka embedding baseline | Apache-2.0, custom NomicBERT |
| 57 | [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | Small multilingual edge baseline | MIT |

The list is a dated planning snapshot. The machine-readable pipeline, not this
Markdown table, becomes the live source for revision, status, and receipt data.

## First-wave model conformance specifications

The first implementation wave is chosen to exercise distinct protocols, not just
many checkpoints with the same head.

| Model/family | Required conformance result |
| --- | --- |
| Liquid Prompt Router | Text plus arbitrary route descriptions returns a stable ranked list, full scores, and abstention; candidate order does not change semantics |
| Liquid Policy Linter | Text plus arbitrary rules returns per-rule token scores and spans with byte/code-point alignment |
| NVIDIA task/complexity classifier | Returns all declared task heads and six complexity dimensions plus any aggregate score without flattening them into one label |
| ProtectAI and Meta Prompt Guard | Returns the full named label distribution; configured positive labels are resolved by name, never array position |
| Qwen3Guard Gen | Messages produce a schema-valid safety decision, categories, and confidence/provenance under bounded decoding |
| Qwen3Guard Stream | Ordered deltas produce incremental states and a final result comparable with full-context reference inference |
| IBM HAP | Text returns a calibrated HAP distribution with documented threshold provenance |
| Granitelib Guardian | A pinned base plus signed adapter returns the adapter's declared structured decision; base/adapter mismatch fails before serving |
| LettuceDetect mmBERT | Context, question, and answer normalization returns hallucination spans with exact offsets |
| LettuceDetect Qwen | The same semantic input returns typed JSON spans, taxonomy, and optional explanation; invalid JSON cannot become a confident result |
| LettuceDetect taxonomy head | Detected spans map to category/subcategory with pinned embeddings and deterministic tie handling |
| OpenAI Privacy Filter | BIOES/Viterbi decoding returns typed privacy spans, categories, offsets, and long-context truncation metadata |
| PPLX context embedding | Ordered chunks retain document boundaries and output one contextual vector per declared chunk |
| Liquid ColBERT | Query/doc token vectors and masks reproduce reference MaxSim ranking within tolerance |
| GTE ModernBERT reranker | Query/document pairs return stable scalar scores and deterministic ranking across single and batch execution |
| Arch Router | A bounded route set produces one valid route or abstention through constrained decoding |

## Vendor Day-0 program

### Partner intake

A partner supplies a reviewed intake manifest containing:

- organization, technical owner, and release contact;
- repository or private artifact source and immutable prerelease revision;
- release time and embargo rules;
- architecture and reference library versions;
- task and codec request;
- representative public fixtures plus optional private golden fixtures;
- expected labels, schemas, offsets, and score semantics;
- supported languages, context, precision, device, and batch claims;
- license, acceptable distribution, gated-access, and branding terms;
- reference inference command and known limitations.

The project assigns one integration owner and one reviewer. Private assets remain
outside the public repository and are destroyed or access-revoked according to the
partner agreement after release validation.

### Day-0 track

1. **Triage:** validate task usefulness, license, artifact completeness, and
   overlap with existing models.
2. **Reference bring-up:** reach L1 in the isolated HF driver and freeze golden
   codec fixtures.
3. **Production driver:** use an existing driver, add a temporary companion plugin,
   or coordinate an upstream engine contribution.
4. **Device receipts:** run the agreed CPU/NVIDIA/AMD profiles.
5. **Shadow integration:** bind the model in a non-enforcing recipe and compare
   quality, calibration, latency, and errors.
6. **Release candidate:** pin all revisions, images, lock files, and receipt
   digests under embargo.
7. **Day 0:** publish compatibility metadata, documentation, example config, and
   an optional joint announcement only after L2 passes.
8. **Post-release:** monitor artifact/revision changes and open remediation or
   deprecation issues for failed receipts.

### Partnership priorities

| Priority | Teams | Collaboration target |
| --- | --- | --- |
| P0 | Liquid AI, KRLabs, Qwen, IBM Granite, NVIDIA | Custom router tasks, streaming guards, multi-head, adapters, AMD/NVIDIA receipts |
| P1 | OpenAI, Meta, Google, Alibaba, BAAI, Perplexity | Privacy, prompt safety, embeddings, reranking, contextual embedding |
| P2 | Voyage, Jina, Patronus, Vectara, Mixedbread, LightOn | Compact retrieval, reranking, factuality, and multilingual coverage |

Branding or a “supported by vLLM Semantic Router” badge requires at least one
current L2 receipt and a named owner for regressions. A vendor logo is not a
substitute for compatibility evidence.

## Continuous model pipeline

### Discovery

A scheduled read-only job monitors:

- selected Hugging Face organizations and collections;
- architecture, pipeline, license, gated status, and revision changes;
- new repository tags relevant to routing, guardrails, classification,
  token-classification, embedding, reranking, NLI, and hallucination;
- current vLLM, TEI, Transformers, ONNX Runtime, and OpenVINO support registries;
- open upstream issues and PRs to avoid duplicate integrations.

Discovery creates or updates L0 candidates. It never promotes support by itself.

### Automated stages

```text
discover
  -> metadata and license triage
  -> immutable manifest
  -> reference load and codec fixtures
  -> driver capability probe
  -> numerical conformance
  -> failure and lifecycle conformance
  -> device performance matrix
  -> signed receipt
  -> generated compatibility documentation
  -> periodic regression and deprecation
```

Suggested repository surfaces:

- `config/model/` for reviewed reusable manifest/deployment declarations;
- `tools/model-support/` for discovery, validation, fixture, and receipt tooling;
- `e2e/model-support/` for model-gated conformance scenarios;
- generated compatibility data under the website, never hand-maintained copies;
- large artifacts in Hugging Face or an artifact registry, not Git.

Exact paths must pass repository structure review before implementation.

### CI tiers

| Tier | Trigger | Environment | Purpose |
| --- | --- | --- | --- |
| Model-free | Every PR | CPU | Schema, normalization, driver mocks, lifecycle, generated docs |
| Small public | Relevant PR/nightly | CPU | MiniLM/BERT/ModernBERT smoke and reference parity |
| Driver matrix | Nightly | CPU/NVIDIA/AMD | Selected L2 receipts and performance regressions |
| Full catalog | Weekly/release | Available device fleet | All non-gated accessible L2 identities |
| Partner prerelease | Manual/embargoed | Agreed hardware | Private Day-0 candidate receipts |

Model downloads are cached by immutable revision and digest. Default PR CI never
depends on mutable `main`, live private data, or a gated token.

## Quality and performance evaluation

Model support requires more than numerical parity. Each task has router-oriented
quality gates:

- classification: macro/micro F1, calibration error, abstention, OOD, multilingual
  and adversarial behavior;
- token spans: entity/span exact and partial F1, offset correctness, long-context
  boundary behavior;
- routing: route accuracy, regret/cost, candidate-set scaling, free-form label
  robustness;
- safety: false-negative/false-positive trade-offs by policy category and language;
- embeddings: retrieval/routing quality, neighborhood stability, Matryoshka or
  early-exit degradation;
- reranking: nDCG/MRR and route-quality improvement over bi-encoder-only paths;
- generative structured tasks: schema validity, ambiguity, refusal, category
  accuracy, output-token count;
- streaming: time-to-detection, final/full-context agreement, state memory, and
  false early blocks.

Performance receipts use declared input-length and concurrency distributions and
record P50/P95/P99 latency, throughput, queue time, cold start, warmup, resident
memory, peak memory, artifact size, and energy when available. No universal
latency number is claimed across devices and model scales.

An L3 update fails if it exceeds an approved regression envelope against the same
model, driver, precision, and hardware baseline unless the quality trade-off is
reviewed and recorded.

## Security, privacy, licensing, and supply chain

### Artifact security

- Pin revisions and verify digests before load.
- Prefer safetensors and data-only tokenizer formats.
- Reject unexpected executable files unless the isolated reference or companion
  driver explicitly permits and sandboxes them.
- Record model, tokenizer, adapter, head, mapping, container, driver, and lock-file
  provenance.
- Scan artifacts and runtime images and produce an SBOM where supported.
- Treat changed files at the same repository revision as a supply-chain failure.

### Remote-code policy

`trust_remote_code` is false in the Go process. Custom code runs only in a
version-pinned sidecar with least privilege. Promotion to L2 requires either a
reviewed production plugin, an upstream engine implementation, or an explicitly
approved isolated production deployment.

### Data privacy

- Remote execution requires explicit operator configuration.
- Secrets are resolved at the driver boundary and redacted from config, inventory,
  traces, receipts, and Replay.
- Prompt/body logging is off by default.
- Public receipts contain synthetic or approved fixtures only.
- A recipe cannot send data to a remote driver without its configured privacy
  policy and credential permission.

### License policy

Every manifest classifies access and use:

- `redistributable`: compatible with release or optional image distribution;
- `reference-only`: users fetch the model under its own terms;
- `gated`: an authorized token and terms acceptance are required;
- `non-commercial`: never a commercial/default recommendation;
- `unknown`: blocked from L2 production status until clarified.

Liquid LFM, Meta/Llama, Gemma, NVIDIA, Jina/Patronus non-commercial, and missing
license cases therefore remain visible without being presented as unrestricted
defaults.

## Observability and failure policy

Metrics and traces use deployment/task identities, not raw prompts or high-cardinality
revision strings as unrestricted labels. Required signals include:

- resolved driver and fallback reason;
- session state transitions and duration;
- queue, batch size, input/output units, and truncation;
- request, timeout, cancellation, invalid-output, and driver-restart counts;
- latency by task/deployment with bounded labels;
- receipt age and runtime/receipt mismatch;
- hot-swap activation, drain, rollback, and leaked-session detection;
- accelerator provider ownership and graph fallback where applicable.

Failure behavior is declared per binding:

- `fail_closed`: block or return unavailable for mandatory safety controls;
- `fail_open`: continue with an explicit unverified signal state;
- `unknown`: propagate a typed unknown value for policy handling;
- `fallback`: call a named compatible deployment with a separate receipt.

No backend error becomes an ordinary negative classification.

## Implementation roadmap

The work lands as reviewable vertical slices. Each PR must keep existing default
behavior working unless its migration section explicitly changes it.

### PR 0: proposal and executable terminology

- Land this proposal.
- Define support levels, task/codec vocabulary, and the deployment identity rule.
- Add a machine-readable schema draft and model-free contract tests.
- Cross-link and reconcile existing roadmap issues.

**Exit:** maintainers agree on the control-plane/driver boundary and support claim
policy.

### PR 1: task and codec domain types

- Add typed task kinds and versioned request/result structs.
- Implement label distribution, token spans, scalar/pair score, and dense embedding
  codecs first.
- Add normalization, Unicode offsets, error, timeout, and batch-order tests.

**Exit:** signals can consume typed results without importing a binding package.

### PR 2: manifests, deployments, and capability validation

- Add immutable manifest, artifact, deployment, driver policy, and binding types.
- Resolve revision/digest and reject incompatible capability combinations.
- Normalize current v0.3 model fields into implicit deployments.
- Add inventory DTOs without changing `/v1/models` semantics.
- Add read-only `model inspect` and `model plan --json` flows over the same
  resolver.

**Exit:** a model-free config can be resolved and diagnosed end to end.

### PR 3: sessions and atomic snapshots

- Introduce driver/session interfaces and a registry.
- Add two-phase load/warm/activate, request references, drain, close, and rollback.
- Publish config, services, and model bindings as one immutable snapshot.
- Add concurrent reload and failure-injection tests.

**Exit:** two mock deployments can hot-swap without mixed state or in-flight loss.

### PR 4: native driver adapters

- Wrap Candle, ONNX Runtime, and OpenVINO behind independent drivers.
- Remove build-time module substitution as the long-term selection mechanism.
- Report honest per-driver capability and singleton limits.
- Add ABI, provider, unsupported-capability, and existing-default regression tests.

**Exit:** at least Candle and ONNX can coexist as independently selected drivers
in one supported build or deployment topology.

### PR 5: remote encoder drivers

- Add explicit TEI embed/predict/rerank protocols.
- Add isolated HF reference driver and supervisor.
- Add privacy, credential, body, timeout, restart, and circuit-breaker behavior.
- Port PR #2759's useful sequence-classifier semantics into the common codec.

**Exit:** one external sequence classifier, embedder, and reranker reach L2.

### PR 6: vLLM companion driver

- Implement typed classify/pool/score/rerank/generate client adapters.
- Add version and capability handshake.
- Define the optional version-paired `vllm-sr-runtime` plugin package.
- Add CPU/model-free protocol tests plus NVIDIA/AMD receipts.

**Exit:** the 14 vLLM fast-path candidates can be certified without signal-specific
client code.

### PR 7: advanced codecs

- Add candidate routing, multi-head, rule-token matrix, contextual embedding,
  late-interaction, structured generation, and incremental guard codecs.
- Land first reference receipts for Liquid, NVIDIA, Qwen3Guard Stream, KRLabs,
  and PPLX.

**Exit:** every first-wave protocol has at least L1 reference conformance and an
assigned L2 driver path.

### PR 8: base, adapter, and head deployments

- Implement explicit base/adapter/head resolution.
- Support merged and selected dynamic adapter paths.
- Evaluate current mmBERT and Granitelib deployments.
- Keep unsupported ONNX/OpenVINO adapter forms as explicit merged-only paths.

**Exit:** at least one project-owned and one external base-plus-adapter deployment
reach L2 with rollback.

### PR 9: receipts and generated compatibility site

- Add discovery, manifest validation, conformance, benchmark, receipt, and
  documentation generation tools.
- Add nightly/weekly/device CI tiers.
- Expose receipt-backed model inventory.
- Add catalog search, `model verify`, and `model doctor` using generated receipt
  data and typed runtime diagnostics.

**Exit:** support status is generated from receipts rather than static prose.

### PR 10: 36-model milestone and vendor program

- Certify the initial 36 target identities across declared profiles.
- Publish partner intake documentation and templates.
- Run at least two prerelease/Day-0 pilot integrations with external model teams.
- Document ownership, regression, deprecation, and release processes.
- Complete the catalog browser and plan/enable/bind workflow in CLI and Dashboard.

**Exit:** 36 current L2 identities, multiple vendors, and a repeatable pipeline—not
36 unchecked registry rows.

## Existing work reconciliation

| Existing work | Relationship to this proposal |
| --- | --- |
| [#2587 custom classifier motivation](https://github.com/vllm-project/semantic-router/issues/2587) | Supplies the original production failure and universal-classifier requirement; close only after named-label codecs, custom deployment choice, and quality receipts replace a jailbreak-only model switch |
| [#2396 native runtime contracts](https://github.com/vllm-project/semantic-router/issues/2396) | Becomes the core local-driver workstream; expand its contract to remote drivers, codecs, sessions, snapshots, and receipts |
| [#2395 AMD MIGraphX](https://github.com/vllm-project/semantic-router/issues/2395) | Owns ORT AMD provider evidence and diagnostics under the common driver contract |
| [#2394 base+multi-LoRA](https://github.com/vllm-project/semantic-router/issues/2394) | Owns measured adapter strategy for current mmBERT assets; feeds generic artifact/deployment design |
| [#2382 next-generation model family](https://github.com/vllm-project/semantic-router/issues/2382) | Continues project-owned model research, but project models use the same open support program |
| [#2360 model lifecycle](https://github.com/vllm-project/semantic-router/issues/2360) | Expands into manifest provenance, receipts, release, regression, and deprecation requirements |
| [#2247/#2250/#2252 reranking](https://github.com/vllm-project/semantic-router/issues/2247) | Use `pair_score` codec and deployment lifecycle; avoid an environment-only second model system |
| [#2760 pluggable classifier backends](https://github.com/vllm-project/semantic-router/issues/2760) | Useful vertical consumers of sequence/token/score codecs; backend selection moves to deployments |
| [PR #2759](https://github.com/vllm-project/semantic-router/pull/2759) | Interim proof for full probabilities and external classification; do not replicate its signal-specific enum as the final architecture |
| Closed [#2415 generative hallucination backend](https://github.com/vllm-project/semantic-router/issues/2415) | Migrate the endpoint implementation to `generate_structured` and shared driver/lifecycle contracts |
| [vLLM #42094](https://github.com/vllm-project/vllm/pull/42094) | Existing DeBERTa sequence-classification work; contribute tests instead of duplicating it |

## Alternatives considered

### Continue adding backends inside each signal

Rejected. It produces separate enums, endpoint types, label handling, lifecycle,
and observability for domain, jailbreak, PII, complexity, and hallucination. A new
model that serves several tasks still requires unrelated code changes.

### Make Candle the universal engine

Rejected. Candle remains valuable for `sr_native`, but rebuilding the broad model,
GPU, pooling, generation, and adapter ecosystem already available in vLLM and
other runtimes would slow Day-0 support and increase maintenance.

### Use only vLLM

Rejected. vLLM is the primary GPU driver, but edge CPU, Intel, portable graph,
small in-process models, and reference custom-code bring-up need different
execution paths. Driver diversity is a requirement, not an implementation detail.

### Copy vLLM-Omni's internal integration

Rejected as the default. Its version-paired companion model is useful, but its
internal EngineCore/worker/scheduler coupling would make the Go router track vLLM
internal changes. A bounded optional plugin plus process boundary captures the
benefit with a smaller compatibility surface.

### Count every model card or adapter as support

Rejected. It creates impressive but unverifiable totals. Only L2 receipt-backed
deployment identities count.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| The abstraction becomes too broad before real models use it | Land vertical codecs with first-wave models; do not add empty interfaces |
| Driver results have subtly different semantics | Versioned codecs, reference parity, named labels, and negative tests |
| vLLM internal APIs drift | Prefer serving protocols; version-pair the optional companion; upstream general support |
| Too many resident router models exhaust memory | Device budgets, shared sessions, adapter evaluation, sleep/offload, explicit unavailable state |
| Hot reload leaks or mixes state | Immutable snapshots, request references, drain, bounded close, failure injection |
| Remote execution leaks sensitive input | Explicit privacy policy, isolated credentials, no implicit fallback, redacted observability |
| License terms make a popular model unusable as a default | Separate technical compatibility from distribution/default eligibility |
| “30+” becomes a vanity metric | L2 identity rule, signed receipts, current compatibility page, receipt expiry |
| Partner-specific code accumulates | Vendor-free domain types, temporary plugin exit criteria, upstream contribution |
| AMD provider assumptions are wrong | MIGraphX/ROCm/vLLM measurements on actual MI300X workloads |

## Acceptance criteria

The proposal is implemented when all of the following are true:

1. The router has typed task and codec contracts covering the initial task table.
2. Manifests, artifacts, deployments, drivers, sessions, bindings, snapshots, and
   receipts are distinct domain objects.
3. Unsupported driver/model/task/device combinations fail before traffic with an
   actionable reason.
4. At least Candle, ONNX Runtime, vLLM, TEI, and HF reference execution paths use
   the shared lifecycle and capability surface; OpenVINO remains supported through
   the same contract where built.
5. Config reload prepares, warms, atomically activates, drains, and closes sessions
   without mixed state or lost in-flight requests.
6. Current default models and legacy v0.3 configs have an explicit, tested migration
   path.
7. One project-owned and one external base-plus-adapter deployment reach L2.
8. Encoder, decoder, and streaming paths have distinct measured scheduling and
   optimization policies.
9. CPU edge, CPU server, NVIDIA, and AMD MI300X profiles have real receipts for
   representative models.
10. At least 36 unique deployment identities from multiple model vendors have
    current L2 receipts.
11. Gated, non-commercial, custom-license, and missing-license models are labeled
    and never bundled or recommended contrary to their terms.
12. Compatibility documentation and runtime inventory are generated from receipt
    data.
13. The Day-0 process is exercised with at least two external model teams and has a
    named owner, intake manifest, golden cases, release checklist, and regression
    path.
14. Existing relevant issues are updated or superseded explicitly; duplicate
    upstream vLLM work is not opened.
15. A user can search, inspect, plan, enable, verify, diagnose, and disable a
    reviewed model through canonical CLI/API flows without authoring
    backend-specific configuration.

## Decision summary

- Build a **Routing Inference Runtime**, not a universal tensor engine.
- Keep a focused in-process `sr_native` driver for router-specialized edge and
  fused workloads.
- Make vLLM the first-class cloud GPU driver through a process boundary.
- Use a version-paired vLLM companion plugin only for bounded Day-0 architecture
  gaps.
- Treat ONNX Runtime/MIGraphX and OpenVINO as first-class portable/CPU drivers.
- Separate encoder, decoder, and streaming scheduling and optimization.
- Model base, adapter, head, and merged artifacts explicitly.
- Count only receipt-backed L2 deployment identities toward 30+ support.
- Make vendor partnership an executable conformance and release process, not a
  logo list.

## References

- [Liquid LFM2.5 Encoders](https://www.liquid.ai/blog/lfm2-5-encoders)
- [Liquid LFM2.5 Retrievers](https://www.liquid.ai/blog/lfm2-5-retrievers)
- [vLLM pooling models](https://docs.vllm.ai/en/latest/models/pooling_models/)
- [vLLM LoRA documentation](https://docs.vllm.ai/en/latest/features/lora/)
- [vLLM ROCm installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [Hugging Face Text Embeddings Inference supported models](https://huggingface.co/docs/text-embeddings-inference/en/supported_models)
- [ONNX Runtime MIGraphX Execution Provider](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)
- [vLLM-Omni architecture](https://github.com/vllm-project/vllm-omni/blob/78c144f3a8f1e4fb3e9d9e0c38bc0a0e635c7c98/docs/design/architecture_overview.md)
