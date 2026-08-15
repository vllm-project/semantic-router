# Routing Inference Runtime and Model Portfolio

**Status:** Proposal<br />
**Target:** vLLM Semantic Router model-runtime roadmap

## Summary and decisions

Routing-time inference already includes sequence and token classifiers,
embeddings, rerankers, and small decoders used by signals, algorithms, and
plugins. This RFC defines common model identity, task contracts, deployment,
placement, and support evidence beneath the existing signal → projection →
decision → algorithm → plugin pipeline. Decision semantics and request-backend
routing remain unchanged.

| Area | Decision |
| --- | --- |
| Runtime boundary | The router owns task contracts, bindings, validation, and deployment identity. Candle, ONNX Runtime, vLLM, TEI, and other engines own inference. |
| Extensibility | A new architecture belongs in its serving engine; a new wire API adds a connector; model-specific behavior adds a task adapter; a new semantic result adds a task contract. |
| Binding points | Models bind to recipe signals, decision algorithms, or request/response plugins. Request-facing LLM backends remain provider-owned. |
| Execution path | Use embedded connectors for reviewed low-overhead artifacts; use gateway or direct service targets for decoders, custom code, batching, shared accelerators, multimodality, and dynamic LoRA. |
| Engine ownership | The Go request path does not implement kernels, batching, scheduling, accelerator allocation, or model-server lifecycle. |
| Support evidence | Execution evidence is deployment-specific; quality evidence is binding- and scenario-specific. |
| Model priority | Select Liquid and non-Liquid models by routing value and contract coverage, not family completeness. |
| User data | Prefer configuration-only adaptation. Training runs offline and produces immutable artifacts for evaluation, shadowing, and promotion. |

Terminology is precise throughout this RFC. A **task contract** defines
versioned semantic input, result, validation, and error behavior. A **task
adapter** maps that contract to one runtime operation and owns model-specific
prompting, templates, parsing, pooling, and postprocessing. A **connector**
invokes an embedded ABI or service protocol. An **engine** executes inference.
A **receipt** is signed execution, adapter-conformance, or quality evidence
keyed to the identities defined under Ownership.

## Model priorities

Portfolio priority ranks routing value and contract coverage; it is independent
of delivery phase and support status. Execution and quality use the E/Q levels
defined later.

| Portfolio priority | Capability | Initial models | Serving boundary |
| --- | --- | --- | --- |
| M0 | Current classifiers, spans, embeddings, and NLI | mmBERT-32K system models, current LettuceDetect, ModernBERT NLI | Preserve current Candle/provider behavior |
| P0 | Learned routing and policy | [Liquid Prompt Router](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Prompt-Router), [Policy Linter](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Policy-Linter) | Attached Transformers services with pinned code and dependencies |
| P0 | Small-decoder routing and tools | [Liquid LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M), [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | vLLM service |
| P0 | Dense, late-interaction, and pairwise retrieval | [Liquid Embedding 350M](https://huggingface.co/LiquidAI/LFM2.5-Embedding-350M), [Liquid ColBERT 350M](https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M), [Qwen3 Reranker 0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | Attached task-specific service or vLLM scoring endpoint |
| P0 | Safety, privacy, and grounding | [OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter), [Qwen3Guard Gen 0.6B](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B), LettuceDetect v2 encoder and decoder variants | vLLM or attached Transformers service with model-specific postprocessing |
| P1 | Larger helpers, multimodal input, and streaming guards | [Liquid LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B), [Liquid VL 450M Extract](https://huggingface.co/LiquidAI/LFM2.5-VL-450M-Extract), [Qwen3Guard Stream 0.6B](https://huggingface.co/Qwen/Qwen3Guard-Stream-0.6B) | Added only with task-specific conformance and latency evidence |

Liquid artifacts use LFM Open License v1.0. Commercial entities with at least
USD 10 million in annual revenue require a separate license; these artifacts
are not bundled or enabled as unrestricted defaults.

<details>
<summary><strong>Full model portfolio and deferred candidates</strong></summary>

### M0: current integrations

M0 lists representative current execution families. Registry presence does not
imply a common runnable path or support level.

| Model | Architecture class | Current vLLM-SR placement and value | Migration path |
| --- | --- | --- | --- |
| [mmBERT-32K jailbreak merged](https://huggingface.co/llm-semantic-router/mmbert32k-jailbreak-detector-merged) | mmBERT/ModernBERT sequence classifier | Prompt-risk signal | Preserve current Candle behavior through <code>label_distribution.v1</code> |
| [mmBERT-32K intent merged](https://huggingface.co/llm-semantic-router/mmbert32k-intent-classifier-merged) | mmBERT/ModernBERT sequence classifier | Domain signal and user-taxonomy baseline | Candle embedded; retain label mapping and recipe projections |
| [mmBERT-32K PII merged](https://huggingface.co/llm-semantic-router/mmbert32k-pii-detector-merged) | mmBERT token classifier | PII spans and redaction | Candle embedded; preserve offset semantics through <code>token_spans.v1</code> |
| [mmBERT-32K fact-check merged](https://huggingface.co/llm-semantic-router/mmbert32k-factcheck-classifier-merged) | mmBERT sequence classifier | Fact-check signal | Candle embedded; typed distribution plus calibration metadata |
| [mmBERT-32K feedback merged](https://huggingface.co/llm-semantic-router/mmbert32k-feedback-detector-merged) | mmBERT sequence classifier | Feedback signal and offline routing evidence | Candle embedded; keep feedback policy outside the deployment |
| [mmBERT-32K modality merged](https://huggingface.co/llm-semantic-router/mmbert32k-modality-router-merged) | mmBERT sequence classifier | Modality signal | Candle embedded; retain current request-phase ownership |
| [mmBERT 2D Matryoshka embedding](https://huggingface.co/llm-semantic-router/mmbert-embed-32k-2d-matryoshka) | Bidirectional dense encoder | Routing, cache, model/tool retrieval | Include pooling, normalization, dimensions, and truncation in deployment identity |
| [LettuceDetect current baseline](https://huggingface.co/KRLabsOrg/lettucedect-base-modernbert-en-v1) | ModernBERT token classifier | Hallucination spans | Preserve current detector behavior while normalizing offsets |
| [ModernBERT NLI](https://huggingface.co/tasksource/ModernBERT-base-nli) | ModernBERT sequence-pair classifier | Entailment/factuality signal | Keep as NLI evidence; do not describe it as a native hallucination-span model |
| [Qwen3 Embedding 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Qwen3 pooling/embedding model | Embedding catalog option | Preserve pooling, dimensions, prefixes, and the current native/provider distinction |
| [EmbeddingGemma 300M](https://huggingface.co/google/embeddinggemma-300m) | Gemma dense/Matryoshka embedding model | Gated Matryoshka embedding option | Preserve access and dimension-specific semantics |
| [MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) | Compact sentence-transformer encoder | CPU semantic-similarity baseline and CI fixture | Preserve current pooling and normalization |
| [vLLM-SR multimodal embed small](https://huggingface.co/llm-semantic-router/multi-modal-embed-small) | Text/image/audio embedding model | Current multimodal embedding catalog option | Preserve modality-specific normalization and capability reporting |

### P0: initial capability targets

P0 identifies initial models for new contracts or production integrations.

| Model | Architecture | Router role / result contract | Serving path / model-specific constraint |
| --- | --- | --- | --- |
| [Liquid LFM2.5 Prompt Router](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Prompt-Router) | Bidirectional LFM hybrid encoder + zero-shot routing head | Signal or algorithm helper over user-defined route descriptions; <code>candidate_scores.v1</code>; no native abstention | Attached Transformers service with pinned code and dependencies |
| [Liquid LFM2.5 Policy Linter](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Policy-Linter) | Bidirectional LFM hybrid encoder + rule-token head | Request/response policy signal; runtime-supplied rules × token scores; <code>rule_token_scores.v1</code> | Attached Transformers custom-code service; spans are a downstream projection |
| [Liquid LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M) | Small hybrid causal decoder | Prompt-driven signal, bounded algorithm helper, or tool worker; <code>model_choice.v1</code>/<code>tool_calls.v1</code> | vLLM service, attached or external |
| [Liquid LFM2.5 Embedding 350M](https://huggingface.co/LiquidAI/LFM2.5-Embedding-350M) | Bidirectional LFM dense bi-encoder | Semantic, model, and tool retrieval; <code>dense_embedding.v1</code> | Attached SentenceTransformers service |
| [Liquid LFM2.5 ColBERT 350M](https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M) | Bidirectional LFM late-interaction encoder | Shortlist/rerank with per-token vectors, masks, and MaxSim; <code>late_interaction.v1</code> | Attached PyLate service; llama.cpp reference path |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Small causal decoder | User-fine-tuned structured signal or model choice; <code>model_choice.v1</code> | vLLM service; PEFT and merged forms have separate identities |
| [Arch-Router-1.5B](https://huggingface.co/katanemo/Arch-Router-1.5B) | Qwen2.5 causal decoder routing fine-tune | Bounded model choice; <code>model_choice.v1</code> | vLLM service; commercial use requires separate DigitalOcean permission |
| [OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter) | Bidirectional sparse-MoE token classifier + constrained Viterbi decode | PII/privacy signal and redaction; <code>token_spans.v1</code> | vLLM token classifier plus pinned Viterbi and calibration postprocessor |
| [Qwen3Guard Gen 0.6B](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B) | Qwen3 causal generative guard | Request/response verifier; <code>safety_verdict.v1</code> | vLLM service |
| [LettuceDetect v2 mmBERT](https://huggingface.co/KRLabsOrg/lettucedect-v2-mmbert-base) | Multilingual mmBERT token classifier | Grounding/hallucination spans; <code>token_spans.v1</code> | Transformers reference; native/ONNX Runtime deferred |
| [LettuceDetect v2 Qwen 2B](https://huggingface.co/KRLabsOrg/lettucedect-v2-qwen-2b) | Qwen3.5 hybrid causal decoder fine-tune | Generated grounding spans with category, subcategory, and character offsets; <code>token_spans.v1</code> | vLLM service with strict JSON and offset recovery |
| [Qwen3 Reranker 0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | Causal yes/no relevance scorer | Model, tool, and document rerank; <code>pair_score.v1</code> | vLLM scoring endpoint with <code>Qwen3ForSequenceClassification</code> override, <code>classifier_from_token: ["no", "yes"]</code>, and pinned reranker template |
| [NVIDIA prompt task and complexity classifier](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier) | DeBERTa-v3 custom multi-head classifier | Fixed 11-task taxonomy, six complexity dimensions, and overall score; <code>named_heads.v1</code> | Attached service using pinned NeMo Curator/custom-PyTorch code; NVIDIA Open Model License review |
| [GTE ModernBERT reranker](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) | ModernBERT cross encoder | Query/document scoring; <code>pair_score.v1</code> | vLLM scoring endpoint; TEI alternate |

### P1: follow-on targets

| Model | Architecture class | vLLM-SR use | Serving path and gate |
| --- | --- | --- | --- |
| [Liquid PII Detector](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-PII-Detector) | Bidirectional LFM token classifier | Multilingual PII spans | Attached Transformers service with pinned code and dependencies |
| [Liquid LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) | Hybrid causal decoder with agentic post-training | Algorithm helper, tool worker, extraction, and verifier | vLLM service; always-on reasoning requires latency qualification |
| [Liquid VL 450M Extract](https://huggingface.co/LiquidAI/LFM2.5-VL-450M-Extract) | SigLIP2 vision encoder + LFM decoder | Image/document structured signal; <code>multimodal_structured.v1</code> | vLLM service; Transformers reference |
| [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | Small multimodal hybrid causal model | Small-decoder, image, and tool conformance | vLLM service |
| [FunctionGemma 270M](https://huggingface.co/google/functiongemma-270m-it) | Tiny Gemma causal tool model | Bounded tool/action selector; <code>tool_calls.v1</code> | vLLM service; gated Gemma terms |
| [Qwen3Guard Stream 0.6B](https://huggingface.co/Qwen/Qwen3Guard-Stream-0.6B) | Qwen3 custom incremental guard head | Streaming input/output safety; <code>incremental_guard.v1</code> | Attached Transformers service or pinned <code>support_qwen3_guard</code> SGLang branch; tokenizer/state conformance required |
| [Nemotron 3.5 Content Safety](https://huggingface.co/nvidia/Nemotron-3.5-Content-Safety) | Gemma-3 4B multimodal causal safety fine-tune | Text/image and bring-your-own-policy verifier | vLLM service; SGLang/Transformers reference; custom license and GPU qualification |
| [Llama Guard 4 12B](https://huggingface.co/meta-llama/Llama-Guard-4-12B) | Llama-4 multimodal causal guard | Text/multi-image safety | vLLM; gated Llama terms |
| [gpt-oss-safeguard-20b](https://huggingface.co/openai/gpt-oss-safeguard-20b) | Sparse-MoE causal safety-reasoning model | Bring-your-own-policy evaluation | vLLM or SGLang service; harmony-format and latency qualification |
| [Granite Guardian 4.1 8B](https://huggingface.co/ibm-granite/granite-guardian-4.1-8b) | Granite causal judge | Safety, hallucination, and custom criteria | vLLM service |
| [Granitelib Guardian r1.0](https://huggingface.co/ibm-granite/granitelib-guardian-r1.0) | Signed LoRA library over multiple Granite bases | Base+adapter compatibility seam | Pin one exact base+adapter deployment; the repository itself is not one model identity |
| [Lion Warden](https://huggingface.co/patronus-studio/lion-warden-ai-security-classifier) | Multilingual mmBERT/ModernBERT encoder with seven heads | Agent security, injection, tool risk, and routing signals | Attached Transformers or ONNX Runtime service; <code>named_heads.v1</code> |
| [ProtectAI DeBERTa prompt-injection v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) | DeBERTa-v3 sequence classifier | Prompt-injection distribution | Candle FFI with recipe/catalog integration pending; Transformers or ONNX Runtime fallback |
| [GliGuard 300M](https://huggingface.co/fastino/gliguard-LLMGuardrails-300M) | Schema-conditioned GLiNER-style encoder | Schema-constrained guard labels; no free-text rule evaluation | Attached Transformers custom-code service |
| [Vectara HHEM](https://huggingface.co/vectara/hallucination_evaluation_model) | Custom T5-style consistency classifier | Factuality scalar/verdict | Attached Transformers custom-code service |
| [BGE-M3](https://huggingface.co/BAAI/bge-m3) | Multilingual XLM-R multi-function encoder | Dense, sparse, and multi-vector retrieval | vLLM with BGE-M3 overrides for all three tasks; TEI dense only |
| [BGE reranker v2 M3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | Multilingual XLM-R cross encoder | Pair reranking | vLLM service; TEI/Transformers alternate |
| [PPLX context embedding 0.6B](https://huggingface.co/perplexity-ai/pplx-embed-context-v1-0.6b) | Contextual document embedding model | Document-aware chunk vectors; <code>contextual_embedding.v1</code> | Attached Transformers service with pinned code and dependencies |
| [GTE ModernColBERT](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) | ModernBERT late-interaction encoder | Non-Liquid ColBERT comparison | vLLM token-embedding/scoring endpoint with <code>ColBERTModernBertModel</code> override; PyLate reference |
| [Qwen3-VL Embedding 2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) / [Reranker 2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) | Multimodal embedding and reranking pair | Image/text retrieval and rank | vLLM with reranker architecture override; Transformers reference |
| [Voyage 4 Nano](https://huggingface.co/voyageai/voyage-4-nano) | Qwen-derived dense encoder | Multilingual dense embedding comparison | vLLM with <code>VoyageQwen3BidirectionalEmbedModel</code> override and <code>trust_remote_code</code>, or TEI; pin pooling, query/document prompts, and dimensions |

### P2: deferred or reference-only

P2 entries are evaluation targets, not production commitments.

| Models | Architecture class and possible vLLM-SR use | Disposition |
| --- | --- | --- |
| [Liquid LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct), [LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B), [Encoder 230M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M), and [Encoder 350M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M) | Additional decoder scales and base bidirectional encoders for helper/classifier fine-tuning | The selected 350M and 2.6B models cover these contracts; add a size only for a measured quality, latency, or fine-tuning requirement |
| [Liquid LFM2.5 Audio 1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B), [LFM2.5 encoder diffusion](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Diffusion), and [LFM2.5 Spellchecker](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Spellchecker) | Audio generation, iterative masked generation, and token correction | Require new input/state contracts or lack a defined routing use case |
| [Qwen3Guard Gen 4B](https://huggingface.co/Qwen/Qwen3Guard-Gen-4B), [Qwen3Guard Gen 8B](https://huggingface.co/Qwen/Qwen3Guard-Gen-8B), [Qwen3 Embedding 4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B), [Qwen3 Embedding 8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B), [Qwen3 Reranker 4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B), and [Qwen3 Reranker 8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B) | Larger causal guards, dense encoders, and causal rerankers using the same contracts as their small anchors | Add scale receipts only when measured quality offsets memory and latency; size alone is not a new capability |
| [Jina embeddings v5 small](https://huggingface.co/jinaai/jina-embeddings-v5-text-small), [v5 nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano), and [Jina reranker v3.5](https://huggingface.co/jinaai/jina-reranker-v3.5) | Qwen-derived dense encoders and a causal listwise reranker | Evaluation-only because all three use CC-BY-NC-4.0; listwise ranking is outside core v1 |
| [xLAM-2-1B FC](https://huggingface.co/Salesforce/xLAM-2-1b-fc-r) | Qwen2.5 causal function-calling model; <code>tool_calls</code> conformance fixture | Research/non-commercial release; benchmark-only until license and production quality gates are satisfied |
| [Bespoke MiniCheck 7B](https://huggingface.co/bespokelabs/Bespoke-MiniCheck-7B), [Patronus Lynx 8B](https://huggingface.co/PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct), and [PairRM](https://huggingface.co/llm-blender/PairRM) | Causal/pairwise factuality and preference verifiers | Older, larger, overlapping, or non-commercial candidates; retain only as quality benchmarks or partner-requested work |
| [Llama Guard 3 8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B), [ShieldGemma 2 4B](https://huggingface.co/google/shieldgemma-2-4b-it), and [WildGuard](https://huggingface.co/allenai/wildguard) | Generative text/multimodal safety classifiers | Superseded or overlapping coverage, plus access and cost constraints; benchmark-only |
| [BGE small English v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), [Arctic Embed small](https://huggingface.co/Snowflake/snowflake-arctic-embed-s), and [ColBERT v2](https://huggingface.co/colbert-ir/colbertv2.0) | Lightweight dense and classic late-interaction retrieval encoders | CPU/CI fixtures; selected retrieval models already cover the contracts |

Portfolio admission requires a defined use case, measured quality or cost value,
a serving path, and an owner. Compatible custom deployments remain attachable
without portfolio status.

</details>

## Background and alignment with upstream main

The design is based on upstream main at the revision recorded in
Traceability. Current code establishes these constraints:

| Current-main fact | Design consequence |
| --- | --- |
| A request-facing virtual model resolves to one recipe before signal evaluation; a concrete provider model can pass through | The runtime does not invent another routing namespace or run recipe models on concrete-model passthrough |
| Recipe classifiers evaluate only referenced signals, then groups, composers, output policies, and projections feed the decision engine | Typed inference is a substrate for existing signals, not a replacement decision engine |
| A matched decision owns its candidate models, algorithm, and plugins | An algorithm helper may select only inside that bounded candidate set |
| <code>PromptSelector</code>, generic/preference classifiers, Router-R1, and AutoMix already call helper/verifier models over HTTP | Small decoders and verifiers are existing first-class routing workloads |
| The PromptSelector call path combines strict JSON, an allowlist, no recursive selection, an ExtProc-owned deadline, and outer deterministic fallback | The common decoder contract should preserve the complete orchestration behavior, not attribute it to one class |
| PR #2759 is merged and provides a jailbreak-specific <code>/classify</code> path with full-label validation | Generalize the existing seam instead of creating a parallel classifier client |
| Embedded model startup is coordinated in <code>pkg/modelruntime</code>, but native model state is still largely global/singleton | Embedded model/device changes are <code>restart_required</code> in v1; do not promise universal hot swap |
| Config reload builds a candidate router and swaps only the router pointer; candidate construction can mutate globals and the old router closes immediately | Phase 1 must publish one immutable generation and drain it by lease before closing owned resources |
| The Go router mutates headers/body and emits the selected model; Envoy/Gateway performs endpoint routing and load balancing | Request-facing backend traffic and router-helper inference remain separate data planes |
| CPU, AMD, and NVIDIA images are distinct build-time binding variants | Current capability is not a dynamically pluggable in-process connector registry |
| Training scripts cover classifier LoRA, embeddings, Qwen3 generative LoRA, and classical selectors | They are useful artifact producers, but not yet a versioned training/promotion platform |

Current gaps are:

- model repositories are commonly resolved from mutable <code>main</code> and are
  not identified by complete digests;
- task-specific modules duplicate loading, protocol, labels, timeouts, and error
  semantics;
- upstream engine architecture support is often mistaken for an implemented
  router connector/task adapter or a verified model deployment;
- <code>use_cpu</code> is too coarse to express placement and can move small router
  models onto GPUs reserved for request backends;
- remote code, model adapters, tokenizer/chat templates, pooling, calibration, and
  offset decoding are not consistently included in artifact, deployment, and
  binding identities;
- ordinary, streaming, and Looper paths do not yet share one uniform
  request/response plugin phase contract.

Safety bindings require more than binary sequence labels. PII localization,
policy linting, streaming guards, and generative guards can return label
distributions, token spans, rule-token scores, incremental state, or structured
verdicts. Recipe policy projects that evidence into a gate, redaction, or
escalation.

## Goals and boundaries

### Goals

- define common identity, task, execution, and evidence contracts across encoder,
  decoder, embedding, reranking, span, streaming, and multimodal workloads;
- reuse immutable deployments across recipe-local bindings without sharing
  policy;
- preserve decision-layer policy and allow only bounded learned or prompt-driven
  selection;
- isolate custom code, make placement explicit, and support user-trained
  artifacts through evaluation, shadowing, promotion, and rollback;
- migrate current models and recipes without behavior regressions.

### Non-goals

- implement inference kernels, batching, quantization, parallelism, accelerator
  allocation, model-server supervision, or request-backend proxying in Go;
- execute arbitrary Hugging Face <code>trust_remote_code</code> or user Python in
  the router process;
- define a universal raw-tensor/custom-gRPC protocol or universal dynamic LoRA,
  hot-swap, and hardware support;
- train on live requests or include private examples in receipts;
- treat an engine architecture list, model card, or smoke test as production
  support.

## Overall architecture

import ZoomableMermaid from '@site/src/components/ZoomableMermaid';

<ZoomableMermaid title="Routing pipeline and inference boundaries" defaultZoom={4.2}>
{`flowchart LR
    Client["Client"] --> Gateway["Envoy / Gateway"]
    Gateway --> Entry["Resolve recipe or passthrough"]
    Entry --> Signals["Referenced signal graph"]
    Signals --> Map["Mapping, groups, composers, output policy, projections"]
    Map --> Decision["Boolean decision"]
    Decision --> Selector["Bounded algorithm selector"]
    Selector --> ReqPlugins["Decision request plugins"]
    ReqPlugins --> Mutation["Provider, header, and body mutation"]
    Mutation --> Gateway
    Gateway --> Backend["Selected request backend"]
    Backend --> Gateway
    Gateway -. response phase .-> ResPlugins["Router response plugins / verifiers"]
    ResPlugins -. response mutation .-> Gateway
    Gateway --> Client

    Signals -. typed call .-> Runtime["Routing Inference Runtime"]
    Selector -. helper / verifier .-> Runtime
    ReqPlugins -. typed call .-> Runtime
    ResPlugins -. typed call .-> Runtime

    Runtime --> Embedded["Embedded connector"]
    Runtime --> GatewayModel["Gateway model"]
    Runtime --> Direct["Direct endpoint"]

    style Runtime fill:#dbeafe
    style Decision fill:#dcfce7
    style Gateway fill:#fef3c7`}
</ZoomableMermaid>

### Data planes

1. **Request-facing traffic:** client → Envoy/Gateway → selected provider/backend.
   The router returns ExtProc mutations and selection metadata; Envoy/Gateway owns
   the upstream endpoint, load balancing, and token stream.
2. **Router-helper inference:** a bounded internal call to an embedded connector
   or typed model endpoint. The router owns the deadline, task contract, result
   validation, and recipe interpretation.

The Phase 1 Looper compatibility path may send helper requests through Envoy
with a concrete model and internal headers. Such requests must bypass recipe
resolution and recursive selection. In this path,
<code>provider_model_ref</code> names a logical provider model and Envoy retains
backend load balancing and retry ownership. A <code>direct_endpoint</code> target
instead resolves one endpoint and leaves retry/circuit-breaker policy with the
router connector. Endpoint objects own network location, authentication,
privacy, identity/readiness paths, and per-attempt reliability; wire protocol
and semantic behavior belong to the task contract and task adapter. The
capability selects a runtime operation and protocol; the connector implements
that protocol.

A gateway-model target is production-admissible only when every eligible
backend is homogeneous for engine build, artifact, precision, protocol, and
capabilities. Its identity includes a canonical digest of the provider-model
configuration and resolved backend set, or an equivalent control-plane
revision. One probe cannot identify an opaque or heterogeneous pool; that pool
remains on the legacy helper path and cannot become a v1 runtime Deployment.
Changing the eligible backend set invalidates the receipt.

Request-backend fallback remains defined by the
[model execution fallback proposal](/docs/proposals/model-execution-fallback); helper-model
fallback is binding policy.

### Ownership model

Artifact architecture, runtime operation, wire protocol, task semantics,
binding location, and placement are separate configuration axes. Configuration
exposes four top-level objects and named capabilities within each deployment:

| Object | Ownership | Meaning |
| --- | --- | --- |
| <code>ModelManifest</code> | Global catalog | Immutable source, architecture, tokenizer/template, license, base/head/adapter lineage, and artifact digests |
| <code>Deployment</code> | Global catalog | One manifest + target + connector + expected engine/build + placement + named runtime capabilities and limits |
| <code>Capability</code> | Nested in a deployment | One runtime operation and protocol, with modalities, invocation shape, features, and limits reported by the prepared handle or endpoint |
| <code>Binding</code> | Actual recipe owner | Connects one owner/phase to <code>deployment.capability</code> through a task contract and task adapter; owns prompt, taxonomy, threshold, projection, deadline, fallback, privacy, and state |
| <code>Receipt</code> | Conformance pipeline | Signed execution evidence for a deployment capability, adapter conformance for a resolved binding, or quality evidence for a binding and named evaluation scenario |

Deployments are reusable. Labels of interest, thresholds, failure policy, and
projections remain recipe-local. One manifest may back multiple deployments;
one deployment may expose several capabilities from the same loaded model. For
example, one vLLM chat capability can support separate <code>model_choice</code>,
<code>tool_calls</code>, and <code>safety_verdict</code> bindings without loading
the model three times.

| Identity | Required components |
| --- | --- |
| Artifact | Source revision and digests for weights, tokenizer/template, base, adapter, head, and auxiliary files |
| Deployment | Artifact identity + connector ID/version + target identity + expected engine/build + precision/placement + declared capability descriptors |
| Binding | Deployment-capability identity + owner/location/phase + task-contract ID + task-adapter ID/version/config + resolved candidates/prompt/taxonomy/projection/failure/privacy + required contexts |

Artifact identity is the SHA-256 digest of an RFC 8785 canonical JSON lock
manifest. Its file entries are sorted by UTF-8 path and record path, byte length,
SHA-256 digest, role, and base/adapter/head lineage. This makes the identity
independent of local download layout.

Support combines deployment-scoped execution evidence with binding- and
scenario-scoped quality evidence.

Resolved Deployment and Binding descriptors use the same canonical JSON rules
as Artifact locks: references are replaced by subject digests before hashing.
Aliases, receipt URIs, admission thresholds, and binding state are excluded, so
promotion does not create a circular or unrelated identity change. Cross-language
golden vectors fix the canonicalization behavior.

The task contract defines semantic input, result, and error behavior. The task
adapter composes and validates a runtime request without changing recipe policy.

Router, graph, bindings, and clients share the generation lifetime defined under
Resolution and generation activation.

### Binding locations

Bindings remain with their existing owner:

| Location | Minimum binding fields | Required constraint |
| --- | --- | --- |
| Recipe signal | <code>capability_ref</code>, task contract/adapter, <code>required_contexts</code>, output mapping, failure policy | Typed result is mapped through existing signal output policy and projections before decision |
| Matched decision algorithm | <code>capability_ref</code>, task contract/adapter, candidate source, <code>required_contexts</code>, deadline, failure policy | Selection stays inside declared candidates; verification returns evidence/escalation; no recursive selection |
| Named decision plugin + <code>request</code>/<code>response</code> phase | <code>capability_ref</code>, task contract/adapter, phase, <code>required_contexts</code>, deadline, privacy and failure policy | Runtime executes inference but never reorders plugins |
| <code>providers.models[]</code> | No internal binding | Request backend remains outside the deployment/binding abstraction |

Every binding declares <code>state: active|shadow|disabled</code>; v0.4 has no
implicit default. Shadow inference may emit evaluation telemetry but cannot
change signal results, decisions, selected models, plugin mutations, or
fallback. Disabled bindings still undergo strict schema, reference, and identity
validation, but skip materialization, probe, and admission.

Signal bindings also declare an output mapping:

| Task contract | Required mapping |
| --- | --- |
| <code>label_distribution.v1</code> | Native label ID → recipe signal name; all declared labels remain present |
| <code>token_spans.v1</code> | Native/BIO label → span type, normalization and offset unit, merge policy, downstream projection/redaction ref |
| <code>candidate_scores.v1</code> | Candidate ID → recipe signal name; preserve every score, ranking, and declared abstention |
| <code>model_choice.v1</code> | Exact candidate ID → recipe signal event; invalid or undeclared IDs remain typed errors |

The diagram defines execution order. Ordinary, streaming, and Looper paths do
not currently run identical response phases; bindings must declare supported
contexts until a common phase contract exists. Runtime connectors do not change
cache/verifier ordering.

## Typed task and result contracts

Runtime consumers depend on versioned semantic results, not architecture names.

Core v1 defines <code>label_distribution.v1</code>,
<code>token_spans.v1</code>, <code>candidate_scores.v1</code>, and
<code>model_choice.v1</code>. Other contracts are added with their first
consumer.

| Core v1 contract | Required input | Required result |
| --- | --- | --- |
| <code>label_distribution.v1</code> | Request ID + normalized text/messages + declared label namespace | Every declared label and score/logit, calibration/truncation provenance |
| <code>token_spans.v1</code> | Request ID + original UTF-8 text + normalization policy | Typed spans with byte/code-point/token offsets and scores |
| <code>candidate_scores.v1</code> | Request ID + normalized conversation + ordered <code>&#123;id, description&#125;</code> candidates | One score per candidate and a deterministic ranking; abstention only when model-native or binding-defined |
| <code>model_choice.v1</code> | The same candidate envelope plus an exact-choice schema | One candidate ID or declared abstain, plus a bounded optional rationale |

Signal candidates come from the declared route/label set; algorithm candidates
come from the matched decision's <code>modelRefs</code>. Stable IDs are
authoritative. Responses include request correlation, deployment identity,
contract version, truncation/provenance, and a typed error or unknown state.

Extension candidates:

| Task contract | Semantic result | Consumer |
| --- | --- | --- |
| <code>structured_signal.v1</code> / <code>named_heads.v1</code> / <code>rule_token_scores.v1</code> | Schema-valid fields, named heads, or token × rule scores without invented probability/spans | Custom signal, task profile, policy linter |
| <code>pair_score.v1</code> / <code>listwise_ranking.v1</code> | Pairwise scalar or listwise order with declared scoring semantics | Model, tool, and document reranking |
| Dense, sparse, contextual, and late-interaction embedding contracts | Vectors, dimensions, masks, normalization, input role, and document boundaries required by the method | Cache, retrieval, model/tool selection |
| <code>safety_verdict.v1</code> / <code>verification_verdict.v1</code> | Typed decision, categories/evidence, unknown state, provenance | Input/output safety and grounding |
| <code>tool_calls.v1</code> | Validated tool names and arguments against an allowlist/schema | Tool/action plugin |
| <code>incremental_guard.v1</code> / <code>multimodal_structured.v1</code> | Ordered risk state, or typed image/text input and schema-valid output | Streaming safety or modality/document signal |

Each contract defines semantic normalization, input/result limits, batch
correlation, score and offset semantics, errors, and version compatibility.
Tokenizer/template behavior and model-specific truncation live in the task
adapter. The binding supplies one end-to-end deadline.

Task adapters must not:

- turn the winning class confidence into a named risk probability;
- convert an exact decoder choice into a calibrated distribution;
- turn malformed JSON into a confident negative;
- silently discard classes, heads, spans, or abstention;
- call a token classifier a sequence classifier because both return logits.

Recipe policy converts typed evidence into a gate, route score, redaction, or
escalation. Safety policy remains in the decision/plugin layer.

## Runtime, transport, and lifecycle

Task semantics, runtime operation, connector, target, lifecycle owner, engine,
and hardware placement are configured independently. Architecture names are
metadata and admission inputs; the router never dispatches by architecture.

### Invocation target and lifecycle

| Target kind | Router data plane | Required reference | Typical lifecycle owner | Reload behavior |
| --- | --- | --- | --- | --- |
| <code>embedded</code> | Typed Go/CGO call through a reviewed connector | None | <code>router</code> | Current native state is process-global; artifact/device change reports <code>restart_required</code> |
| <code>gateway_model</code> | Existing Looper/internal request through Envoy | One <code>provider_model_ref</code> | <code>orchestrator</code> or <code>external</code> | Gateway/control-plane attestation covers the complete backend set; Envoy owns load balancing and retries |
| <code>direct_endpoint</code> | Versioned HTTP/JSON over loopback, bridge DNS, ClusterIP, or HTTPS | One <code>endpoint_ref</code> | <code>orchestrator</code> or <code>external</code> | Prepare and probe a new client, atomically switch the generation, then drain the old client |

An attached service is a deployment topology, not an API mode. Kubernetes
should normally use a separately scalable Deployment/Service instead of one GPU
sidecar per router replica. Local Docker can attach a service to the existing
bridge. Request-path Go never supervises service processes.

### Engine and connector coverage

| Engine | Router connector | Current implementation | Proposed behavior |
| --- | --- | --- | --- |
| Candle | <code>sr.candle.embedded.v1</code> | CPU product path; several process-global model states | Expose reviewed capabilities; retain restart-required semantics until handles exist |
| ONNX Runtime | <code>sr.onnx.embedded.v1</code> | Separate AMD/NVIDIA build-tagged binaries; available tasks differ by build | Report the effective provider and require precision/provider receipts |
| OpenVINO | <code>sr.openvino.embedded.v1</code> | Source path exists, but canonical packaging is incomplete | Experimental until a packaged fail-closed binary, tests, and receipts ship |
| vLLM | OpenAI or task-specific HTTP connector | Existing chat and embedding clients; no unified vLLM task client | Reuse stable endpoints and add only the task adapters needed by a binding |
| TEI | Documented task-specific HTTP connector | No explicit generic TEI connector in upstream main | Claim only exact endpoint/payload versions that pass conformance |
| Isolated Transformers | Narrow typed HTTP connector | Reference path for custom-code models | Version-pinned, least-privilege, bounded, and never an implicit production fallback |
| SGLang | Task-specific HTTP connector | Optional generative/streaming engine | Add only for a measured model or protocol requirement |

### Extension API

The vLLM model registry is useful precedent: it maps architecture declarations
to lazily inspected implementations, derives supported tasks from behavioral
interfaces, separates model loading from executor/platform selection, and only
publishes API routes supported by the loaded runner. Semantic Router applies
the same separation at its own boundary; architecture loading remains inside
the serving engine.

Three immutable startup registries define the router extension surface:

| Registry | Descriptor | Factory output |
| --- | --- | --- |
| <code>TaskContractRegistry</code> | Namespaced ID/major version, request/result schema, semantic validation, compatibility rule, fixtures | Typed contract validator |
| <code>TaskAdapterRegistry</code> | Fully qualified versioned ID, one contract major + runtime operation + protocol, required features, <code>unary</code>/<code>session</code> shape, strict config schema | Generation-scoped semantic mapper/parser |
| <code>ConnectorRegistry</code> | Fully qualified versioned ID, target kinds, protocols, static features/limits, strict config schema | Generation-scoped embedded handle or service client |

Registration completes before config resolution and then freezes. Duplicate or
unknown IDs fail closed. Descriptor lookup is side-effect-free: it does not
load a model, initialize an accelerator, connect to a service, or execute user
code. Factories are lazy and run only while preparing a candidate generation.
Core contract IDs are reserved; third-party IDs include an owner namespace.
The major version is part of each registry key rather than a second version
field. Core fields are strictly decoded; only namespaced <code>extension_config</code>
may contain component-owned fields, which are validated against the registered
schema.

There is no dynamic Go plugin ABI or arbitrary import path. In-process
extensions are reviewed, statically linked, namespaced, and allowlisted.
Out-of-process services can add an architecture without a router binary change
when they expose an existing protocol and capability. Selection is explicit;
the resolver never tries connectors based on model name or engine heuristics.

| Change | Extension point | Unchanged router policy |
| --- | --- | --- |
| New model architecture | Engine registry/interface plus execution receipt for an existing capability | Task contracts, connectors, and bindings |
| New wire API or engine integration | Connector factory and protocol conformance | Semantic result and consumer ownership |
| New model prompt/parser/postprocessor | Versioned task adapter and fixtures | Deployment target and routing decision layer |
| New semantic result | Task contract, task adapter, typed consumer integration, and fixtures | Engine architecture registry |
| New artifact/load format | Artifact resolver and immutable lineage | Request-path invocation API |
| New accelerator/executor | Engine/platform integration and execution receipt | Binding semantics |

### Transport and capability negotiation

Initial service protocols are
<code>openai.embeddings.v1</code>,
<code>openai.chat-completions.v1</code>, and an explicit
<code>sr.sequence-classify.http.v1</code> generalized from the current
<code>/classify</code> client. Token classification, pair scoring,
late-interaction, and session-based streaming receive separate versioned
protocols where an existing endpoint cannot preserve their semantics. A generic
<code>POST &#123;"inputs": ...&#125;</code> endpoint is not called TEI-compatible without
an exact method/path/payload receipt. UDS may carry the same HTTP DTO; ExtProc
gRPC remains Envoy-to-router traffic.

| Runtime operation | Initial protocol examples | Task contracts that may bind to it |
| --- | --- | --- |
| <code>chat.generate</code> | OpenAI chat completions | Model choice, structured signal, tool calls, safety/verification verdict |
| <code>embedding.dense</code> | OpenAI embeddings, embedded ABI | Dense/contextual embedding |
| <code>classify.sequence</code> | SR classify HTTP, embedded ABI | Label distribution, named heads, pair score |
| <code>classify.token</code> | Versioned token-classify HTTP, embedded ABI | Token spans, rule-token scores |
| <code>embedding.token</code> / <code>score.pair</code> | Versioned task endpoint | Late interaction and reranking |
| <code>guard.session</code> | Versioned stateful endpoint | Incremental guard |

Endpoint objects contain network location, secret reference, privacy class,
identity/readiness paths, and per-attempt reliability only. A connector defines
method/path, wire DTO, cancellation, transport errors, and retry safety. A
capability selects the operation/protocol and reports modalities, features, and
limits. Semantic errors remain in the task contract. A Binding owns one
end-to-end deadline. Attempts, backoff, and retries fit within the remaining
budget; generative or otherwise non-idempotent operations do not retry unless
the connector descriptor explicitly marks the operation retry-safe.
Binding privacy is a requirement; endpoint privacy is a deployment guarantee.
Activation rejects a binding whose requirement exceeds that guarantee.

Activation resolves capabilities field by field across config, connector
descriptor, live report, and receipt:

| Field | Resolution rule |
| --- | --- |
| Artifact/deployment identity, operation, protocol major, precision, and placement | Exact match |
| Modalities and features | Set intersection; every binding requirement must remain present |
| Request, response, batch, context, and session limits | Minimum approved limit; each binding request must fit |
| Cancellation, retry safety, and streaming/session support | Enabled only when every relevant source permits it |

The live
report includes deployment/artifact digest, connector/API version, engine build
identity, operations/protocols, modalities, invocation shape, limits, effective
provider, and precision. An endpoint that cannot prove live identity remains
E0/E1 unless its lifecycle owner supplies equivalent attestation.

### Resolution and generation activation

Activation uses the existing startup-DAG mechanics, not a request-time generic
executor:

1. Strictly decode v0.4, resolve artifact locks, endpoints, deployments, and
   owner-local bindings, and compute canonical subject digests.
2. Resolve the exact contract, task adapter, and connector descriptors; enforce
   license/access policy and reject executable code in embedded mode.
3. Prepare candidate handles/clients without mutating live globals, then perform
   bounded identity/readiness and capability probes.
4. Verify signed offline execution, adapter-conformance, and quality receipts.
   Activation does not run full conformance or quality evaluation.
5. Compile immutable <code>ResolvedBinding</code> objects so request handling has
   no registry lookup, model-name switch, or untyped configuration decoding.
6. Publish one generation containing the router, resolved graph, registries,
   selectors, and owned handles/clients. On failure, close candidate resources
   in reverse order without changing live state.

The lifecycle is
<code>Resolved → Prepared → Ready → Active → Draining → Closed</code>. Reloads
are serialized and coalesced latest-wins under a bounded activation context.
Each ExtProc stream and management operation acquires the active generation;
the old generation closes its owned resources only after all leases release or
the drain deadline expires. Embedded artifact, label, or device changes remain
restart-required; service deployments use blue/green rollout. Native handle hot
swap is post-v1.

## Hardware contract

Execution status is keyed by artifact, deployment capability, connector,
engine/provider, precision, and device. Training status is separate. Placement
is a deployment requirement and observed receipt field; it does not alter task
semantics.

| Placement | Observed build/runtime path on upstream main | Target policy |
| --- | --- | --- |
| x86 CPU | CPU image with in-process Candle; alternate ONNX builds exist | Default for small router models when latency fits; receipt per model/precision/context |
| ARM CPU | CPU image cross-builds Candle for arm64 | Receipt per exact model; image availability does not imply numerical/performance parity |
| AMD GPU | x86_64 ONNX-tagged ROCm image; external vLLM ROCm is a separate service | Record the effective ROCm/MIGraphX/CPU provider and report or reject fallback according to policy |
| NVIDIA GPU | x86_64 ONNX CUDA image; external vLLM is separately served | Explicit placement and receipt; engine/orchestrator owns scheduling/batching |
| Intel/OpenVINO | Implementation source exists but canonical packaging is incomplete | Experimental until fail-closed artifacts and tests exist |
| Kubernetes | Router and inference backends are separate services/inference pools | Placement uses endpoint/service refs and orchestrator resources; router does not allocate GPUs |

Placement declares accelerator, optional device IDs, precision, limits, and
fallback. Request-backend accelerator selection must not rewrite router-model
placement; CPU remains the default without a GPU receipt.

## User-data adaptation and fine-tuning

User adaptation has three layers:

| Layer | Examples | Promoted object |
| --- | --- | --- |
| Configuration-only adaptation | Candidate descriptions, KB prototypes/exemplars, free-form policy rules, thresholds, calibration | Recipe-local config/binding |
| Offline trained artifact | Classifier/token head, embedding/ColBERT adapter, small-decoder LoRA, merged checkpoint | Immutable manifest/artifact candidate; it becomes a deployment only after serving resolution |
| Online routing-policy state | Routing-sampling/Beta weights and shadow/active algorithm policy | Versioned algorithm state, not a Hugging Face model |

v1 imports artifacts rather than scheduling training. Artifact bytes remain in
a customer-approved model registry or object store. Content-addressed manifests
and receipts are referenced from version-controlled config/catalog state. An
offline pipeline produces the artifacts and evidence; the router validates them
during activation.

Promotion creates a new recipe/catalog generation that changes the binding from
shadow to active. The proposed generation publisher commits router and runtime
state together, and rollback restores the retained generation. CLI, Dashboard,
or operator automation may
prepare that config change, but the request path is not the artifact or receipt
system of record. Managed training and dataset upload are separate work.

The offline flow is:

~~~text
approved private dataset reference
  -> schema, privacy, license, dedup, and leakage validation
  -> deterministic train/validation/test split
  -> isolated training job
  -> adapter / head / full / merged / selector artifact
  -> held-out quality and calibration receipt
  -> reference-contract conformance
  -> production connector/runtime and hardware receipt
  -> recipe-local shadow binding
  -> explicit atomic promotion
  -> rollback to the retained deployment
~~~

### Training artifact families

| Task family | Input schema | Typical trainer/artifact | Serving constraint |
| --- | --- | --- | --- |
| Sequence classification/routing | Text/messages + named label or candidate choice | Encoder head/LoRA or small-decoder PEFT | Label taxonomy and prompt/template revision are identity |
| Token spans | Text + typed character/token spans | Token head/LoRA | Offset conversion and long-context behavior must pass conformance |
| Embedding/reranking | Positive pairs, triplets, hard negatives, or ranked lists | SentenceTransformers/PEFT/full snapshot | Pooling, normalization, dimensions, and input prefixes are identity |
| Small-decoder structured choice/tool | Messages + allowed choices/schema + expected object | PEFT/TRL LoRA or merged checkpoint | LoRA and merged forms get separate receipts |
| Classical model selection | Query/model outcomes, cost, latency, category | KNN/KMeans/SVM/MLP algorithm artifact | Kept in selection/policy ownership; identity includes embedding deployment, pooling/normalization/dimension, taxonomy/order, feature transform, and candidate set |
| Base + model adapter/head | Approved base plus LoRA or task head | Model adapter or merged snapshot | Base/tokenizer/adapter compatibility is validated before serving |

Training scripts are artifact producers, not the system of record. Record:

- private dataset URI/ACL, schema/version, split hashes, taxonomy, and license;
- base repository commit/digest and tokenizer/chat-template revision;
- training code, image/lock digest, seed, hyperparameters, adapter targets/rank,
  and head definition;
- artifact digests and merged-versus-adapter parity where applicable;
- held-out and slice metrics, calibration/threshold evidence, and failure cases.

Receipts contain hashes and approved aggregates, not private prompts or PII.
Router Replay export requires opt-in, redaction, authorization, and tenant
isolation; current Replay capture is not a training dataset. Request processing
never updates weights in place.

Training hardware is verified per recipe: CPU for classical and selected encoder
jobs, CUDA for common PEFT/Transformers workflows, ROCm per exact recipe, and MPS
as experimental. Training success does not imply serving support.

## Proposed configuration shape

This is a proposed v0.4 shape. Runtime fields are accepted only after exact
version dispatch into a strict v0.4 schema. A v0.3 CLI/operator capability gate
may prevent distribution to older binaries, but an older parser must never be
relied on to enforce v0.4 admission: current v0.3 behavior warns on many unknown
fields and could otherwise ignore a safety-critical setting.

Existing fields map as follows:

| Current field/owner | Target object | Migration behavior |
| --- | --- | --- |
| <code>global.model_catalog.system/modules</code> | Reviewed deployment refs | Existing stable refs resolve through generated manifests; no recipe behavior change |
| <code>global.model_catalog.external</code> and embedding provider config | Endpoint + deployment capability | Existing location, secret, and reliability fields map to an endpoint; protocol and task behavior move to connector/capability definitions |
| Recipe classifier/model settings | Signal-owned binding | Labels, thresholds, and output policy stay recipe-local |
| <code>routing.decisions[].algorithm.prompt.model</code> | Algorithm-owned helper binding | Existing concrete helper remains valid; the binding adds immutable capability, contract, and adapter evidence |
| Decision plugin model settings | Plugin-owned request/response binding | Plugin order and phase stay in the existing owner |
| <code>providers.models[]</code> | No migration | Request-facing backends remain provider-owned |

Bindings reference named deployment capabilities plus binding-scoped adapter
and quality evidence; deployments reference execution evidence. Receipt
contents remain outside recipe config.

~~~yaml
version: v0.4

providers:
  models:
    - name: qwen-router-0.6b
      provider_model_id: Qwen/Qwen3-0.6B
      api_format: openai
      backend_refs:
        - name: router-helper
          endpoint: 127.0.0.1:8010
          protocol: http
          type: chat
          api_key_env: ROUTER_HELPER_API_KEY
    - name: fast-model
      provider_model_id: fast-model
      api_format: openai
      backend_refs:
        - endpoint: 127.0.0.1:8000
          protocol: http
    - name: strong-model
      provider_model_id: strong-model
      api_format: openai
      backend_refs:
        - endpoint: 127.0.0.1:8001
          protocol: http

global:
  model_catalog:
    runtime:
      receipt_verification:
        trust_bundle_ref: org-production-receipts
        admission_policies:
          production:
            mode: enforce
            minimum_execution: E2
            minimum_quality: Q2
      endpoints:  # Direct service objects; not providers.models backend pools.
        liquid-router-local:
          base_url: http://liquid-router:8080
          identity_path: /v1/model-info
          health_path: /health
          secret_ref: liquid-router-token
          privacy: cluster_only
          reliability:
            per_attempt_timeout_ms: 1000
            max_retries: 0
            circuit_breaker_errors: 5
      manifests:
        qwen_router_0_6b:
          source:
            type: huggingface
            repo: Qwen/Qwen3-0.6B
            revision: "<commit-sha>"
          artifact_lock: >-
            oci://registry.example/sr-locks/qwen-router@sha256:<lock-digest>
        mmbert_intent:
          source:
            type: huggingface
            repo: llm-semantic-router/mmbert32k-intent-classifier-merged
            revision: "<commit-sha>"
          artifact_lock: >-
            oci://registry.example/sr-locks/mmbert-intent@sha256:<lock-digest>
      deployments:
        qwen_router_small:
          manifest_ref: qwen_router_0_6b
          expected_engine:
            name: vllm
            build_ref: oci://registry.example/vllm@sha256:<image-digest>
          lifecycle_owner: orchestrator
          connector:
            id: sr.openai-http.v1
          target:
            kind: gateway_model
            provider_model_ref: qwen-router-0.6b
          capabilities:
            chat:
              operation: chat.generate
              protocol: openai.chat-completions.v1
              invocation: unary
              modalities: [text]
              features: [structured_output, tool_calls]
              limits:
                max_request_bytes: 65536
                max_response_bytes: 16384
                max_batch_size: 1
          placement:
            accelerator: amd
            precision: bfloat16
          execution_receipt_refs:
            - oci://registry.example/sr-receipts/qwen-router-rocm@sha256:<receipt-digest>
        mmbert_intent_cpu:
          manifest_ref: mmbert_intent
          expected_engine:
            name: candle
            build_ref: oci://registry.example/vllm-sr-cpu@sha256:<image-digest>
          lifecycle_owner: router
          connector:
            id: sr.candle.embedded.v1
          target:
            kind: embedded
          capabilities:
            sequence:
              operation: classify.sequence
              protocol: sr.embedded.sequence-classify.v1
              invocation: unary
              modalities: [text]
              limits:
                max_input_tokens: 32768
                max_batch_size: 1
          placement:
            accelerator: cpu
            precision: float32
          execution_receipt_refs:
            - oci://registry.example/sr-receipts/mmbert-intent-cpu@sha256:<receipt-digest>

routing:
  modelCards:
    - name: qwen-router-0.6b
    - name: fast-model
    - name: strong-model
  signals:
    classifiers:
      - name: request-domain
        type: runtime
        state: active
        capability_ref: mmbert_intent_cpu.sequence
        contract: label_distribution.v1
        task_adapter:
          id: sr.direct-label-distribution.v1
        required_contexts: [ordinary, streaming, looper]
        deadline_ms: 50
        labels: [business, law, technology]
        output_mapping:
          type: label_distribution
          native_to_signal:
            business: business
            law: law
            technology: technology
          preserve_all_declared_labels: true
        on_error: unknown
        admission:
          policy_ref: production
          conformance_receipt_refs:
            - oci://registry.example/sr-receipts/mmbert-label-adapter@sha256:<receipt-digest>
          quality_receipt_refs:
            - oci://registry.example/sr-receipts/request-domain@sha256:<receipt-digest>
  decisions:
    - name: balanced
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: fast-model
        - model: strong-model
      algorithm:
        type: prompt
        on_error: fallback
        prompt:
          model: qwen-router-0.6b  # Current compatibility field.
          state: active
          capability_ref: qwen_router_small.chat
          contract: model_choice.v1
          task_adapter:
            id: sr.prompt-choice.v1
            extension_config:
              output_schema: exact_candidate
          candidate_source: decision.modelRefs
          required_contexts: [ordinary, streaming, looper]
          instructions: >-
            Choose fast-model for ordinary requests and strong-model for hard
            reasoning, coding, or multi-step analysis.
          deadline_ms: 2000
          admission:
            policy_ref: production
            conformance_receipt_refs:
              - oci://registry.example/sr-receipts/qwen-choice-adapter@sha256:<receipt-digest>
            quality_receipt_refs:
              - oci://registry.example/sr-receipts/balanced-selector@sha256:<receipt-digest>
~~~

Plugin bindings use the same capability, contract, task-adapter, context,
deadline, and admission fields inside the owning plugin configuration and
additionally declare
<code>phase: request</code> or <code>phase: response</code>. There is no
recipe-wide untyped binding map. Every receipt ref is a content-addressed URI;
activation fetches its signed envelope, checks digest, issuer, validity, and
revocation against <code>receipt_verification</code>, validates its predicate and
subject, then derives E/Q admission from policy rather than trusting a status
claimed by the receipt.

A PII token-span binding uses the same signal shape with:

~~~yaml
output_mapping:
  type: token_spans
  label_map_ref: pii-taxonomy-v1
  normalization: original_utf8
  offset_unit: utf8_byte
  merge_policy: bio
  projection_ref: pii-redaction
~~~

Config validation fails before activation when:

- the binary does not support v0.4 or encounters an unknown runtime/binding
  field;
- an artifact lock is absent, non-canonical, or does not match every resolved
  file;
- the target is not a valid tagged object: <code>embedded</code> has no reference,
  <code>gateway_model</code> has exactly one <code>provider_model_ref</code>, and
  <code>direct_endpoint</code> has exactly one <code>endpoint_ref</code>;
- an <code>endpoint_ref</code> is unresolved or lacks auth, privacy, identity, or
  reliability policy;
- the compatibility <code>prompt.model</code> does not resolve to the deployment's
  <code>provider_model_ref</code>;
- a gateway model is heterogeneous, or its resolver-derived provider/backend-set
  digest does not match the execution receipt;
- a referenced contract, task adapter, connector, capability, or major version
  is unknown, duplicated, or incompatible;
- binding requirements fail the field-level capability rules across connector,
  live endpoint, and receipt evidence;
- the binding requests a context not supported by its owner, task adapter, or
  connector;
- a decoder helper can name candidates outside the decision;
- the execution receipt does not match the live deployment identity, or the
  quality receipt does not match the binding and evaluation scenario;
- an active binding does not satisfy its environment admission policy;
- a shadow binding can affect routing output or plugin mutation;
- custom executable code is requested in embedded mode;
- an external endpoint lacks an explicit privacy policy and secret reference;
- a reload requests unsupported embedded replacement without a restart plan.

## Support, conformance, and operations

Execution and quality are tracked independently:

| Execution status | Meaning |
| --- | --- |
| E0 candidate | Discovered metadata only; no execution claim |
| E1 reference | Pinned artifact and task-contract fixtures pass in the isolated reference environment |
| E2 production | One production engine/hardware identity passes execution and failure conformance |
| E3 maintained | E2 identity is exercised continuously with a named owner and regression policy |

| Quality status | Meaning |
| --- | --- |
| Q0 unassessed | No router-scenario quality claim |
| Q1 benchmarked | Reproducible held-out metrics and slices are recorded |
| Q2 reviewed | Quality, calibration, cost, and latency are approved for a named scenario |
| Q3 monitored | Q2 scope has an owner and production/shadow regression policy |

An environment admission policy derives E/Q status from verified predicates;
receipts do not self-assign a trusted status. The recommended production policy
requires E2 + Q2 for active bindings and permits E1 + Q0/Q1 only in shadow. A
development policy may be less strict without changing binding identity.

Receipts use a versioned signed attestation envelope with subject digest,
predicate type/version, issuer, issue/expiry time, signature bundle, and claims.
Execution predicates record the deployment-capability identity, connector and
engine build, protocol, conformance tests, precision, hardware/provider,
performance, and license decision. Adapter predicates cover the resolved task
adapter and contract fixtures. Quality predicates record the binding identity,
evaluation dataset/slices, predeclared metrics and thresholds, calibration,
cost, latency, failure cases, reviewer, and scenario. A material subject change
requires new evidence unless an explicit compatibility rule covers it.

Conformance covers:

- exact labels, heads, taxonomies, route allowlists, tool schemas, and offsets;
- batch/single equivalence, determinism, Unicode, truncation, empty/oversized and
  malformed input;
- timeout, cancellation, bounded I/O, unavailable endpoint, invalid generative
  output, and declared fail-open/fail-closed/unknown behavior;
- reference parity, precision/quantization drift, cold start, concurrency,
  memory, latency, throughput, and effective accelerator provider;
- recipe behavior through the existing T0–T4 conformance framework.

Default PR CI remains model-free. Small public fixtures run on relevant PRs or
nightly; CPU/NVIDIA/AMD receipt matrices run on scheduled or release workflows;
gated/private partner tests use approved environments and never expose data.

Security requirements:

- pin commits and digests; prefer safetensors and data-only tokenizers;
- isolate custom code and block it in embedded mode;
- use secret references, bounded payloads, TLS/network policy, and prompt logging
  off by default;
- classify models as redistributable, reference-only, gated, non-commercial, or
  blocked/unknown;
- never bundle or select a gated/non-commercial model as an unrestricted default.

## Rollout and acceptance

### Phase 1: extension contracts and migration

Scope: define <code>ModelManifest</code>, <code>Deployment</code>, owner-local
<code>Binding</code>, named capabilities, the three frozen registries,
content-addressed identity, the four core task contracts, and signed receipts.
Add the v0.4 fail-closed schema gate and generation lease/drain lifecycle.
Migrate
<code>mmbert32k-intent-classifier-merged</code>,
<code>mmbert32k-pii-detector-merged</code>, and the merged HTTP classifier seam. Run
<code>Qwen/Qwen3-0.6B</code> through the existing PromptSelector/Looper
<code>model_choice.v1</code> path, and generalize the current direct
<code>/classify</code> connector with bounded I/O.

Exit criteria:

- reviewed-baseline intent and PII outputs are added as frozen golden fixtures,
  then pass <code>make test-category-classifier</code>,
  <code>make test-pii-classifier</code>, and selected T0–T4 recipe probes;
- sequence and token fixtures preserve every label and offset; HTTP classifier
  boundary tests reject request/response byte and batch limits at configured
  N+1 and assert deadline cancellation;
- Qwen returns only an allowlisted exact choice; invalid output takes the
  declared deterministic fallback;
- Looper helper calls bypass recipe resolution and recursive selection;
- duplicate/unknown registry IDs fail before preparation, descriptor discovery
  has no model/GPU/network side effects, and request handling uses compiled
  <code>ResolvedBinding</code> objects;
- activation rejects any capability that fails connector/live/receipt
  resolution and never runs offline conformance on reload;
- unsupported embedded reloads report <code>restart_required</code>;
- concurrent reload tests serialize publication, pin in-flight streams to one
  generation, leave live globals unchanged after candidate failure, and close
  owned resources after bounded drain;
- E2 execution receipts exist for the intent, PII, and Qwen deployments, and Q2
  receipts exist for each active Phase 1 binding;
- version, unknown-field, digest, live-identity, and receipt-subject mismatches
  fail before activation.

### Phase 2: Liquid models and user artifacts

Scope: add Liquid Prompt Router <code>candidate_scores.v1</code>, LFM2.5-350M
through <code>model_choice.v1</code>, Policy Linter
<code>rule_token_scores.v1</code>, Liquid task adapters over the direct-endpoint
connector,
user-artifact packaging and promotion, explicit placement, and provider
diagnostics.

Exit criteria:

- ranked scores, exact choice, and rule × token outputs are preserved without
  probability or span conversion;
- one GPU path reports the effective provider;
- one user artifact meets predeclared held-out quality, latency, and error
  thresholds, then completes conformance, shadow, promotion, and rollback;
- Replay export enforces opt-in, redaction, and tenant authorization.

### Phase 3: retrieval and ecosystem extensions

Scope: add rerank, late-interaction, contextual, streaming, and multimodal
contracts with selected P0/P1 models; add service deployment templates and
operator integration; generate compatibility documentation from receipts.

Exit criteria: every introduced contract has an owned model, conformance fixtures,
declared execution/quality status, and documented path coverage for ordinary,
streaming, and Looper execution. A clean-namespace deployment test covers the
service template, and receipt-derived compatibility documentation is
reproducible in CI.

## Traceability and references

**Reviewed baseline:**
<code>vllm-project/semantic-router@251455d6d2c17de81d1ec12274548ccfa34d8de2</code>
(2026-08-15)<br />
**Reviewed vLLM baseline:**
<code>vllm-project/vllm@5cecfc01375052698823fc401e31518fb32a981e</code>
(2026-08-15)<br />
**Original PR baseline:** <code>a0d75fd0</code>

Related issues:
[#2587](https://github.com/vllm-project/semantic-router/issues/2587),
[#2396](https://github.com/vllm-project/semantic-router/issues/2396),
[#2395](https://github.com/vllm-project/semantic-router/issues/2395),
[#2394](https://github.com/vllm-project/semantic-router/issues/2394),
[#2382](https://github.com/vllm-project/semantic-router/issues/2382),
[#2360](https://github.com/vllm-project/semantic-router/issues/2360),
[#2247](https://github.com/vllm-project/semantic-router/issues/2247),
[#2250](https://github.com/vllm-project/semantic-router/issues/2250),
[#2252](https://github.com/vllm-project/semantic-router/issues/2252), and
[#2760](https://github.com/vllm-project/semantic-router/issues/2760).

Related pull requests:
[semantic-router #2759](https://github.com/vllm-project/semantic-router/pull/2759)
(merged current classifier seam) and
[vLLM #42094](https://github.com/vllm-project/vllm/pull/42094)
(open DeBERTa work; not a support commitment).

Engine references:
[vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models.html),
[vLLM pooling models](https://docs.vllm.ai/en/latest/models/pooling_models/),
[vLLM ROCm installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html),
[TEI supported models](https://huggingface.co/docs/text-embeddings-inference/en/supported_models),
[ONNX Runtime execution providers](https://onnxruntime.ai/docs/execution-providers/),
and
[OpenVINO supported devices](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes.html).

The extension design was checked against pinned vLLM source: its
[architecture registry and capability inspection](https://github.com/vllm-project/vllm/blob/5cecfc01375052698823fc401e31518fb32a981e/vllm/model_executor/models/registry.py),
[behavioral model interfaces](https://github.com/vllm-project/vllm/blob/5cecfc01375052698823fc401e31518fb32a981e/vllm/model_executor/models/interfaces_base.py),
[separate loader registry](https://github.com/vllm-project/vllm/blob/5cecfc01375052698823fc401e31518fb32a981e/vllm/model_executor/model_loader/__init__.py),
[executor boundary](https://github.com/vllm-project/vllm/blob/5cecfc01375052698823fc401e31518fb32a981e/vllm/v1/executor/abstract.py),
[task vocabulary](https://github.com/vllm-project/vllm/blob/5cecfc01375052698823fc401e31518fb32a981e/vllm/tasks.py),
and
[capability-gated API registration](https://github.com/vllm-project/vllm/blob/5cecfc01375052698823fc401e31518fb32a981e/vllm/entrypoints/openai/api_server.py).

Repository references:
<code>src/semantic-router/pkg/config</code>,
<code>src/semantic-router/pkg/extproc</code>,
<code>src/semantic-router/pkg/selection</code>,
<code>src/semantic-router/pkg/modelruntime</code>,
<code>src/training</code>,
[model execution fallback](/docs/proposals/model-execution-fallback), and
[recipe conformance](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/CONFORMANCE.md).
