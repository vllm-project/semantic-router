---
slug: "vllm-sr-new-chapter-mom"
title: "Beyond a Single Model: Building Mixture-of-Models Systems with vLLM Semantic Router"
description: "vLLM Semantic Router is expanding from intelligent routing into a system for building, evaluating, and running Mixture-of-Models."
authors:
  - name: "vLLM Semantic Router Team"
    url: "https://github.com/vllm-project/semantic-router"
tags: ["ecosystem","mixture-of-models","semantic-router"]
image: "/img/blog/vllm/2026-07-21-vllm-sr-new-chapter/banner.png"
source_url: "https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-07-21-vllm-sr-new-chapter-mom.md"
---
Most AI applications are built around a single model endpoint. But as models, devices, and deployment constraints diversify, no single model is the best fit for every request or environment. The practical question is how multiple specialized models can be coordinated, evaluated, and served through one interface. We call this systems approach **Mixture-of-Models**.

In less than a year since its public launch, [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) has reached **5,000 stars**, **150+ contributors**, and **more than 300,000 cumulative downloads** across our Hugging Face model family. Across three major releases—**Iris, Athena, and Themis**—the system boundary moved from choosing a model, to governing multi-model inference, to preserving state and coordination across sessions. Those releases built the foundation for the MoM architecture envisioned from day 0.

This post describes the next step for vLLM Semantic Router: moving from routing among models to building dependable model systems from them. Under one versioned contract, independent models, policies, preferences, and execution paths become a system that can be trained, evaluated, exported, imported, deployed, and invoked through one interface. Our goal is to make vLLM Semantic Router a training, evaluation, and inference engine for Mixture-of-Models.

<p align="center">
  <img src="/img/blog/vllm/2026-07-21-vllm-sr-new-chapter/hero.png" alt="A model portfolio flows into a Mixture-of-Models training, evaluation, and inference engine and is exposed as one model" width="100%" />
  <br />
  <em>Figure 1: A Mixture-of-Models turns a heterogeneous model portfolio into one model experience.</em>
</p>

## How vLLM-SR Got Here

The [first vLLM Semantic Router post](/blog/semantic-router) asked a practical question: why give simple and difficult requests the same reasoning budget? A lightweight classifier used fixed domain labels to choose between fast and reasoning paths, helping vLLM spend inference compute more selectively.

Production traffic quickly exposed the limit of that design. Domain alone could not represent privacy, safety, context, language, modality, tools, preferences, latency, and authorization. A static label also could not account for an endpoint that was cheap but overloaded, capable but remote, or unsafe to switch into midway through an agent session.

We rebuilt the classifier layer around modular model support, shared LoRA computation, Rust/Candle inference, and Go integration. We then replaced fixed classification with a Signal–Decision architecture that separated observed evidence from policy and execution. This became the spine of the next three releases.

| Milestone | When | What changed |
| --- | --- | --- |
| **Incubation** | Apr 2025 | Early semantic-routing prototypes began with Mixture-of-Models as the long-term system goal |
| **Initial release** | Sep 2025 | Intent-aware selection between fast and reasoning paths |
| **v0.1 Iris** | Jan 2026 | Signals, decisions, and route-scoped plugins replaced fixed classification |
| **v0.2 Athena** | Mar 2026 | Model selection, memory, RAG, long context, and multimodality expanded routing into an inference control system |
| **v0.3 Themis** | Jun 2026 | Stateful routing, projections, replay, protocol support, session continuity, and one production configuration contract made the system operable |
| **Fusion and Micro-Agent** | Jun 2026 | The router began choosing collaboration patterns, not only individual models |

<p align="center">
  <img src="/img/blog/vllm/2026-07-21-vllm-sr-new-chapter/evolution.png" alt="vLLM Semantic Router evolves from intent routing through Iris, Athena, and Themis into a Mixture-of-Models engine" width="100%" />
  <br />
  <em>Figure 2: Each stage changed the unit of control: model, decision, system, session, and finally the complete model lifecycle.</em>
</p>

[Iris](/blog/vllm-sr-iris) made routing composable. Domain, keyword, embedding, factuality, feedback, and preference signals fed explicit decisions, while safety, PII protection, caching, hallucination detection, and tool selection became route-scoped behavior. Iris also introduced the MoM model family and described vLLM-SR as “System Level Intelligence for Mixture-of-Models.”

[Athena](/blog/v0.2-vllm-sr-athena-release) added first-class model selection, memory and RAG, a multilingual and multimodal model stack, ROCm acceleration, and an operating dashboard. The project was becoming the control system around multi-model inference, not just a classifier in front of vLLM.

[Themis](/blog/v0.3-vllm-sr-themis-release) turned that broader system into an operable contract:

> **Signals become projections. Projections feed decisions. Decisions choose algorithms. Algorithms select models.**

Themis added session-aware agentic routing, replayable traces, stronger protocol support, an operator console, and runtime paths across AMD ROCm, NVIDIA CUDA, Intel OpenVINO, and CPU environments. It also made a route explainable: operators can see the evidence, policy, algorithm, and physical model behind each decision.

### From Signal–Decision to Workload–Router–Pool

The releases built the runtime. Two project papers explained the architecture behind it.

The [white paper, *Signal Driven Decision Routing for Mixture-of-Modality Models*](https://vllm-sr.ai/white-paper/), formalized the separation between neural evidence and symbolic policy. Fast heuristics and learned classifiers turn prompts, context, identity, safety, and modality into a structured signal vector; a Boolean engine then composes those signals into auditable policy. A typed neural-symbolic DSL parses and validates that policy before compiling it into deployable configuration. When the paper was published, the system covered thirteen signal types and thirteen model-selection algorithms, with per-decision plugins for caching, RAG, memory, safety, provider handling, and response validation.

The [vision paper, *The Workload–Router–Pool Architecture for LLM Inference Optimization*](https://vllm-sr.ai/vision-paper/), widened the frame. It argues that three variables have to be designed together:

- **Workload:** chat or agent, single-turn or multi-turn, warm or cold, prefill-heavy or decode-heavy
- **Router:** static semantic policy, online feedback or bandit adaptation, RL-based selection, and quality-aware cascades
- **Pool:** homogeneous or heterogeneous accelerators, prefill/decode topology, model placement, and KV-cache management

Those variables cannot be optimized independently. Workload shape changes which routing policy works; routing policy changes the required pool size and topology; pool state changes which route is efficient. Safety and privacy cut across all three dimensions, while cost, quality, latency, and energy define the optimization frontier. The paper maps the project's research into a 3 × 3 WRP matrix and identifies twenty-one open directions where those dimensions still need to meet.

<p align="center">
  <img src="/img/blog/vllm/2026-07-21-vllm-sr-new-chapter/research-arc.png" alt="The white paper formalizes signal decision routing while the vision paper connects workload router and pool co-design" width="100%" />
  <br />
  <em>Figure 3: The white paper defines the programmable routing engine; the vision paper connects it to workload and physical pool design.</em>
</p>

Together, the papers made routing programmable and tied it to workload and hardware—the two foundations MoM brings under one model contract.

Meanwhile, the runtime was already moving beyond single-model selection. [Fusion](/blog/vllm-sr-fusion-api), ReMoM, Confidence, Ratings, and bounded Workflows let one request invoke a controlled collaboration among models. As the [Micro-Agent work](/blog/micro-agent-frontier-models) showed, a client can call one model name while the serving layer selects a recipe, fans out to workers, verifies or synthesizes their results, and returns one ordinary response.

| First chapter | New chapter |
| --- | --- |
| Route a request | Build a model system |
| Choose a model or capability path | Train, evaluate, and execute the whole MoM |
| Configure runtime policy | Package a portable, versioned model artifact |
| Optimize a routing decision | Optimize system intelligence across quality, cost, latency, safety, and energy |
| Hide backend choice behind one API | Make the complete multi-model system behave like one model |

Routing remains fundamental. It is how a Mixture-of-Models allocates work, applies policy, and coordinates its parts. But routing is the mechanism. **The model system is the product.**

## Why the Model Boundary Has to Move

Today's AI stack is fragmented along four axes:

+ **Models are fragmented.** Closed frontier models, open general models, domain experts, compact local models, verifiers, and multimodal models will coexist. None wins simultaneously on quality, cost, latency, trust, privacy, and domain fit.

+ **Compute is fragmented.** GPUs, CPUs, specialized accelerators, edge devices, cloud capacity, and private clusters differ in memory, kernels, availability, price, and energy use. Model choice and placement are becoming the same decision.

+ **Location is fragmented.** Inference spans cloud, data center, and edge. Privacy or residency may rule out a stronger remote model, while a local workload may still need an on-demand cloud expert.

+ **Preference is fragmented.** There is no universal “best.” Products and users make different tradeoffs among accuracy, latency, price, privacy, safety, style, and multimodality. Those choices should shape execution directly.

Today, each application has to reconcile these fragments on its own.

<p align="center">
  <img src="/img/blog/vllm/2026-07-21-vllm-sr-new-chapter/fragmentation-before-mom.png" alt="Before Mixture-of-Models, each application owns separate routing glue across fragmented models, compute, locations, and preferences" width="100%" />
  <br />
  <em>Figure 4: Before MoM, fragmented intelligence becomes application-side routing glue.</em>
</p>

Mixture-of-Models moves that responsibility behind one model boundary.

At that boundary, **intelligent allocation** becomes part of the model. The engine determines which models are eligible, where execution can run, whether models should collaborate, and how to satisfy hard constraints.

Energy makes allocation inseparable from efficiency. Hardware and inference engines improve the supply side by producing more tokens per watt per dollar. The allocation layer controls demand: which work deserves those tokens, and which model or collaboration can provide them within the required quality, latency, and energy budget.

The application selects one versioned model identity and receives one attributable response. Its physical realization can still span open and closed models, cloud and edge, and different accelerator generations. The fragmentation remains, but it becomes internal to the model system instead of leaking into every application.

<p align="center">
  <img src="/img/blog/vllm/2026-07-21-vllm-sr-new-chapter/fragmentation-after-mom.png" alt="With Mixture-of-Models, one model identity contains intelligent allocation across fragmented models, compute, locations, and preferences" width="100%" />
  <br />
  <em>Figure 5: With MoM, the same fragmented resources become the internal realization of one model.</em>
</p>

## What We Mean by Mixture-of-Models

A **Mixture-of-Models** is a versioned composite model whose engine realizes each request through a preference-conditioned, resource-bounded path across independent models and operators. It is presented to the user through one model interface and returns one attributable result.

A multi-upstream gateway can forward traffic without owning system quality. An MoM owns an objective, an evaluation contract, a reproducible composition, and the runtime that executes it.

MoM also differs from Mixture-of-Experts. MoE routes tokens among internal experts during one forward pass; MoM coordinates independent models that may differ in architecture, owner, license, modality, protocol, context window, and hardware. An MoE checkpoint can itself be one MoM component.

| | Conventional model | Mixture-of-Models |
| --- | --- | --- |
| Unit of intelligence | One checkpoint | A governed system of models |
| Specialization | Primarily encoded in weights | Composed across independent specialists |
| Execution | One generation path | Selection, cascade, verification, fusion, or workflow |
| Optimization target | One model's quality and efficiency | The system frontier across quality, cost, latency, safety, privacy, and energy |
| Deployment boundary | One runtime | Cloud, data center, and edge |
| User contract | One model identity | One model identity |

<p align="center">
  <img src="/img/blog/vllm/2026-07-21-vllm-sr-new-chapter/execution-topologies.png" alt="One model call enters a Mixture-of-Models engine that may select, cascade, fuse, or execute a bounded workflow before returning one response" width="100%" />
  <br />
  <em>Figure 6: Selection is one MoM topology. Cascades, parallel fusion, and bounded workflows share the same model boundary.</em>
</p>

A portable MoM therefore needs more than weights and configuration: it needs a component manifest, capability metadata, routing and collaboration recipes, policies, preferences, evaluation suites, runtime constraints, provenance, and version history.

Open checkpoints can travel with the artifact; closed models remain authenticated external references with explicit capability and policy contracts. Exporting an MoM does not make a proprietary checkpoint portable. It makes the **model system** reproducible.

### Turn Preferences into Models

Preferences become concrete when they are published as model identities. One MoM family can offer several operating points:

| Model identity | Contract |
| --- | --- |
| `vllm-sr/mom-v1-flash` | Minimize expected latency |
| `vllm-sr/mom-v1-light` | Minimize cost above a quality floor |
| `vllm-sr/mom-v1-ultra` | Maximize quality within a declared budget |
| `vllm-sr/mom-v1-halu` | Require grounding checks and fail-closed fallback |
| `vllm-sr/mom-v1-secu` | Enforce jailbreak and PII policy before execution |

Each name is a versioned model contract, not a router preset. The application chooses the behavior it needs; vLLM-SR selects and coordinates the models that deliver it while preserving hard privacy, residency, authorization, and safety constraints.

To an application, the full system remains an ordinary model call:

```json
{
  "model": "vllm-sr/mom-v1-ultra",
  "messages": [
    {"role": "user", "content": "Review this design and identify its weakest assumption."}
  ]
}
```

That identity may select one model, escalate through a cascade, compare parallel answers, require grounding, or run a bounded workflow—without changing the external interface, version, or response contract.

<p align="center">
  <img src="/img/blog/vllm/2026-07-21-vllm-sr-new-chapter/preference-models.png" alt="One Mixture-of-Models family exposes flash, light, ultra, grounding, and security variants as individually versioned model identities" width="100%" />
  <br />
  <em>Figure 7: Preferences are published as bounded, versioned model contracts—not hidden application-side routing presets.</em>
</p>

Four planes separate ownership:

| Plane | What it owns | Foundation already in vLLM-SR | Next step |
| --- | --- | --- | --- |
| **Artifact** | Components, capabilities, objectives, policy, eval contract, provenance | Canonical config, model references, DSL, versioned policy | Portable MoM import/export specification |
| **Learning** | Router-owned models, preferences, outcomes, recipe improvement | Training stack, Router Learning, replay, outcome APIs | Joint training and system-level release gates |
| **Execution** | Signals, projections, decisions, selectors, loopers, plugins | Signal–Decision runtime, Fusion, ReMoM, Workflows, safety and memory | One lifecycle-aware MoM engine |
| **Physical** | Providers, model pools, accelerators, locality, cache and energy state | vLLM backends, cloud providers, ROCm, CUDA, OpenVINO, CPU | Portable placement across cloud, data center, edge, and local devices |

<p align="center">
  <img src="/img/blog/vllm/2026-07-21-vllm-sr-new-chapter/four-planes.png" alt="Artifact, learning, execution, and physical planes combine to define, improve, realize, and run one Mixture-of-Models identity" width="100%" />
  <br />
  <em>Figure 8: A complete MoM spans four planes: artifact, learning, execution, and physical realization.</em>
</p>

A deployment must map logical requirements onto the models and machines available in its environment. The proposal uses four objects:

1. The **bundle** fixes the interface, graph, policies, behavior variant, bounds, and immutable semantic assets.
2. The **binding** maps logical components to eligible deployments without changing the model's decision semantics.
3. The **resolution lock** freezes the constituent revisions, runtimes, images, accelerators, and provider observations.
4. The **run record** attributes every decision, call, constraint check, cost, and outcome to the bundle, binding, and lock that produced it.

<p align="center">
  <img src="/img/blog/vllm/2026-07-21-vllm-sr-new-chapter/artifact-resolution-lifecycle.png" alt="One Mixture-of-Models identity moves through bundle, binding, resolution lock, and run record without changing its logical model name" width="100%" />
  <br />
  <em>Figure 9: One stable model identity, from portable contract to attributable run.</em>
</p>

This separation keeps portability honest. The same `mom-v1-ultra` can bind to ROCm, CUDA, a private CPU or NPU node, or a hybrid deployment without promising identical outputs from opaque providers. Instead, it preserves control semantics, exposes substitutions, and gives serving and evaluation the same resolved system.

## vLLM-SR as the MoM Engine

Training, evaluation, and inference must share one contract; otherwise research, benchmarks, and production drift into different systems.

### Training allocation, not only weights

MoM training covers router-owned embeddings, signal encoders, preference and safety models, and selectors. It also learns allocation and collaboration: which path fits a workload and budget, when a cascade should stop, how a panel should judge or synthesize, and when an agent session should switch models. Because constituents may be independent or closed, progress does not require gradients through all of them; policies, thresholds, pools, prompts, contracts, and topology can be optimized from traces and outcomes.

The target is a frontier across quality, latency, cost, safety, privacy, reliability, locality, and energy. Replay and outcomes feed production experience back into offline training without letting the hot path silently rewrite policy.

### Evaluating the MoM as one model

Evaluation must score the model identity end to end; backend benchmarks are inputs, not the result. A versioned scorecard should measure routing regret, collaboration gain, recovery, session continuity, tail latency, cost, safety, privacy, and energy. It should stress provider failures, device loss, model disagreement, workload drift, and preference changes. Each declared operating point also needs its own test: `flash` on its latency–quality frontier, `light` against its quality floor, and `ultra` within its budget.

The scientific test is stricter than asking whether more calls improve a benchmark. Under matched active compute, can a conditional system exploit complementary strengths and failure modes better than the best fixed model? Without that control, MoM can hide brute-force scaling behind a clever graph. Evaluations must report calls, tokens, cost, latency, and energy alongside quality—and publish when composition does not help.

<p align="center">
  <img src="/img/blog/vllm/2026-07-21-vllm-sr-new-chapter/matched-compute-evaluation.png" alt="A best fixed model and a conditional Mixture-of-Models are compared under matched active compute using quality, calls, tokens, cost, latency, and energy" width="100%" />
  <br />
  <em>Figure 10: Composition gain is meaningful only under matched active compute, with quality reported alongside calls, tokens, cost, latency, and energy.</em>
</p>

### Executing intelligence at inference time

At inference time, the engine decides whether one model is enough. It may choose a local specialist, preserve a warm session, escalate through a confidence cascade, require retrieval or verification, run a Fusion panel, or execute a bounded workflow. The runtime owns the budget, topology, fallback, trace, and response contract; the application makes a normal model call.

<p align="center">
  <img src="/img/blog/vllm/2026-07-21-vllm-sr-new-chapter/mom-lifecycle.png" alt="A portable Mixture-of-Models artifact moves through training evaluation inference and outcome feedback while preserving one model identity" width="100%" />
  <br />
  <em>Figure 11: MoM is a closed lifecycle: train the allocation policy, evaluate the full system, execute it, and turn outcomes into the next validated version.</em>
</p>

## One Model That Can Move

Our target is a complete MoM that can be **built, exported, imported, versioned, evaluated, deployed, and invoked as a unified model**. A logical specification compiles into an immutable bundle, binds to an environment, resolves the concrete deployment, and retains the same identity for serving and evaluation.

The artifact should run across developer machines, private clusters, cloud fleets, and edge environments while its physical realization changes. A specialist may resolve to an admissible local checkpoint or managed endpoint; an accelerator runtime may be replaced. If privacy makes a remote expert unavailable, the engine follows a declared fallback or abstention path. A binding cannot silently rewrite the graph, relax a guard, or turn a panel into a cascade—those changes require a new model version.

“Run on any hardware” is an architectural requirement, not a claim that every component is portable today. The project already supports paths across ROCm, CUDA, OpenVINO, and CPU. Next, hardware capability and placement become part of the MoM contract, allowing the engine to map the model system onto what is available.

The standard for the user experience is simple:

> **One model identity. Many models. Any hardware.**

<p align="center">
  <img src="/img/blog/vllm/2026-07-21-vllm-sr-new-chapter/portable-realizations.png" alt="One vLLM Semantic Router Mixture-of-Models identity is packaged as a bundle and bound to developer, data-center, cloud, and edge environments" width="100%" />
  <br />
  <em>Figure 12: One logical model identity can be realized across developer, data-center, cloud, and edge hardware.</em>
</p>

If the application needs to know which provider owns every submodel, which device runs it, or which fallback graph to execute, the abstraction has leaked.

## What Changes Now

The next stage focuses on four connected areas:

1. **Define a portable MoM specification.** Package components, objectives, policy, preferences, evaluation, constraints, and execution semantics as one versioned artifact.
2. **Close the training–evaluation–inference loop.** Improve models and recipes from evaluation and replay, then ship them through reviewable, rollback-safe releases.
3. **Build a heterogeneous runtime.** Map one MoM across cloud, data center, and edge using hardware, locality, energy, and data boundaries as inputs.
4. **Keep the model interface boring.** Make an MoM as easy to import, deploy, and invoke as a single model.

<p align="center">
  <img src="/img/blog/vllm/2026-07-21-vllm-sr-new-chapter/next-stage-roadmap.png" alt="The next vLLM Semantic Router chapter connects a portable specification, a closed training evaluation inference loop, a heterogeneous runtime, and one model API" width="100%" />
  <br />
  <em>Figure 13: Four connected workstreams turn Mixture-of-Models from an execution pattern into the next model architecture.</em>
</p>

This is a research program for how independent models should specialize, compete, verify, and collaborate; how to measure the resulting system; and how one model contract can survive across devices and environments. Our mission is:

> **Advancing the science of intelligence across models, devices, and environments.**

We will study when composition produces capabilities beyond a single checkpoint, treat placement and energy as part of intelligence, and carry the same model contract from edge to cloud and from research to production.

## Build It With Us

Building Mixture-of-Models requires more than routing. The work spans model training, evaluation, serving systems, hardware, and production operations.

Iris, Athena, and Themis improved because contributors brought real workloads, added backends, trained models, published benchmarks, found failure cases, and argued for better interfaces. MoM needs the same range of work: learned allocation, preference optimization, model cooperation, energy-aware inference, portable artifacts, open evaluation, and heterogeneous runtimes.

If you work on these problems, we want to learn from your workloads and measurements. Build an operating point, add a runtime, test a collaboration recipe, or publish a case where composition fails. MoM will be stronger if its assumptions are tested in the open.

### Acknowledgments

vLLM-SR has grown through work across engineering, research, and the wider ecosystem. We thank [Xunzhuo Liu](https://www.linkedin.com/in/bitliu), [Huamin Chen](https://www.linkedin.com/in/huaminchen), [Bowei He](https://www.linkedin.com/in/bowei-he-8a9450199/), [Yankai Chen](https://www.linkedin.com/in/yankai-chen-923001154/), [Fuyuan Lyu](https://www.linkedin.com/in/fuyuan-lyu-560756167/), and [Steve Liu](https://ca.linkedin.com/in/xueliu) for helping shape its technical and research direction. We also thank [Andy Luo](https://www.linkedin.com/in/andyluo77/) and [Haichen Zhang](https://www.linkedin.com/in/haichen-zhang-9010b6382/) for their work on ROCm enablement, router-model training, and open MoM experimentation.

The work has also been carried by [FAUST](https://github.com/FAUST-BENCHOU), [David Shrader](https://www.linkedin.com/in/shraderdm/), [Yang Wu](https://github.com/drivebyer), [Ramakrishnan Sathyavageeswaran](https://github.com/ramkrishs), [Kuntai Wu](https://github.com/WUKUNTAI-0211), [Aayush Saini](https://github.com/AayushSaini101), [siloteemu](https://github.com/siloteemu), [Chen Wang](https://www.linkedin.com/in/chenw615/), [Yue Zhu](https://www.linkedin.com/in/yue-zhu-b26526a3/), [Senan Zedan](https://www.linkedin.com/in/senan-zedan-2041855b/), [Yossi Ovadia](https://www.linkedin.com/in/yossi-ovadia-336b314/), [Samzong Lu](https://www.linkedin.com/in/samzong), [Liav Weiss](https://www.linkedin.com/in/liav-weiss-2a0428208), [Asaad Balum](https://www.linkedin.com/in/asaad-balum-0928771a9/), [Yehudit](https://www.linkedin.com/in/yehuditkerido/), [Noa Limoy](https://www.linkedin.com/in/noalimoy/), [Marina Koushnir](https://github.com/mkoushni), [Jared Wen](https://github.com/JaredforReal), [Abdallah Samara](https://www.linkedin.com/in/abdallah-samara), [Hen Schwartz](https://www.linkedin.com/in/henschwartz), [Srinivas A](https://www.linkedin.com/in/sriniabhiram), [Yang Zhu](https://github.com/carlory), [Jintao Zhang](https://www.linkedin.com/in/jintao-zhang-402645193/), [yuluo-yx](https://github.com/yuluo-yx), [cryo](https://github.com/cryo-zd), [Bishen Yu](https://github.com/OneZero-Y), [Zhijie Wang](https://github.com/aeft), [Hao Wu](https://github.com/haowu1234), and [Qiping Pan](https://www.linkedin.com/in/qiping-pan-8662ab215/). Their code, reviews, testing, documentation, and stewardship carried the project from one release to the next.

At this milestone, the project stands at **1,734 commits** and **150+ contributors**. We thank collaborators at MBZUAI, McGill University, Mila, and Rice University, and the broader vLLM, AMD, Intel, Meta, Red Hat, Microsoft, Google, IBM, NVIDIA, Hugging Face, NASA, Nutanix, DaoCloud, and open-source communities. This milestone belongs to everyone who helped turn an early router into a real system.

<p align="center">
  <img src="/img/blog/vllm/2026-07-21-vllm-sr-new-chapter/community.png" alt="Model researchers, evaluation researchers, systems engineers, hardware teams, model builders, and operators collaborate to build the Mixture-of-Models engine" width="100%" />
  <br />
  <em>Figure 14: Building the MoM engine is an open systems problem that needs the full model and infrastructure community.</em>
</p>

Join us on [GitHub](https://github.com/vllm-project/semantic-router), explore the [documentation](https://vllm-sr.ai), try the [MoM model family](https://huggingface.co/LLM-Semantic-Router), and meet the community in the `#semantic-router` channel on [vLLM Slack](https://vllm-dev.slack.com/archives/C09CTGF8KCN).

vLLM Semantic Router began by helping infrastructure choose the right model for each request.

Now we are extending that foundation beyond a single model: toward systems that can coordinate, evaluate, and operate multiple models across devices and environments.

We invite the community to help build and test that approach in the open.
