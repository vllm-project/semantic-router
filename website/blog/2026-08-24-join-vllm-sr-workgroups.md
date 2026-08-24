---
slug: "join-vllm-sr-workgroups"
title: "Find Your Focus: Join the vLLM Semantic Router Workgroups"
description: "Seven durable directions give contributors a clear place to build, lead, and grow with the vLLM Semantic Router community."
authors:
  - name: "vLLM Semantic Router Team"
    url: "https://github.com/vllm-project/semantic-router"
tags: ["community","ecosystem","semantic-router"]
image: "/img/blog/vllm/2026-08-24-workgroups-invitation/workgroups-invitation-hero.png"
---

Open source grows when people can see where their work belongs, who they can
build with, and how they can make a lasting contribution. vLLM Semantic Router
now has seven direction-based Workgroups to make that path clear.

![Find your focus and passion across seven vLLM Semantic Router Workgroups](/img/blog/vllm/2026-08-24-workgroups-invitation/workgroups-invitation-hero.png)

If you are new to the project, start here: vLLM Semantic Router sits between an
AI application and a collection of models. It understands each request,
chooses one model or a bounded way for several models to collaborate, executes
that choice, and makes the result observable and measurable. The seven
Workgroups own the durable technical directions needed to make that experience
intelligent, fast, reliable, and easy to use.

## Seven Directions, One Community

### [MoM & Routing](https://github.com/vllm-project/semantic-router/issues/2965)

> **Mission:** Make a pool of models behave like one measurable,
> continuously improving Mixture-of-Models.

#### Why this matters

A user should be able to call one stable model name without manually deciding
which backend to use for every request. A Mixture-of-Models (MoM) needs a model
pool and a versioned recipe: the pool defines which models are available, while
the recipe defines when and how they are used. Both must improve without making
behavior unpredictable.

#### Scope

- Design objective-specific model pools and portable routing recipes.
- Support different forms of model collaboration: direct selection, fallback,
  cascade, parallel answers with a judge, synthesis, and bounded workflows.
- Define how models and recipes are evaluated, shadowed, promoted, retired, and
  rolled back.
- Connect offline evaluation, routing outcomes, replay data, and human feedback
  to reviewable recipe improvements.
- Improve cross-model efficiency, including safe reuse of computation when a
  request moves between compatible models.

#### Non-scope

This Workgroup does not train the lightweight models that produce routing
signals, build the live network path, decide how conversation history is
compressed, or operate a hosted service. Those responsibilities have their own
Workgroups.

#### Epic directions

- [Portable MoM, recipe, evaluation, and model-card contracts](https://github.com/vllm-project/semantic-router/issues/2971)
- [Router Learning and recipe optimization](https://github.com/vllm-project/semantic-router/issues/2238)
- [Cross-model KV-cache reuse with LMCache](https://github.com/vllm-project/semantic-router/issues/2976)
- [Session-aware model and workflow switching with Agentic & Context](https://github.com/vllm-project/semantic-router/issues/2973)

### [Router Models & Inference Runtime](https://github.com/vllm-project/semantic-router/issues/2966)

> **Mission:** Build better Router Models and one extensible runtime for
> executing them across the ecosystem.

#### Why this matters

Before vLLM-SR can choose a route, it needs signals such as intent, complexity,
safety, preference, or expected quality. Router Models are the specialized
models that produce those signals and scores. Their quality should improve over
time, and adding a new model should not require spreading engine-specific code
throughout the Router.

#### Scope

- Post-train, distill, calibrate, and evaluate the Router Models already built
  into the project.
- Research routing-native Small Language Model families beyond BERT-only
  designs, including long-context, multimodal, retrieval, and agentic signals.
- Support privacy-aware fine-tuning and immutable candidate artifacts that can
  be evaluated before activation.
- Define typed inference inputs and outputs, model capabilities, artifact
  identity, activation, diagnostics, draining, and rollback.
- Qualify built-in and ecosystem execution through Candle, ONNX Runtime, vLLM,
  TEI, and other supported engines and hardware.

#### Non-scope

This Workgroup produces routing intelligence; it does not choose the end-user
MoM pool, own generic gateway forwarding, manage users and quotas, or rebuild
the tensor engines and GPU schedulers it integrates with.

#### Epic directions

- [Extensible routing inference runtime](https://github.com/vllm-project/semantic-router/issues/2782)
- [Built-in model post-training and router-native model families](https://github.com/vllm-project/semantic-router/issues/2974)
- [SLM self-improvement and fine-tuning with IBM Research](https://github.com/vllm-project/semantic-router/issues/2975)

### [Data Plane & Networking](https://github.com/vllm-project/semantic-router/issues/2967)

> **Mission:** Execute every live routing decision through a fast, reliable,
> and portable request path.

#### Why this matters

A good decision is useless if the request path is slow, fragile, or behaves
differently in each deployment. vLLM-SR needs one routing behavior whether it
runs as a standalone OpenAI-compatible service or integrates with Envoy and an
existing gateway environment.

#### Scope

- Process requests and responses, including streaming, dispatch, retries,
  fallback, errors, telemetry, and immediate responses.
- Provide a standalone mode for Docker and Kubernetes without making Envoy a
  mandatory dependency.
- Keep Envoy ExtProc and qualified gateway integrations first-class and
  behaviorally consistent with standalone mode.
- Define engine-neutral backend connectivity and cooperate with serving and
  load-balancing layers on inference-aware endpoint selection.
- Optimize latency, throughput, resource efficiency, streaming behavior, and
  failure recovery with reproducible measurements.

#### Non-scope

This Workgroup executes access and routing policy but does not define who may
access a model, which quota applies, which hardware is officially supported,
how a Router Model is trained, or which MoM recipe is best.

#### Epic directions

- [Protocol-neutral data plane and standalone adapter](https://github.com/vllm-project/semantic-router/issues/1138)
- [Inference-aware backend integration below semantic routing](https://github.com/vllm-project/semantic-router/issues/2332)
- Request-path performance, streaming, interoperability, and failure hardening

### [Enterprise & Environment](https://github.com/vllm-project/semantic-router/issues/2968)

> **Mission:** Make vLLM Semantic Router production-grade across supported
> organizations, environments, and hardware.

#### Why this matters

Production users need clear answers to practical questions: Who can call each
model? How much can they use? What changed? Is the system healthy? Can a model,
recipe, or Router upgrade be rolled out and reversed safely? The answers must
remain consistent across deployment environments.

#### Scope

- Registration, invitations, organizations, teams, serving identities, roles,
  and model access.
- API-key creation, rotation, revocation, request and token rate limits, usage
  accounting, tenant isolation, and audit.
- Stability, scalability, monitoring, diagnostics, multi-replica consistency,
  and failure recovery.
- Model and recipe onboarding, draining, gray release, promotion, rollback, and
  auditable vLLM-SR upgrades.
- A tested support matrix across Docker, Kubernetes, CPU, AMD, NVIDIA,
  precision, images, and maintained configurations.

#### Non-scope

This Workgroup does not promise a public hosted-service SLA, expose private
infrastructure or credentials, define model quality, own evaluation standards,
or implement networking protocols. It supplies reusable open-source production
capabilities rather than publishing private product plans.

#### Epic directions

- [Scalable multi-tenant inference access control](https://github.com/vllm-project/semantic-router/issues/2960)
- [Versioned management snapshots, activation, and rollback](https://github.com/vllm-project/semantic-router/issues/2326)
- Production observability, lifecycle controls, and environment/hardware
  qualification

### [Agentic & Context](https://github.com/vllm-project/semantic-router/issues/2987)

> **Mission:** Keep long-running agent workloads context-efficient,
> state-aware, bounded, and safe.

#### Why this matters

Long conversations accumulate messages, retrieved memory, tool output, tokens,
cost, and risk. Important instructions can be lost in that growth, and the best
model may change as the task moves from planning to tool use to final response.
vLLM-SR should manage those constraints without becoming a general-purpose
agent framework.

#### Scope

- Context compression, message and tool-output pruning, retrieval and memory
  selection, and prompt restructuring.
- Explicit protection for system and developer instructions, authorization,
  safety constraints, tool contracts, and task-critical user intent.
- Session budgets, state boundaries, retention, recovery, and graceful
  degradation when context or memory capabilities are unavailable.
- Requirements and safety constraints for switching models or bounded
  workflows as a session evolves.
- Evaluation of instruction retention, task quality, token reduction, latency,
  cost, safety, and critical-information loss.

#### Non-scope

This Workgroup does not build an unrestricted agent orchestrator, own all MoM
selection algorithms, transport KV caches between models, automate CLI
installation, or allow silent lossy transformation and unbounded online
training.

#### Epic directions

- [Traceable context optimization](https://github.com/vllm-project/semantic-router/issues/2984)
- [Long-session model and workflow switching with Looper research](https://github.com/vllm-project/semantic-router/issues/2973)
- [Gateway context envelope](https://github.com/vllm-project/semantic-router/issues/2546)
- [Router Memory stabilization](https://github.com/vllm-project/semantic-router/issues/2339)

### [Developer Experience & Ecosystem](https://github.com/vllm-project/semantic-router/issues/2970)

> **Mission:** Make vLLM Semantic Router easy to discover, install, configure,
> extend, and operate.

#### Why this matters

Technical depth has little impact when a new user cannot reach a first request
or understand what happened. The project also needs clear extension paths so
contributors, model builders, infrastructure projects, and educators can build
on it without reverse-engineering the repository.

#### Scope

- Installation, configuration, validation, deployment, tuning, diagnostics,
  and operations across the CLI, Dashboard, and APIs.
- Dashboard information architecture and developer or operator workflows built
  on maintained enterprise contracts.
- An agent-facing vLLM-SR skill for deployment, recipe generation, evaluation,
  tuning, and reviewed operations.
- Architecture documentation, tutorials, reference material, examples, model
  cards, evaluation results, blogs, video tutorials, and release guidance.
- Use case sharing, reference implementations, extension guidance, partner
  integrations, and contributor entry points.

#### Non-scope

This Workgroup does not control repository permissions or promotion, run
marketing events or AMD-internal programs, or redefine algorithms, production
policy, and quality standards owned by other Workgroups.

#### Epic directions

- [Agent skill for deployment, recipe generation, evaluation, and tuning](https://github.com/vllm-project/semantic-router/issues/2977)
- [A faster path to the first successful request](https://github.com/vllm-project/semantic-router/issues/2690)
- [Maintained reference recipes and tutorials](https://github.com/vllm-project/semantic-router/issues/2334)

### [Evaluation & Quality](https://github.com/vllm-project/semantic-router/issues/2969)

> **Mission:** Make every supported capability measurable and every change
> verifiable.

#### Why this matters

Claims about a Router Model, MoM recipe, runtime optimization, or deployment
are difficult to trust when each uses a different dataset and reporting method.
The project needs shared evaluation contracts and regression gates, while each
technical Workgroup remains accountable for the quality of what it builds.

#### Scope

- Common benchmark structure, dataset provenance, metrics, comparison rules,
  reproducibility, and result publication.
- MoM, recipe, Router Model, context, serving, platform, and developer-workflow
  evaluation without replacing each domain's objectives.
- Model-card and evaluation-result requirements for published MoM identities.
- CI, E2E, compatibility, security, performance, and operational regression
  coverage across supported modes and environments.
- Shared definitions of done and the evaluation required for milestone
  readiness, upgrade, and rollback.

#### Non-scope

This Workgroup does not choose another direction's quality target, accept an
issue, make the final release decision, replace Maintainer review, or own the
model research itself. It defines shared measurement and gates.

#### Epic directions

- [Evaluation and model cards within the portable MoM contract](https://github.com/vllm-project/semantic-router/issues/2971)
- [Decision-level routing evaluation](https://github.com/vllm-project/semantic-router/issues/2333)
- [Cross-platform performance regression coverage](https://github.com/vllm-project/semantic-router/issues/1510)
- [Module and Dashboard E2E coverage](https://github.com/vllm-project/semantic-router/issues/1519)

## What a Workgroup Is

> A Workgroup is a technical home, not a permission level.

Each Workgroup owns one durable project Direction. It keeps that charter clear,
organizes the bounded epics under it, helps triage related issues, recommends
when proposed work is clear enough to accept, and connects contributors with
people who share the same focus. Cross-cutting work still has one accountable
Workgroup and one named delivery owner.

| Workgroups | Open Source Team |
| --- | --- |
| Technical focus and contributor discovery | Project-wide governance |
| Direction, scope, boundaries, and epic coordination | Repository permissions and role promotion |
| Triage and acceptance recommendations | Final acceptance, review, merge, and release authority |
| Lead and Member roles | Maintainer, Committer, and Contributor roles |

The planning hierarchy is deliberately simple:

`Direction → Workgroup → domain → bounded epic → accepted issue`

## Roles and How to Join

An active Workgroup has at least one Lead. It may have multiple equal Leads and
multiple Members, and a person may participate in more than one Workgroup.

### Lead

A Lead is an active Committer or Maintainer, or a Contributor with at least one
merged commit and a named active Committer/Maintainer Sponsor.

Leads keep the charter and epic map understandable, coordinate triage,
recommend acceptance, identify work that is ready for a contributor, and make
sure each accepted initiative has a named delivery owner. The role provides
technical visibility and responsibility; it does not grant merge or release
authority.

### Member

A Member is a Contributor with at least one merged repository commit who wants
to keep building in the Direction. Members help shape epics, contribute to
delivery, review work when eligible, and welcome people entering the area.

External researchers, users, and partners can collaborate before they meet the
Member requirement. Their participation is valuable, but a merged contribution
is required for the public Member roster.

### Join an Existing Workgroup

Open the Workgroup charter linked from its name above and comment with:

1. your GitHub handle and requested role;
2. a merged contribution;
3. the part of the Direction you want to focus on;
4. your expected availability; and
5. your Sponsor, when applying as a sponsored Lead.

Confirmed Leads and Members appear with their avatar and name on the public
[Workgroups map](/community/work-groups), helping new contributors find the
people building in each area.

## How to Propose a New Workgroup

### When a new Workgroup makes sense

Propose one when the community faces an important, long-lived technical problem
that cannot fit cleanly into any existing charter. A useful Workgroup should
remain relevant across several releases and contain more than one bounded epic.

### When to create an epic instead

A single feature, algorithm, research collaboration, partner integration,
event, or release milestone belongs inside an existing Workgroup. These efforts
can be important without requiring permanent organizational structure.

### What the proposal must explain

Open a proposal issue that includes:

1. the problem and durable Direction;
2. why the seven existing Workgroups cannot own it clearly;
3. the goal, scope, non-scope, and interfaces with other Workgroups;
4. an initial map of bounded epic directions; and
5. at least one eligible Lead willing to form the group.

Maintainers review the boundary, community need, and proposed leadership. Once
accepted, the project creates the charter and owner label, opens Lead and Member
self-nomination, and adds the Workgroup to the website.

## Come Build With Us

Start with the [Workgroups map](/community/work-groups), open the charter that
matches your interests, and introduce yourself. You can join an existing epic,
help shape a contributor-ready issue, share a use case, or propose a focused
piece of work.

Find your focus and passion. Build with the community. Grow together.
