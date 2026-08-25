---
slug: "join-vllm-sr-workgroups"
title: "Find Your Focus and Join a Workgroup"
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
AI application and its model and agent backends. It understands each request
and session, selects a qualified model or agent, can coordinate a bounded form
of model or agent collaboration, executes that route, and makes the result
observable and measurable. The seven Workgroups own the durable technical
directions needed to make that experience intelligent, fast, reliable, and
easy to use.

## How the Pieces Fit Together

A routed request crosses several responsibilities, but each responsibility has
one clear technical home:

1. **Developer Experience & Ecosystem** gives users the CLI, Dashboard, APIs,
   recipes, and guidance needed to describe and operate the system.
2. **Enterprise & Environment** determines who may use each backend, applies
   quotas and lifecycle policy, and keeps supported deployments operable.
3. **Router Models & Inference Runtime** turns the request and session into
   useful routing signals.
4. **Agentic & Context** manages bounded context and uses those signals to
   select, hand off, or compose agent backends for long-running sessions;
   **MoM & Routing** selects models or bounded multi-model collaboration.
5. **Data Plane & Networking** invokes the selected model or agent path, while
   **Evaluation & Quality** measures behavior and protects it from regression.

The Workgroups therefore form one system. They are separated by responsibility,
not by isolated code ownership.

Choose the Direction that matches the problem you want to solve. Each section
shows its motivation, ownership boundary, and accepted Epic directions.

## [MoM & Routing](https://github.com/vllm-project/semantic-router/issues/2965)

> **Mission:** Make a pool of models behave like one measurable,
> continuously improving Mixture-of-Models.

![One request enters a versioned recipe and qualified model pool, which can select, cascade, compare, or combine models before returning one measurable response](/img/blog/vllm/2026-08-24-workgroups-invitation/workgroups/mom-routing.svg)

### Why this matters

A user should be able to call one stable model name without manually deciding
which backend to use for every request. A Mixture-of-Models (MoM) needs a model
pool and a versioned recipe: the pool defines which models are available, while
the recipe defines when and how they are used. Both must improve without making
behavior unpredictable.

### Scope

- Design objective-specific model pools and portable routing recipes.
- Support different forms of model collaboration: direct selection, fallback,
  cascade, parallel answers with a judge, synthesis, and bounded workflows.
- Define how models and recipes are evaluated, shadowed, promoted, retired, and
  rolled back.
- Connect offline evaluation, routing outcomes, replay data, and human feedback
  to reviewable recipe improvements.
- Improve cross-model efficiency, including safe reuse of computation when a
  request moves between compatible models.

### Non-scope

This Workgroup does not train the lightweight models that produce routing
signals, build the live network path, decide how conversation history is
compressed, or operate a hosted service. Those responsibilities have their own
Workgroups.

### Epic directions

- [Define portable MoM model pools, routing recipes, evaluation, and model cards](https://github.com/vllm-project/semantic-router/issues/2971)
- [Optimize routing recipes through an offline-to-online lifecycle](https://github.com/vllm-project/semantic-router/issues/2238)
- [Enable safe cross-model KV-cache reuse](https://github.com/vllm-project/semantic-router/issues/2976)

## [Router Models & Inference Runtime](https://github.com/vllm-project/semantic-router/issues/2966)

> **Mission:** Build better Router Models and one extensible runtime for
> executing them across the ecosystem.

![A Router Model family improves through a model flywheel while a layered inference runtime executes versioned artifacts and emits typed signals](/img/blog/vllm/2026-08-24-workgroups-invitation/workgroups/router-models-inference-runtime.svg)

### Why this matters

Before vLLM-SR can choose a route, it needs signals such as intent, complexity,
safety, preference, or expected quality. Router Models are the specialized
models that produce those signals and scores. Their quality should improve over
time, and adding a new model should not require spreading engine-specific code
throughout the Router.

### Scope

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

### Non-scope

This Workgroup produces routing intelligence; it does not choose the end-user
MoM pool, own generic gateway forwarding, manage users and quotas, or rebuild
the tensor engines and GPU schedulers it integrates with.

### Epic directions

- [Build an extensible inference runtime for Router Models](https://github.com/vllm-project/semantic-router/issues/2782)
- [Improve current Router Models and develop routing-native model families](https://github.com/vllm-project/semantic-router/issues/2974)
- [Build a reproducible SLM self-improvement and Router Model fine-tuning pipeline](https://github.com/vllm-project/semantic-router/issues/2975)

## [Data Plane & Networking](https://github.com/vllm-project/semantic-router/issues/2967)

> **Mission:** Execute every live routing decision through a fast, reliable,
> and portable request path.

![Direct HTTP and Envoy gateway entry paths converge on one shared routing core, backend dispatch path, and response stream](/img/blog/vllm/2026-08-24-workgroups-invitation/workgroups/data-plane-networking.svg)

### Why this matters

A good decision is useless if the request path is slow, fragile, or behaves
differently in each deployment. vLLM-SR needs one routing behavior whether it
runs through its direct OpenAI-compatible HTTP path or integrates with Envoy and an
existing gateway environment.

### Scope

- Process requests and responses, including streaming, dispatch, retries,
  fallback, errors, telemetry, and immediate responses.
- Provide direct HTTP serving for Docker and Kubernetes without making Envoy a
  mandatory dependency.
- Keep Envoy ExtProc and qualified gateway integrations first-class and
  behaviorally consistent with the direct path.
- Define engine-neutral backend connectivity and cooperate with serving and
  load-balancing layers on inference-aware endpoint selection.
- Optimize latency, throughput, resource efficiency, streaming behavior, and
  failure recovery with reproducible measurements.

### Non-scope

This Workgroup executes access and routing policy but does not define who may
access a model, which quota applies, which hardware is officially supported,
how a Router Model is trained, or which MoM recipe is best.

### Epic directions

- [Support direct HTTP and gateway-integrated data paths](https://github.com/vllm-project/semantic-router/issues/1138)
- [Connect semantic routing to inference-aware backend selection](https://github.com/vllm-project/semantic-router/issues/2332)
- [Optimize data-plane performance, streaming, and failure recovery](https://github.com/vllm-project/semantic-router/issues/2992)

## [Enterprise & Environment](https://github.com/vllm-project/semantic-router/issues/2968)

> **Mission:** Make vLLM Semantic Router production-grade across supported
> organizations, environments, and hardware.

![Identity, access, quotas, production lifecycle controls, observability, and supported environments form one production platform](/img/blog/vllm/2026-08-24-workgroups-invitation/workgroups/enterprise-environment.svg)

### Why this matters

Production users need clear answers to practical questions: Who can call each
model? How much can they use? What changed? Is the system healthy? Can a model,
recipe, or Router upgrade be rolled out and reversed safely? The answers must
remain consistent across deployment environments.

### Scope

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

### Non-scope

This Workgroup does not promise a public hosted-service SLA, expose private
infrastructure or credentials, define model quality, own evaluation standards,
or implement networking protocols. It supplies reusable open-source production
capabilities rather than publishing private product plans.

### Epic directions

- [Build multi-tenant inference access, quotas, and usage controls](https://github.com/vllm-project/semantic-router/issues/2960)
- [Build versioned configuration activation and rollback](https://github.com/vllm-project/semantic-router/issues/2326)
- [Establish production observability and supported-environment qualification](https://github.com/vllm-project/semantic-router/issues/2993)

## [Agentic & Context](https://github.com/vllm-project/semantic-router/issues/2987)

> **Mission:** Manage context and safely select, hand off, and compose agent
> backends for long-running workloads.

![A long session is protected and optimized before the Router selects, hands off to, or composes agent backends within explicit limits](/img/blog/vllm/2026-08-24-workgroups-invitation/workgroups/agentic-context.svg)

### Why this matters

Long conversations accumulate messages, retrieved memory, tool output, tokens,
cost, and risk. Important instructions can be lost in that growth, and the best
model or agent may change as the task moves from planning to tool use to final
response. vLLM-SR should select and compose agent backends while managing those
constraints without becoming a general-purpose agent framework.

### Scope

- Context compression, message and tool-output pruning, retrieval and memory
  selection, and prompt restructuring.
- Explicit protection for system and developer instructions, authorization,
  safety constraints, tool contracts, and task-critical user intent.
- Session budgets, state boundaries, retention, recovery, and graceful
  degradation when context or memory capabilities are unavailable.
- Typed agent identity, capability, tool, state, trust, and lifecycle contracts
  for selection, fallback, and handoff.
- Bounded agent composition and multi-agent collaboration with explicit limits
  on participants, depth, turns, tokens, time, cost, authority, and failures.
- Requirements and safety constraints for switching models or bounded
  workflows as a session evolves.
- Evaluation of selection quality, collaboration gain, handoff loss,
  instruction retention, task quality, token reduction, latency, cost, safety,
  and critical-information loss.

### Non-scope

This Workgroup does not build an unrestricted agent orchestrator, own all MoM
selection algorithms, transport KV caches between models, automate CLI
installation, or allow silent lossy transformation and unbounded online
training.

### Epic directions

- [Optimize context for long-session and agentic workloads](https://github.com/vllm-project/semantic-router/issues/2984)
- [Enable agent-based routing and multi-agent composition](https://github.com/vllm-project/semantic-router/issues/2994)
- [Develop safe model and workflow switching for long-running agents](https://github.com/vllm-project/semantic-router/issues/2973)
- [Define a trusted gateway context envelope for agent memory, tools, and budgets](https://github.com/vllm-project/semantic-router/issues/2546)

## [Developer Experience & Ecosystem](https://github.com/vllm-project/semantic-router/issues/2970)

> **Mission:** Make vLLM Semantic Router easy to discover, install, configure,
> extend, and operate.

![A developer journey connects discovery, installation, configuration, the first routed request, understanding, sharing, and contribution](/img/blog/vllm/2026-08-24-workgroups-invitation/workgroups/developer-experience-ecosystem.svg)

### Why this matters

Technical depth has little impact when a new user cannot reach a first request
or understand what happened. The project also needs clear extension paths so
contributors, model builders, infrastructure projects, and educators can build
on it without reverse-engineering the repository.

### Scope

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

### Non-scope

This Workgroup does not control repository permissions or promotion, run
marketing events or AMD-internal programs, or redefine algorithms, production
policy, and quality standards owned by other Workgroups.

### Epic directions

- [Build agent-assisted vLLM-SR deployment, recipe, evaluation, and tuning workflows](https://github.com/vllm-project/semantic-router/issues/2977)
- [Reduce time to the first successful routed request](https://github.com/vllm-project/semantic-router/issues/2690)
- [Publish maintained reference recipes and end-to-end tutorials](https://github.com/vllm-project/semantic-router/issues/2334)

## [Evaluation & Quality](https://github.com/vllm-project/semantic-router/issues/2969)

> **Mission:** Make every supported capability measurable and every change
> verifiable.

![Capabilities from every direction enter a common evaluation contract, layered evaluation stack, regression gates, and published results](/img/blog/vllm/2026-08-24-workgroups-invitation/workgroups/evaluation-quality.svg)

### Why this matters

Claims about a Router Model, MoM recipe, agent-selection policy, runtime
optimization, or deployment are difficult to trust when each uses a different
dataset and reporting method. The project needs shared evaluation contracts and
regression gates, while each technical Workgroup remains accountable for the
quality of what it builds.

### Scope

- Common benchmark structure, dataset provenance, metrics, comparison rules,
  reproducibility, and result publication.
- MoM, recipe, Router Model, agent selection and composition, context, serving,
  platform, and developer-workflow evaluation without replacing each domain's
  objectives.
- Model-card and evaluation-result requirements for published MoM identities.
- CI, E2E, compatibility, security, performance, and operational regression
  coverage across supported modes and environments.
- Shared definitions of done and the evaluation required for milestone
  readiness, upgrade, and rollback.

### Non-scope

This Workgroup does not choose another direction's quality target, accept an
issue, make the final release decision, replace Maintainer review, or own the
model research itself. It defines shared measurement and gates.

### Epic directions

- [Build reproducible decision-level routing evaluation](https://github.com/vllm-project/semantic-router/issues/2333)
- [Establish cross-platform performance regression coverage](https://github.com/vllm-project/semantic-router/issues/1510)
- [Expand end-to-end quality coverage across Router, Dashboard, and deployments](https://github.com/vllm-project/semantic-router/issues/1519)

## How Workgroups Operate

> A Workgroup is a technical home, not a permission level.

Each Workgroup owns one durable technical Direction, its charter, and the
bounded Epics beneath it. It connects contributors, helps triage proposals, and
recommends when work is ready to accept. The Open Source Team retains final
acceptance, review, merge, role, and release authority.

| Workgroups | Open Source Team |
| --- | --- |
| Direction, boundaries, Epics, and contributor focus | Project governance and repository permissions |
| Triage and acceptance recommendations | Final acceptance, merge, and release authority |
| Lead and Member roles | Maintainer, Committer, and Contributor roles |

`Direction → Workgroup → domain → bounded Epic → accepted issue`

**Lead.** Every active Workgroup has at least one Lead and may have several. A
Lead is a Committer or Maintainer, or a Contributor with a merged commit and an
active Committer/Maintainer Sponsor. Leads maintain the charter and Epic map,
coordinate triage, and ensure accepted initiatives have delivery owners.

**Member.** A Member has at least one merged repository commit and continues to
build in the Direction. Anyone may collaborate before meeting the roster
requirement, and people may join more than one Workgroup.

To join, open the linked charter and comment with your requested role, a merged
contribution, your focus and availability, and your Sponsor when applying as a
sponsored Lead. Confirmed Leads and Members appear on the public
[Workgroups map](/community/work-groups); these roles provide visibility and
responsibility, not additional repository permissions.

## Proposing a New Workgroup

Create a Workgroup only for a durable technical problem that spans releases,
contains multiple bounded Epics, and cannot fit an existing charter. A single
feature, algorithm, collaboration, integration, event, or milestone remains an
Epic inside an existing Workgroup.

A proposal must define the problem, goal, scope, non-scope, interfaces, initial
Epic map, why existing charters cannot own it, and at least one eligible Lead.
After Maintainer acceptance, the project creates its charter and owner label,
opens Lead and Member self-nomination, and adds it to the website.

## Come Build With Us

Start with the [Workgroups map](/community/work-groups), choose a charter, and
introduce yourself. Join an Epic, shape a contributor-ready issue, or propose a
focused piece of work.

Find your focus and passion. Build with the community. Grow together.
