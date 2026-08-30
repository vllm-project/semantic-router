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

Open source grows when people can see where their work belongs and who they can
build with. vLLM Semantic Router now has seven Workgroups, each responsible for
one durable technical direction.

![Find your focus and passion across seven vLLM Semantic Router Workgroups](/img/blog/vllm/2026-08-24-workgroups-invitation/workgroups-invitation-hero.png)

If you are new to the project, start here: vLLM Semantic Router sits between an
AI application and its model or agent backends. It understands the request,
chooses how it should be handled, executes that route, and measures the result.
The Workgroups divide that system into clear places to contribute.

## One System, Seven Clear Owners

A routed request crosses several responsibilities:

1. **Developer Experience & Ecosystem** provides the CLI, Dashboard, APIs,
   recipes, and learning path.
2. **Enterprise & Environment** applies access, usage, lifecycle, and
   deployment policy.
3. **Router Models & Inference Runtime** produces the signals used to route.
4. **MoM & Routing** chooses models and multi-model strategies.
5. **Agentic & Context** manages context and chooses or composes agents.
6. **Data Plane & Networking** executes the chosen path.
7. **Evaluation & Quality** measures results and catches regressions.

They form one system, separated by responsibility rather than isolated code
ownership. Every Epic below has one owning Workgroup. Dependencies on other
groups are recorded as shared interfaces in the linked charter.

Choose the direction that matches the problem you want to solve. GitHub labels
on each linked Epic are the source of truth for acceptance and delivery status.

## [MoM & Routing](https://github.com/vllm-project/semantic-router/issues/2965)

> **Mission:** Make a pool of models behave like one measurable and improving
> Mixture-of-Models.

![One request enters a versioned recipe and qualified model pool, which can select, cascade, compare, or combine models before returning one measurable response](/img/blog/vllm/2026-08-24-workgroups-invitation/workgroups/mom-routing.svg)

### The problem

A user should be able to call one stable model name without choosing a backend
for every request. A Mixture-of-Models (MoM) needs a qualified model pool and a
versioned recipe that can improve without making behavior unpredictable.

### What this Workgroup owns

- Model pools, model roles, portable recipes, and their versioned lifecycle.
- Model selection and collaboration through fallback, cascade, judging,
  synthesis, and bounded workflows.
- Offline-to-online improvement of recipes and pool members against explicit
  quality, cost, latency, safety, domain, or modality objectives.
- Modality-aware pools, approved reasoning reuse, and safe reuse of compatible
  computation across models.

### What it does not own

This Workgroup does not train the lightweight models that produce routing
signals, build the live network path, decide how conversation history is
compressed, or operate a hosted service.

### Owned Epics

- [Define portable MoM model pools, routing recipes, evaluation, and model cards](https://github.com/vllm-project/semantic-router/issues/2971)
- [Optimize routing recipes through an offline-to-online lifecycle](https://github.com/vllm-project/semantic-router/issues/2238)
- [Optimize model-pool members against Mixture-of-Models objectives](https://github.com/vllm-project/semantic-router/issues/3041)
- [Advance bounded multi-model collaboration algorithms](https://github.com/vllm-project/semantic-router/issues/3037)
- [Enable safe cross-model KV-cache reuse](https://github.com/vllm-project/semantic-router/issues/2976)
- [Enable vLLM-Omni backends and modality-aware MoM pools](https://github.com/vllm-project/semantic-router/issues/3030)
- [Reuse validated reasoning experiences for small-model inference](https://github.com/vllm-project/semantic-router/issues/3031)

## [Router Models & Inference Runtime](https://github.com/vllm-project/semantic-router/issues/2966)

> **Mission:** Build better Router Models and one extensible runtime for
> executing them across the ecosystem.

![A Router Model family improves through a model flywheel while a layered inference runtime executes versioned artifacts and emits typed signals](/img/blog/vllm/2026-08-24-workgroups-invitation/workgroups/router-models-inference-runtime.svg)

### The problem

Routing depends on signals such as intent, complexity, safety, preference, and
expected quality. The models that produce those signals must improve over time,
and new models should not spread engine-specific code throughout the Router.

### What this Workgroup owns

- Improve, calibrate, and release the Router Models built into the project.
- Develop routing-native model families beyond BERT-only designs.
- Build reproducible self-improvement, distillation, and fine-tuning pipelines.
- Provide one versioned execution contract across supported engines and
  hardware, with clear activation, diagnostics, and rollback.

### What it does not own

This Workgroup produces routing intelligence; it does not choose the end-user
MoM pool, own generic gateway forwarding, manage users and quotas, or rebuild
the tensor engines and GPU schedulers it integrates with.

### Owned Epics

- [Build an extensible inference runtime for Router Models](https://github.com/vllm-project/semantic-router/issues/2782)
- [Improve current Router Models and develop routing-native model families](https://github.com/vllm-project/semantic-router/issues/2974)
- [Build a reproducible SLM self-improvement and Router Model fine-tuning pipeline](https://github.com/vllm-project/semantic-router/issues/2975)
- [Harden multimodal and image-routing signal robustness](https://github.com/vllm-project/semantic-router/issues/2347)

## [Data Plane & Networking](https://github.com/vllm-project/semantic-router/issues/2967)

> **Mission:** Execute every live routing decision through a fast, reliable,
> and portable request path.

![Standalone HTTP and Envoy gateway entry modes converge on one shared routing core, backend dispatch path, and response stream](/img/blog/vllm/2026-08-24-workgroups-invitation/workgroups/data-plane-networking.svg)

### The problem

A routing decision has little value if the request path is slow, fragile, or
different in every deployment. Standalone serving and gateway integration need
the same behavior and failure semantics.

### What this Workgroup owns

- Standalone OpenAI-compatible serving and Envoy or gateway integrations.
- Request, response, streaming, dispatch, retry, fallback, error, and telemetry
  behavior.
- Engine-neutral backend connectivity and inference-aware endpoint selection.
- Safe semantic caching, performance optimization, and failure recovery.

### What it does not own

This Workgroup executes access and routing policy but does not define who may
access a model, which quota applies, which hardware is officially supported,
how a Router Model is trained, or which MoM recipe is best.

### Owned Epics

- [Support standalone and gateway-integrated data plane modes](https://github.com/vllm-project/semantic-router/issues/1138)
- [Connect semantic routing to inference-aware backend selection](https://github.com/vllm-project/semantic-router/issues/2332)
- [Make semantic caching safe, measurable, and lifecycle-aware](https://github.com/vllm-project/semantic-router/issues/3036)
- [Optimize data-plane performance, streaming, and failure recovery](https://github.com/vllm-project/semantic-router/issues/2992)

## [Enterprise & Environment](https://github.com/vllm-project/semantic-router/issues/2968)

> **Mission:** Make vLLM Semantic Router production-grade across supported
> organizations, environments, and hardware.

![Identity, access, quotas, production lifecycle controls, observability, and supported environments form one production platform](/img/blog/vllm/2026-08-24-workgroups-invitation/workgroups/enterprise-environment.svg)

### The problem

Production users need clear answers to practical questions: Who can call each
model? How much can they use? What changed? Is the system healthy? Can a model,
recipe, or Router upgrade be rolled out and reversed safely? Which deployment
path is maintained, and which components does it own? The answers must remain
consistent across deployment environments.

### What this Workgroup owns

- Users, organizations, serving identities, model access, API keys, quotas, and
  usage accounting.
- Tenant isolation, audit, reliability, scalability, monitoring, and
  diagnostics.
- Model, recipe, configuration, and vLLM-SR activation, rollout, and rollback.
- Workload simulation and capacity planning that connect observed traffic,
  routing behavior, serving topology, and calibrated hardware profiles to
  reviewable deployment proposals.
- Stable deployment and lifecycle APIs, maintained reference stacks, and a
  tested support matrix across deployment environments and hardware.

### What it does not own

This Workgroup does not promise a public hosted-service SLA, expose private
infrastructure or credentials, define model quality, own evaluation standards,
or implement networking protocols. It supplies reusable open-source production
capabilities rather than publishing private product plans.

### Owned Epics

- [Build multi-tenant inference access, quotas, and usage controls](https://github.com/vllm-project/semantic-router/issues/2960)
- [Build versioned configuration activation and rollback](https://github.com/vllm-project/semantic-router/issues/2326)
- [Define deployment architecture and reference stacks across environments and hardware](https://github.com/vllm-project/semantic-router/issues/3043)
- [Establish production observability and supported-environment qualification](https://github.com/vllm-project/semantic-router/issues/2993)
- [Build workload-driven capacity planning for inference fleets](https://github.com/vllm-project/semantic-router/issues/3091)

## [Agentic & Context](https://github.com/vllm-project/semantic-router/issues/2987)

> **Mission:** Manage context and safely select, hand off, and compose agent
> backends for long-running workloads.

![A long session is protected and optimized before the Router selects, hands off to, or composes agent backends within explicit limits](/img/blog/vllm/2026-08-24-workgroups-invitation/workgroups/agentic-context.svg)

### The problem

Long-running work accumulates messages, memory, tool output, cost, and risk.
Important instructions can be lost, while the best model or agent may change as
the task evolves. The Router must handle those changes without becoming a
general-purpose agent framework.

### What this Workgroup owns

- Context compression, pruning, memory selection, prompt restructuring, and
  protection of critical instructions.
- Session budgets, state boundaries, retention, recovery, and graceful
  degradation.
- Agent selection, fallback, handoff, and bounded multi-agent composition.
- Safe model or workflow switching as a session evolves, with measurable task,
  cost, latency, and safety outcomes.

### What it does not own

This Workgroup does not build an unrestricted agent orchestrator, own all MoM
selection algorithms, transport KV caches between models, automate CLI
installation, or allow silent lossy transformation and unbounded online
training.

### Owned Epics

- [Optimize context for long-session and agentic workloads](https://github.com/vllm-project/semantic-router/issues/2984)
- [Enable agent-based routing and multi-agent composition](https://github.com/vllm-project/semantic-router/issues/2994)
- [Develop safe model and workflow switching for long-running agents](https://github.com/vllm-project/semantic-router/issues/2973)

## [Developer Experience & Ecosystem](https://github.com/vllm-project/semantic-router/issues/2970)

> **Mission:** Make vLLM Semantic Router easy to discover, install, configure,
> extend, and operate.

![A developer journey connects discovery, installation, configuration, the first routed request, understanding, sharing, and contribution](/img/blog/vllm/2026-08-24-workgroups-invitation/workgroups/developer-experience-ecosystem.svg)

### The problem

Technical depth has little impact when a new user cannot reach a first request
or understand what happened. The project also needs clear extension paths so
contributors, model builders, infrastructure projects, and educators can build
on it without reverse-engineering the repository.

### What this Workgroup owns

- Installation, configuration, deployment, tuning, diagnosis, and operation
  through the CLI, Dashboard, and APIs.
- An agent-facing skill for deployment, recipe generation, evaluation, tuning,
  and reviewed operations.
- Documentation, reference recipes, tutorials, model cards, blogs, videos, and
  use-case sharing.
- Clear extension and contribution paths for models, runtimes, gateways, and
  deployment systems.

### What it does not own

This Workgroup does not control repository permissions or promotion, run
marketing events or AMD-internal programs, or redefine algorithms, production
policy, and quality standards owned by other Workgroups.

### Owned Epics

- [Build agent-assisted vLLM-SR deployment, recipe, evaluation, and tuning workflows](https://github.com/vllm-project/semantic-router/issues/2977)
- [Reduce time to the first successful routed request](https://github.com/vllm-project/semantic-router/issues/2690)
- [Publish maintained reference recipes and end-to-end tutorials](https://github.com/vllm-project/semantic-router/issues/2334)

## [Evaluation & Quality](https://github.com/vllm-project/semantic-router/issues/2969)

> **Mission:** Make every supported capability measurable and every change
> verifiable.

![Capabilities from every direction enter a common evaluation contract, layered evaluation stack, regression gates, and published results](/img/blog/vllm/2026-08-24-workgroups-invitation/workgroups/evaluation-quality.svg)

### The problem

Claims about a Router Model, MoM recipe, agent-selection policy, runtime
optimization, or deployment are difficult to trust when each uses a different
dataset and reporting method. The project needs shared evaluation contracts and
regression gates. Each technical Workgroup remains accountable for what it
builds; this Workgroup makes the results comparable.

### What this Workgroup owns

- Common benchmark, data provenance, metric, comparison, reproducibility, and
  publication contracts.
- First-class evaluation of each MoM against standalone models, with a common
  core and objective-specific extensions.
- Shared evaluation for routing, agents, context, serving, platform, and
  developer workflows.
- CI, E2E, compatibility, security, performance, and operational regression
  gates.

### What it does not own

This Workgroup does not choose another direction's quality target, accept an
issue, make the final release decision, replace Maintainer review, or own the
model research itself. It defines shared measurement and gates.

### Owned Epics

- [Build reproducible decision-level routing evaluation](https://github.com/vllm-project/semantic-router/issues/2333)
- [Evaluate each Mixture-of-Models as a first-class model](https://github.com/vllm-project/semantic-router/issues/3038)
- [Establish cross-platform performance regression coverage](https://github.com/vllm-project/semantic-router/issues/1510)
- [Expand end-to-end quality coverage across Router, Dashboard, and deployments](https://github.com/vllm-project/semantic-router/issues/1519)

## How Workgroups Operate

> A Workgroup is a technical home, not a permission level.

Each Workgroup owns one durable technical direction and its bounded Epics. It
connects contributors, maintains the charter, and helps prepare work for
acceptance. The Open Source Team retains final acceptance, merge, role, and
release authority.

| Workgroups | Open Source Team |
| --- | --- |
| Direction, boundaries, Epics, and contributor focus | Project governance and repository permissions |
| Triage and acceptance recommendations | Final acceptance, merge, and release authority |
| Lead and Member roles | Maintainer, Committer, and Contributor roles |

**Lead.** Every active Workgroup has at least one Lead and may have several. A
Lead is a Committer or Maintainer, or a Contributor with a merged commit and a
Committer or Maintainer Sponsor. Leads maintain the charter and Epic map and
coordinate triage.

**Member.** A Member has at least one merged repository commit and continues to
build in the Direction. Anyone may collaborate before meeting the roster
requirement, and people may join more than one Workgroup.

To join, open the charter and comment with your requested role, a merged
contribution, your focus, availability, and Sponsor when required. Confirmed
Leads and Members appear on the public
[Workgroups map](/community/work-groups); these roles provide visibility and
responsibility, not additional repository permissions.

## Proposing a New Workgroup

Create a Workgroup only for a durable technical problem that spans releases,
contains several Epics, and cannot fit an existing charter. A feature,
algorithm, integration, event, or milestone belongs inside an existing group.

A proposal must define the problem, scope, non-scope, shared interfaces, initial
Epic map, why existing groups cannot own it, and at least one eligible Lead.
After Maintainer acceptance, it receives a charter, owner label, public roster,
and Lead or Member self-nomination.

## Come Build With Us

Start with the [Workgroups map](/community/work-groups), choose a charter, and
introduce yourself. Join an Epic, shape a contributor-ready issue, or propose a
focused piece of work.

Find your focus and passion. Build with the community. Grow together.
