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
build with, and how an idea becomes part of the project. vLLM Semantic Router
now has seven direction-based Workgroups to make that path clear.

![Find your focus and passion across seven vLLM Semantic Router Workgroups](/img/blog/vllm/2026-08-24-workgroups-invitation/workgroups-invitation-hero.png)

Each Workgroup owns one durable project direction. Together, they cover the
journey from models and routing policy to online serving, production operations,
developer experience, and measurable quality.

## Seven Directions, One Community

### [MoM & Routing](https://github.com/vllm-project/semantic-router/issues/2965)

- **Motivation:** A Mixture-of-Models should behave like one measurable model,
  not a static list of backends. Its objective, candidate pool, collaboration
  policy, evaluation, and lifecycle must evolve together.
- **Scope:** Model-pool design, portable routing recipes, multi-model
  collaboration, offline-to-online recipe improvement, model and recipe
  promotion or rollback, and cross-model efficiency. Collaboration includes
  selection, fallback, cascades, parallel reasoning, judging, synthesis, and
  bounded workflows rather than one fixed algorithm.
- **Non-scope:** Router Model training, semantic context construction, request
  transport, or hosted-service operations.
- **Epic directions:** Portable MoM, recipe, evaluation, and model-card
  contracts ([#2971](https://github.com/vllm-project/semantic-router/issues/2971));
  Router Learning and recipe optimization
  ([#2238](https://github.com/vllm-project/semantic-router/issues/2238));
  cross-model KV-cache reuse with LMCache
  ([#2976](https://github.com/vllm-project/semantic-router/issues/2976)); and
  session-aware model or workflow switching with Agentic & Context
  ([#2973](https://github.com/vllm-project/semantic-router/issues/2973)).

### [Router Models & Inference Runtime](https://github.com/vllm-project/semantic-router/issues/2966)

- **Motivation:** Routing quality cannot remain tied to one BERT-era model
  design or a collection of model-specific execution paths. The project needs
  Router Models that improve continuously and a runtime that can qualify both
  built-in and ecosystem models consistently.
- **Scope:** Post-training and calibration of current models; pretraining new
  routing-native SLM families; privacy-aware adaptation; and typed inference
  contracts spanning adapters, bindings, artifact identity, capabilities,
  activation, diagnostics, and rollback across supported engines and hardware.
- **Non-scope:** Choosing MoM pools and recipes, generic gateway forwarding,
  account and quota management, or rebuilding tensor engines and GPU
  schedulers.
- **Epic directions:** The extensible routing inference runtime
  ([#2782](https://github.com/vllm-project/semantic-router/issues/2782));
  built-in model post-training and router-native model-family research
  ([#2974](https://github.com/vllm-project/semantic-router/issues/2974)); and
  the SLM self-improvement and fine-tuning flywheel with IBM Research
  ([#2975](https://github.com/vllm-project/semantic-router/issues/2975)).

### [Data Plane & Networking](https://github.com/vllm-project/semantic-router/issues/2967)

- **Motivation:** A routing decision only matters if the online path executes it
  quickly and predictably. Standalone and gateway-integrated deployments must
  share routing behavior instead of becoming separate products.
- **Scope:** Request and response processing, streaming, dispatch, retries,
  fallback, telemetry, backend connectivity, and performance optimization. It
  owns a standalone OpenAI-compatible path without mandatory Envoy, while
  keeping Envoy ExtProc and qualified gateway integrations first-class.
- **Non-scope:** Defining identity or quota policy, certifying environments and
  hardware, training Router Models, defining MoM quality, or replacing backend
  schedulers with synchronous Router logic.
- **Epic directions:** A protocol-neutral data plane and standalone adapter
  while preserving ExtProc
  ([#1138](https://github.com/vllm-project/semantic-router/issues/1138));
  inference-aware backend integration below semantic routing
  ([#2332](https://github.com/vllm-project/semantic-router/issues/2332)); and
  reproducible latency, throughput, streaming, and failure-path optimization.

### [Enterprise & Environment](https://github.com/vllm-project/semantic-router/issues/2968)

- **Motivation:** Production adoption requires more than a working request
  path. Teams need consistent access controls, operational visibility,
  lifecycle safety, and a supportable experience across environments and
  hardware.
- **Scope:** Registration and invitations, organizations and serving
  identities, API-key lifecycle, model grants, token and request quotas, usage
  and audit, stability, scaling, monitoring, diagnostics, upgrade and rollback,
  model or recipe rollout, and validated Docker, Kubernetes, CPU, AMD, and
  NVIDIA support matrices.
- **Non-scope:** Public SLA promises, private infrastructure and credentials,
  MoM or Router Model quality, evaluation standards, or ownership of networking
  protocols and the request path.
- **Epic directions:** Scalable multi-tenant inference access control
  ([#2960](https://github.com/vllm-project/semantic-router/issues/2960));
  versioned management snapshots, activation, and rollback
  ([#2326](https://github.com/vllm-project/semantic-router/issues/2326)); and
  production observability, lifecycle controls, and environment/hardware
  qualification.

### [Agentic & Context](https://github.com/vllm-project/semantic-router/issues/2987)

- **Motivation:** Long-running agents accumulate conversations, memory, tool
  output, cost, and risk. They need model-aware context and session policy, but
  the Router must not become an unbounded agent framework.
- **Scope:** Context compression, message and tool-output pruning, retrieval and
  memory selection, prompt restructuring, protected instruction handling,
  session budgets and recovery, and the requirements for safe model or bounded
  workflow switching as a session evolves.
- **Non-scope:** General-purpose agent orchestration, generic MoM algorithms,
  KV-cache transport, CLI installation automation, silent lossy transforms, or
  unbounded online training and search.
- **Epic directions:** Traceable context optimization
  ([#2984](https://github.com/vllm-project/semantic-router/issues/2984));
  long-session model and workflow switching with Looper research
  ([#2973](https://github.com/vllm-project/semantic-router/issues/2973));
  the gateway context envelope
  ([#2546](https://github.com/vllm-project/semantic-router/issues/2546)); and
  Router Memory stabilization
  ([#2339](https://github.com/vllm-project/semantic-router/issues/2339)).

### [Developer Experience & Ecosystem](https://github.com/vllm-project/semantic-router/issues/2970)

- **Motivation:** Powerful routing capabilities have little impact when users
  cannot install them, reach a first request, understand the result, or operate
  the system confidently. Contributors also need a clear path for extending
  and teaching the project.
- **Scope:** CLI, Dashboard, APIs, configuration and validation workflows,
  installation, deployment, tuning, diagnostics, and operations. It also owns
  the vLLM-SR agent skill, documentation, examples, blogs, video tutorials, use
  case sharing, reference integrations, and ecosystem onboarding.
- **Non-scope:** Open Source Team governance, permissions and promotion,
  marketing events or AMD-internal programs, or redefining the technical
  contracts owned by another Workgroup.
- **Epic directions:** The agent skill for installation, deployment, recipe
  generation, evaluation, tuning, and operations
  ([#2977](https://github.com/vllm-project/semantic-router/issues/2977));
  faster first-success onboarding
  ([#2690](https://github.com/vllm-project/semantic-router/issues/2690)); and
  maintained reference recipes and tutorials
  ([#2334](https://github.com/vllm-project/semantic-router/issues/2334)).

### [Evaluation & Quality](https://github.com/vllm-project/semantic-router/issues/2969)

- **Motivation:** Every direction needs comparable evaluation and dependable
  regression gates. Without shared contracts, model quality, routing quality,
  runtime performance, and release readiness become isolated claims.
- **Scope:** Benchmark structure, dataset provenance, metrics, reproducibility,
  MoM evaluation and model-card publication, decision-level routing evaluation,
  CI, E2E, compatibility, security, performance, and operational regression
  coverage. Domain Workgroups still own their objectives and fixes.
- **Non-scope:** Choosing another Workgroup's quality target, accepting work,
  making release decisions, replacing Maintainer review, or owning model-quality
  research.
- **Epic directions:** The evaluation portion of the portable MoM contract
  ([#2971](https://github.com/vllm-project/semantic-router/issues/2971));
  decision-level routing evaluation
  ([#2333](https://github.com/vllm-project/semantic-router/issues/2333));
  cross-platform performance regression coverage
  ([#1510](https://github.com/vllm-project/semantic-router/issues/1510)); and
  module and Dashboard E2E coverage
  ([#1519](https://github.com/vllm-project/semantic-router/issues/1519)).

## What a Workgroup Does

A Workgroup is the long-lived home for one technical Direction. It keeps a
clear charter, maintains the map of bounded epics under that Direction, helps
triage related issues, recommends when work is sufficiently scoped for
acceptance, and connects contributors with the people already working in the
area. Each accepted initiative should have one accountable Workgroup and one
DRI, even when several Workgroups collaborate.

The planning hierarchy is deliberately simple:

`Direction → Workgroup → domain → bounded epic → accepted issue`

Workgroups do not grant repository permissions, merge authority, release
authority, or project-wide roadmap authority. Those responsibilities remain
with the Open Source Team: Maintainers, Committers, and Contributors.

## Roles and How to Join

Every active Workgroup has at least one Lead and may have multiple equal Leads
and Members.

- **Lead:** An active Committer or Maintainer, or a Contributor with at least
  one merged commit and a named active Committer/Maintainer Sponsor. Leads keep
  the charter and epic map clear, coordinate triage, recommend acceptance,
  identify contributor-ready work, and make sure accepted initiatives have a
  DRI.
- **Member:** A Contributor with at least one merged repository commit. Members
  contribute within the Direction, help shape its work, and may join more than
  one Workgroup.

To nominate yourself, comment on the Workgroup's linked charter with your
GitHub handle, requested role, merged contribution, intended focus, expected
availability, and Sponsor when applying as a sponsored Lead. Confirmed Leads
and Members are listed on the public [Workgroups map](/community/work-groups)
with their avatar and name so new contributors know whom to find.

## Proposing a New Workgroup

Start with a proposal issue when an important, durable project problem cannot
fit an existing charter. The proposal should include:

1. the problem and long-term Direction;
2. why the existing Workgroups cannot own it cleanly;
3. a goal, scope, non-scope, and interfaces with other Workgroups;
4. an initial map of bounded epic directions; and
5. at least one eligible Lead willing to form the group.

Maintainers review the boundary and community need before accepting a new
Workgroup, creating its charter and label, and adding it to the website. A
single feature, release milestone, event, algorithm, or partner collaboration
should remain an epic inside an existing Workgroup rather than create a
permanent organization.

## Come Build With Us

Start with the [Workgroups map](/community/work-groups), open the charter that
matches your interests, and introduce yourself. You can join an existing epic,
help shape a contributor-ready issue, share a use case, or propose a focused
piece of work.

Find your focus and passion. Build with the community. Grow together.
