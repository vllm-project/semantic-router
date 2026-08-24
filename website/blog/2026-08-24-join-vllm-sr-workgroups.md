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

Build measurable model pools, recipes, and multi-model collaboration. Its major
epics cover portable MoM contracts and model cards, the Router Learning recipe
lifecycle, and cross-model KV-cache reuse ([#2971](https://github.com/vllm-project/semantic-router/issues/2971),
[#2238](https://github.com/vllm-project/semantic-router/issues/2238),
[#2976](https://github.com/vllm-project/semantic-router/issues/2976)).

### [Router Models & Inference Runtime](https://github.com/vllm-project/semantic-router/issues/2966)

Improve the built-in Router Models and make their inference runtime extensible
to the wider model ecosystem. Current epics span the runtime contract, model
post-training and router-native model families, and the self-improvement
flywheel ([#2782](https://github.com/vllm-project/semantic-router/issues/2782),
[#2974](https://github.com/vllm-project/semantic-router/issues/2974),
[#2975](https://github.com/vllm-project/semantic-router/issues/2975)).

### [Data Plane & Networking](https://github.com/vllm-project/semantic-router/issues/2967)

Own the fast, reliable request path. This includes standalone serving, Envoy
ExtProc and gateway integrations, streaming, dispatch, retries, telemetry, and
performance optimization ([#1138](https://github.com/vllm-project/semantic-router/issues/1138),
[#2332](https://github.com/vllm-project/semantic-router/issues/2332)).

### [Enterprise & Environment](https://github.com/vllm-project/semantic-router/issues/2968)

Make the system production-grade across environments and hardware. The scope
includes multi-tenancy, identity, API keys, quotas, audit, observability,
stability, scaling, upgrades, and rollback
([#2960](https://github.com/vllm-project/semantic-router/issues/2960),
[#2326](https://github.com/vllm-project/semantic-router/issues/2326)).

### [Agentic & Context](https://github.com/vllm-project/semantic-router/issues/2987)

Keep long-running agent workloads efficient, state-aware, bounded, and safe.
Its epics cover context compression and selection, session state and budgets,
and dynamic model or workflow switching for long sessions
([#2984](https://github.com/vllm-project/semantic-router/issues/2984),
[#2973](https://github.com/vllm-project/semantic-router/issues/2973)).

### [Developer Experience & Ecosystem](https://github.com/vllm-project/semantic-router/issues/2970)

Make vLLM Semantic Router easier to adopt, configure, deploy, tune, and operate.
This Workgroup connects the CLI, Dashboard, APIs, recipes, agent skills,
documentation, blogs, video tutorials, integrations, and shared use cases
([#2977](https://github.com/vllm-project/semantic-router/issues/2977),
[#2690](https://github.com/vllm-project/semantic-router/issues/2690),
[#2334](https://github.com/vllm-project/semantic-router/issues/2334)).

### [Evaluation & Quality](https://github.com/vllm-project/semantic-router/issues/2969)

Give every direction common evidence and release gates. It owns evaluation
contracts, model cards, benchmarks, reproducibility, CI, E2E, compatibility,
and regression coverage ([#2333](https://github.com/vllm-project/semantic-router/issues/2333),
[#1510](https://github.com/vllm-project/semantic-router/issues/1510),
[#1519](https://github.com/vllm-project/semantic-router/issues/1519)).

## How the Workgroups Fit Together

MoM & Routing defines how models collaborate. Router Models & Inference Runtime
provides the learned intelligence behind those decisions. Data Plane &
Networking executes them online, while Enterprise & Environment makes that path
operable in production. Agentic & Context carries the contract across long
sessions. Developer Experience & Ecosystem makes the whole system accessible.
Evaluation & Quality supplies the shared measurement and gates across all six.

This is a collaboration map, not a collection of silos. Cross-cutting work has
one accountable Workgroup and explicit dependencies on the others.

## From an Idea to Accepted Work

New issues start with `needs-acceptance`. A collaborator with repository write
access uses `/accept` after the outcome is bounded and one Workgroup owns it.
`accepted` means the project wants the outcome; `ready-for-dev` means an
unassigned issue is prepared for a contributor; `in-progress` means someone is
actively delivering it. A milestone is the explicit release commitment.

A Workgroup can have multiple Leads and Members. Leads are Committers or
contributors sponsored by a Maintainer. Members have at least one merged
repository commit. Both are listed publicly so contributors can find the people
building in their area. The Open Source Team remains a separate governance
body.

If a durable problem does not fit an existing charter, the community can
propose a new Workgroup with a clear goal, scope, boundaries, and at least one
eligible Lead. Short-lived projects stay as epics inside an existing Workgroup.

## Come Build With Us

Start with the [Workgroups map](/community/work-groups), open the charter that
matches your interests, and introduce yourself. You can join an existing epic,
help shape a contributor-ready issue, share a use case, or propose a focused
piece of work.

Find your focus and passion. Build with the community. Grow together.

