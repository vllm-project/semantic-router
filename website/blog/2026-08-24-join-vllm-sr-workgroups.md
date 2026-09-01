---
slug: "join-vllm-sr-workgroups"
title: "Find Your Focus: How to Join and Work Together"
description: "Choose a vLLM Semantic Router Workgroup, claim a useful task, grow into a Member or Lead, and make every contribution visible."
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
2. **Enterprise & Environment** secures management surfaces and owns lifecycle,
   capacity, and deployment policy.
3. **Router Models & Inference Runtime** produces the signals used to route.
4. **MoM & Routing** chooses models and multi-model strategies.
5. **Agentic & Context** manages bounded context, memory, and session continuity.
6. **Data Plane & Networking** executes the chosen path.
7. **Evaluation & Quality** measures results and catches regressions.

They form one system, separated by responsibility rather than isolated code
ownership. Every Epic in the live views below has one owning Workgroup.
Dependencies on other groups are recorded as shared interfaces in the linked
charter.

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

[View all current MoM & Routing Epics](https://github.com/vllm-project/semantic-router/issues?q=is%3Aissue+is%3Aopen+label%3Aepic+label%3Awg%2Fmom-routing)

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
MoM pool, own generic gateway forwarding, secure management surfaces, or rebuild
the tensor engines and GPU schedulers it integrates with.

### Owned Epics

[View all current Router Models & Inference Runtime Epics](https://github.com/vllm-project/semantic-router/issues?q=is%3Aissue+is%3Aopen+label%3Aepic+label%3Awg%2Frouter-models-inference-runtime)

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

This Workgroup executes request-path networking and deployment-provided access
policy, but it does not define management identity and authorization, decide
which hardware is officially supported, train Router Models, or choose the best
MoM recipe.

### Owned Epics

[View all current Data Plane & Networking Epics](https://github.com/vllm-project/semantic-router/issues?q=is%3Aissue+is%3Aopen+label%3Aepic+label%3Awg%2Fdata-plane-networking)

## [Enterprise & Environment](https://github.com/vllm-project/semantic-router/issues/2968)

> **Mission:** Make vLLM Semantic Router production-grade across supported
> environments and hardware.

![Management security, production lifecycle controls, observability, capacity planning, and supported environments form one production platform](/img/blog/vllm/2026-08-24-workgroups-invitation/workgroups/enterprise-environment.svg)

### The problem

Production operators need clear answers to practical questions: Which
management surfaces are protected, and by which provider-backed identity? What
changed? Is the system healthy? Can a model, recipe, or Router upgrade be rolled
out and reversed safely? Which deployment path is maintained, and which
components does it own? The answers must remain consistent across deployment
environments.

### What this Workgroup owns

- Management authentication, provider-backed identity integration, route-bound
  authorization, input and credential boundaries, and durable audit.
- Reliability, scalability, monitoring, diagnostics, and the existing Insights
  and operational surfaces.
- Model, recipe, configuration, and vLLM-SR activation, rollout, and rollback.
- Workload simulation and capacity planning that connect observed traffic,
  routing behavior, serving topology, and calibrated hardware profiles to
  reviewable deployment proposals.
- Stable deployment and lifecycle APIs, maintained reference stacks, and a
  tested support matrix across deployment environments and hardware.

### What it does not own

This Workgroup does not build organization, team, or project administration;
virtual API keys; tenant quotas, token rate limits, budgets, billing, or usage
settlement. It also does not plan routing analytics beyond the existing
Insights and operational surfaces.

It does not promise a public hosted-service SLA, expose private infrastructure
or credentials, define model quality, own evaluation standards, or implement
networking protocols. It supplies reusable open-source production capabilities
rather than publishing private product plans.

### Owned Epics

[View all current Enterprise & Environment Epics](https://github.com/vllm-project/semantic-router/issues?q=is%3Aissue+is%3Aopen+label%3Aepic+label%3Awg%2Fenterprise-environment)

## [Agentic & Context](https://github.com/vllm-project/semantic-router/issues/2987)

> **Mission:** Optimize bounded context, memory, session continuity, and safe
> model or workflow switching for long-running workloads.

![A long session is protected and optimized while the Router preserves bounded context, memory, and continuity](/img/blog/vllm/2026-08-24-workgroups-invitation/workgroups/agentic-context.svg)

### The problem

Long-running work accumulates messages, memory, tool output, cost, and risk.
Important instructions can be lost, and model or workflow changes can break
tool loops or provider state. The Router needs bounded continuity contracts
without becoming a general-purpose agent framework.

### What this Workgroup owns

- Context compression, pruning, memory selection, prompt restructuring, and
  protection of critical instructions.
- Prompt-visible Router Memory with explicit persistence and lifecycle receipts.
- Session budgets, state boundaries, retention, tool-loop continuity, recovery,
  and graceful degradation.
- Safe model or workflow switching as a session evolves.
- Typed task, context-portability, capability, and collaboration receipts that
  external agent runtimes can consume.

### What it does not own

This Workgroup does not select, invoke, host, or compose agent endpoints inside
the Router. It does not build an agent endpoint catalog, unrestricted agent
orchestrator, tool platform, or workflow engine; own general MoM selection;
transport KV caches; or allow silent lossy transformation and unbounded online
training.

### Owned Epics

[View all current Agentic & Context Epics](https://github.com/vllm-project/semantic-router/issues?q=is%3Aissue+is%3Aopen+label%3Aepic+label%3Awg%2Fagentic-context)

## [Developer Experience & Ecosystem](https://github.com/vllm-project/semantic-router/issues/2970)

> **Mission:** Make vLLM Semantic Router easy to adopt, configure, extend,
> diagnose, and contribute to.

![A developer journey connects discovery, installation, configuration, the first routed request, understanding, sharing, and contribution](/img/blog/vllm/2026-08-24-workgroups-invitation/workgroups/developer-experience-ecosystem.svg)

### The problem

Technical depth has little impact when a new user cannot reach a first request
or understand what happened. The project also needs clear extension paths so
contributors, model builders, infrastructure projects, and educators can build
on it without reverse-engineering the repository.

### What this Workgroup owns

- One supported first-run path through the CLI, configuration, recipes, errors,
  and troubleshooting.
- Dashboard configuration and diagnostics built on canonical Router and
  deployment contracts.
- An agent-facing skill for deployment, recipe generation, evaluation, tuning,
  and reviewed operations.
- Documentation, localization, integration guides, Router Model development
  guides, technical content, and contributor entry points.
- Clear extension and contribution paths for models, runtimes, gateways, and
  deployment systems.

### What it does not own

This Workgroup does not control repository permissions or promotion, run
marketing events or AMD-internal programs, or redefine algorithms, production
policy, and quality standards owned by other Workgroups.

### Owned Epics

[View all current Developer Experience & Ecosystem Epics](https://github.com/vllm-project/semantic-router/issues?q=is%3Aissue+is%3Aopen+label%3Aepic+label%3Awg%2Fdeveloper-experience-ecosystem)

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

[View all current Evaluation & Quality Epics](https://github.com/vllm-project/semantic-router/issues?q=is%3Aissue+is%3Aopen+label%3Aepic+label%3Awg%2Fevaluation-quality)

## How to Join and Work Together

> A Workgroup is a technical home, not a permission level.

Each Workgroup owns one durable technical direction and its bounded Epics. It
connects contributors, maintains the charter, and helps prepare work for
acceptance. The Open Source Team retains final acceptance, merge, role, and
release authority.

The normal Member path has four clear steps:

1. **Apply** — choose a Workgroup in the
   [Workgroup join guide](https://github.com/vllm-project/semantic-router/issues/15),
   open its charter, and leave a first-person application comment.
2. **Claim** — choose an eligible Starter task owned by that Workgroup, open the
   target issue, and comment `/assign`. The task counts as assigned only after
   your name appears in the issue's **Assignees** field.
3. **Deliver** — open a focused PR linked to the claimed issue and get a
   substantive contribution merged.
4. **Join the roster** — the Workgroup Operations run collects every eligible
   person into one reviewed roster PR. You become a canonical Member after that
   PR merges.

A contribution belongs to the Workgroup that owns its issue or PR. Work in a
different Workgroup can qualify you there, but it does not complete the
application you made elsewhere. A requested Lead follows the same contribution
path and then receives a separate human role review.

### If you have not applied

Pick the Workgroup whose charter matches the work you want to do, then comment
on that charter in your own words. This compact shape is enough:

```text
I'd like to join this Workgroup.
Role: Member
Background: <relevant experience>
Interested in: <areas you want to help with>
```

You do not need to be on a roster before collaborating. After applying, open
the current
[weekly Workgroup Issues](https://github.com/vllm-project/semantic-router/issues?q=is%3Aissue%20state%3Aopen%20in%3Atitle%20%22%5BCommunity%5D%20Workgroup%20Issues%22)
and continue with a Starter task under the same Workgroup.

### If you have applied but do not know how to get started

Choose an `AVAILABLE` Starter task under your applied-to Workgroup, open the
target issue, comment `/assign`, and verify that GitHub lists you as an
assignee. That assignee state is what makes the application `TASK_ASSIGNED`;
the weekly listing or the `/assign` text by itself does not.

If none of the Starter tasks fit, comment on the weekly Workgroup Issues entry
with exactly:

```text
Need task: wg/<slug>
```

The next operations run will bring that request into the task-replenishment
discussion. It will not assign a task automatically.

Once a substantive PR owned by your applied-to Workgroup merges, your
application becomes `READY_FOR_ROSTER`. The roster PR still needs review and
merge before your Member card becomes canonical.

### If you already joined

Choose an `AVAILABLE` Member task under your Workgroup; Leads may also choose a
bounded Lead task. Use `/assign` on the target issue, keep the issue updated
when scope or progress changes, and prefer one active implementation task at a
time. If nothing fits, use the same `Need task: wg/<slug>` comment on the weekly
Workgroup Issues entry so the next replenishment discussion includes you.

### What each role is responsible for

| Role | How to participate |
| --- | --- |
| Applicant | Choose one Workgroup, apply on its charter, claim a Starter task, and deliver a focused PR. |
| Member | Keep building in the Workgroup direction, claim Member tasks, help with scoped discussion and review, and make current ownership visible. |
| Lead | Maintain the charter and Epic map, keep tasks contributor-ready, coordinate triage and dependencies, and help Members find useful work. |
| Open Source Team | Confirm acceptance and role decisions, review and merge the roster batch, and retain final merge, release, and repository-permission authority. |

Every active Workgroup has at least one Lead and may have several. A Lead is a
Committer or Maintainer, or a Contributor with a merged commit and a Committer
or Maintainer Sponsor. A Member has at least one substantive merged repository
contribution and continues to build in the direction. People may join more than
one Workgroup, and these roles provide visibility and responsibility rather
than additional repository permissions.

## Make Your Work Visible

Contribution is more rewarding when people can see the work, the technical
home behind it, and the responsibility it grows into. The
[Community Console](https://community.vllm-sr.ai) turns public repository
evidence into three shareable cards:

| Workgroup role | Contribution impact | Project team |
| --- | --- | --- |
| [![Example Workgroup Member card for the Enterprise and Environment Workgroup](/img/blog/vllm/2026-08-24-workgroups-invitation/cards/workgroup-member-card.webp)](https://community.vllm-sr.ai/workgroups?wg=wg%2Fenterprise-environment&card=abhinav-m22) | [![Example contribution card showing community activity and rank](/img/blog/vllm/2026-08-24-workgroups-invitation/cards/contribution-card.webp)](https://community.vllm-sr.ai/contributors?range=all&card=abhinav-m22) | [![Example Team card for a repository Maintainer](/img/blog/vllm/2026-08-24-workgroups-invitation/cards/team-card.webp)](https://community.vllm-sr.ai/team?range=all&card=Xunzhuo) |
| Your canonical Lead or Member identity after the roster PR merges. | Your public merged PRs, reviews, commits, discussions, and current rank. | Maintainer, Committer, or Emeritus responsibility after the project's separate promotion process. |

Click a person's avatar or name on the
[Workgroups](https://community.vllm-sr.ai/workgroups),
[Contributors](https://community.vllm-sr.ai/contributors), or
[Team](https://community.vllm-sr.ai/team) page to create the live 3:4 card,
copy it, or download the full-resolution PNG. The examples above are snapshots;
the live cards update with the repository.

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
