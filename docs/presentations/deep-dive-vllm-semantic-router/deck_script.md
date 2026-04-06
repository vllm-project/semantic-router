# Deck Script

## Slide 01 — Cover

### Time

`1 min`

### On-slide thesis

`Deep Dive into vLLM Semantic Router`

`从 Signal-Driven Routing 到 Mixture-of-Models Control Plane`

### Key beats

- Frame the system as more than a router binary.
- Set scope early: runtime, control plane, deployment, and operational contracts.
- Set expectation that this is a system deep dive, not a feature list.

### Source anchors

- `website/docs/intro.md`
- `docs/agent/repo-map.md`

## Slide 02 — Agenda

### Time

`2 min`

### On-slide thesis

The talk answers four questions:

1. Why does this project exist?
2. How does the routing pipeline work?
3. How is it operated across local, dashboard, Kubernetes, and AMD?
4. Why is the system shaped this way?

### Key beats

- Preview the four acts and set expectations for a long-form technical talk.
- Signal that the demo bridge appears at the end, so the early sections can focus on system understanding.

### Source anchors

- `deck_brief.md`
- `docs/agent/repo-map.md`

## Slide 03 — Why Semantic Routing Exists

### Time

`2 min`

### On-slide thesis

Move from:

- one-model-fits-all inference
- app-local `if/else` routing
- fragmented governance logic

To:

- explicit policy
- observable decisions
- spend-aware model selection

### Key beats

- Explain why this is a system problem, not only a model problem.
- Tie routing to token economy, safety, and operability.
- Make the four core claims explicit:
  - control plane for LLMRouting
  - practical token economy layer
  - governance in the request path
  - research surface that can ship

### Source anchors

- `website/docs/intro.md`
- `website/docs/overview/semantic-router-overview.md`

## Slide 04 — The Project’s Five Big Questions

### Time

`3 min`

### On-slide thesis

The system explicitly asks:

- how to capture missing signals
- how to combine them
- how models collaborate
- how to secure the serving stack
- how to learn from interaction

### Key beats

- Use these questions as the governing lens for the rest of the talk.
- Show that the product surfaces exist to answer those questions, not as unrelated features.
- Explain that those questions compress into a small set of product claims the
  audience should remember after the talk.

### Source anchors

- `website/docs/overview/goals.md`
- `website/docs/intro.md`

## Slide 05 — Where the Router Lives

### Time

`2 min`

### On-slide thesis

`Client -> Envoy or Gateway -> Semantic Router (ext_proc) -> Model Backends`

### Key beats

- Explain what `ext_proc` buys architecturally: a shared insertion point in the live request path.
- Emphasize that model backends stay independent; the control plane sits in front.

### Source anchors

- `website/docs/intro.md`
- `deploy/local/envoy.yaml`

## Slide 06 — Request Lifecycle

### Time

`3 min`

### On-slide thesis

A single request moves through:

1. ingress
2. signal extraction
3. projection coordination
4. route decision
5. plugin processing
6. backend dispatch
7. response-side controls and observability

### Key beats

- Explain the difference between detector phases and policy phases.
- Ground the path in one maintained probe from the balance profile:
  `Verify the claim and answer with citations: why did the Roman Republic collapse?`
  Use it to show history or verification signals, the `verification_required`
  projection, the `verified_explainer` decision, the alias expectation, and the
  replay plugin.
- Mention that the dashboard and observability surfaces sit around this path rather than replacing it.

### Source anchors

- `website/docs/overview/semantic-router-overview.md`
- `dashboard/README.md`
- `deploy/recipes/balance.yaml`
- `deploy/recipes/balance.probes.yaml`

## Slide 07 — The Routing Stack in One View

### Time

`2 min`

### On-slide thesis

Core mental model:

- Signals extract facts
- Projections coordinate facts
- Decisions choose the route
- Algorithms choose among route candidates
- Plugins apply route-local behavior
- Dispatch sends the request

### Key beats

- Give the audience a stable vocabulary for the next ten slides.
- Make the ownership boundaries explicit.
- Name the three distinctions that make the architecture easier to reason about:
  detector output is not route policy; route policy is not candidate selection;
  candidate selection is not request or response behavior.

### Source anchors

- `website/docs/intro.md`
- `website/docs/tutorials/signal/overview.md`
- `website/docs/tutorials/projection/overview.md`
- `website/docs/tutorials/algorithm/overview.md`
- `website/docs/tutorials/plugin/overview.md`
- `docs/agent/architecture-guardrails.md`

## Slide 08 — Signal Families

### Time

`3 min`

### On-slide thesis

The maintained signal catalog is split into:

- `5` heuristic families
- `11` learned families
- total maintained count: `16`

### Key beats

- Explain why the split matters: cost model, dependency model, and routing semantics.
- Call out the current doc drift on the old `15` count only as a brief caveat.

### Source anchors

- `website/docs/intro.md`
- `website/docs/tutorials/signal/overview.md`

## Slide 09 — Signal Tradeoffs

### Time

`3 min`

### On-slide thesis

No single signal is enough.

Heuristic signals are cheap and deterministic.
Learned signals are richer but slower and more model-dependent.

### Key beats

- Walk through latency/semantic richness tradeoffs.
- Show why the architecture needs more than “one classifier says math.”

### Source anchors

- `website/docs/overview/signal-driven-decisions.md`
- `website/docs/tutorials/signal/overview.md`
- `website/docs/overview/collective-intelligence.md`

## Slide 10 — Projections: The Coordination Layer

### Time

`3 min`

### On-slide thesis

`routing.projections` exists so decisions do not have to own:

- winner selection across competing signals
- weighted aggregation logic
- threshold policy copied across routes

### Key beats

- Explain `partitions`, `scores`, and `mappings`.
- Use `balance_domain_partition` and `difficulty_band` as memorable examples.

### Source anchors

- `website/docs/tutorials/projection/overview.md`
- `deploy/amd/README.md`

## Slide 11 — Decisions: Boolean Policy, Not Detector Logic

### Time

`2 min`

### On-slide thesis

Decisions combine raw signals and named projection outputs into route policy.

### Key beats

- Explain why boolean route rules stay readable when detection and coordination are separated.
- Use the math + reasoning example from the docs.

### Source anchors

- `website/docs/overview/semantic-router-overview.md`

## Slide 12 — Algorithms: Post-Match Model Policy

### Time

`3 min`

### On-slide thesis

Once a decision matches, `decision.algorithm` answers a different question:

`which candidate model or orchestration strategy should win?`

### Key beats

- Separate route eligibility from model selection.
- Contrast selection algorithms with looper algorithms.
- Mention `router_dc`, `automix`, `rl-driven`, `confidence`, `ratings`, `remom`.

### Source anchors

- `website/docs/tutorials/algorithm/overview.md`

## Slide 13 — Plugins: Route-Local Behavior

### Time

`3 min`

### On-slide thesis

Plugins attach behavior to matched routes instead of pushing everything into
`global:`.

### Key beats

- Show three plugin groups: mutation/response, retrieval/memory, safety/generation.
- Explain why this keeps behavior local to the route that needs it.

### Source anchors

- `website/docs/tutorials/plugin/overview.md`
- `src/vllm-sr/README.md`

## Slide 14 — Collective Intelligence

### Time

`2 min`

### On-slide thesis

The “collective intelligence” claim is architectural:

- many signals
- coordinated evidence
- multiple model specialists
- layered plugins

### Key beats

- Explain that the intelligence emerges from system composition, not one magical component.
- Bridge from abstract concept back to the concrete routing stack.

### Source anchors

- `website/docs/overview/collective-intelligence.md`

## Slide 15 — Canonical Config Contract

### Time

`3 min`

### On-slide thesis

One canonical YAML contract spans local CLI, dashboard, Helm, and operator:

`version / listeners / providers / routing / global`

### Key beats

- Explain section ownership.
- Emphasize that routing policy is meant to be a stable product surface.

### Source anchors

- `website/docs/installation/configuration.md`

## Slide 16 — Authoring and Validation Flow

### Time

`3 min`

### On-slide thesis

The authoring story is not just YAML editing.
It includes:

- canonical YAML
- DSL roundtrip
- dashboard config UI
- fragment catalogs
- CLI validation and migration

### Key beats

- Mention `config/signal`, `config/decision`, `config/algorithm`, `config/plugin`.
- Mention `vllm-sr validate`, `config migrate`, and `config import`.
- Explain why fragment-catalog consistency matters for safe policy evolution.

### Source anchors

- `website/docs/installation/configuration.md`
- `src/vllm-sr/README.md`
- `dashboard/README.md`

## Slide 17 — Local Runtime CLI

### Time

`3 min`

### On-slide thesis

The primary local control surface is the `vllm-sr` CLI:

- `serve`
- `status`
- `logs`
- `dashboard`
- `stop`

### Key beats

- Explain the user workflow as a product, not only a developer convenience.
- Mention `.vllm-sr/dashboard-data`, stack name isolation, and port offsets.

### Source anchors

- `src/vllm-sr/README.md`

## Slide 18 — Split Local Topology and Observability

### Time

`3 min`

### On-slide thesis

The local runtime defaults to a split topology with dashboard, router, and Envoy
as separate managed containers on a shared network.

### Key beats

- Explain why this matters for faithful local behavior.
- Mention embedded Grafana, Jaeger, Prometheus, and router metrics.

### Source anchors

- `docs/agent/environments.md`
- `docs/agent/plans/pl-0020-local-runtime-topology-separation.md`
- `dashboard/README.md`

## Slide 19 — Dashboard as a Real Control Plane

### Time

`3 min`

### On-slide thesis

The dashboard is a runtime product surface with:

- Config
- Topology
- Playground
- Monitoring
- ML Setup
- Fleet Sim integration

### Key beats

- Stress that the dashboard is not a thin demo shell.
- Explain why stable same-origin proxying matters.

### Source anchors

- `dashboard/README.md`

## Slide 20 — Fleet Sim: Planning, Not Serving

### Time

`2 min`

### On-slide thesis

Fleet Sim is the maintained planning surface for GPU fleet sizing, strategy
comparison, and what-if analysis.

### Key beats

- Explain what Fleet Sim is and is not.
- Mention the default local sidecar path and dashboard proxy integration.

### Source anchors

- `website/docs/fleet-sim/overview.md`
- `website/docs/fleet-sim/dashboard-integration.md`

## Slide 21 — Kubernetes via the Same CLI

### Time

`3 min`

### On-slide thesis

The same CLI can switch from local Docker to Kubernetes with `--target k8s`.

### Key beats

- Explain why Docker and Kubernetes are presented as backends behind one lifecycle.
- Mention secret handling and common commands.

### Source anchors

- `src/vllm-sr/README.md`
- `docs/agent/environments.md`

## Slide 22 — Operator and Cluster Discovery

### Time

`3 min`

### On-slide thesis

The operator turns cluster deployment into a CRD-driven contract with several
backend discovery modes.

### Key beats

- Walk through `kserve`, `llamastack`, and direct `service` discovery.
- Mention semantic cache backend choices and the distinction between configuration and deployment responsibility.

### Source anchors

- `deploy/operator/README.md`

## Slide 23 — AMD Balance Profile: Architecture in Concrete Form

### Time

`4 min`

### On-slide thesis

The `balance` profile is the cleanest concrete example of the architecture:

- one ROCm backend
- multiple semantic aliases
- five routing tiers
- thirteen decisions

### Key beats

- Show how a single physical backend can present several semantic lanes.
- Emphasize that the router owns the policy, not the backend.

### Source anchors

- `deploy/amd/README.md`
- `deploy/recipes/balance.yaml`

## Slide 24 — AMD Calibration and Decision Design

### Time

`3 min`

### On-slide thesis

The AMD profile is not hand-wavy.
It is tied to:

- maintained signals and projections
- decision ordering
- probe manifests
- calibration loops

### Key beats

- Explain why `balance.probes.yaml` matters.
- Show how calibration connects YAML, DSL, and live router behavior.

### Source anchors

- `deploy/amd/README.md`
- `deploy/recipes/balance.dsl`

## Slide 25 — Training Stack and Model Assets

### Time

`3 min`

### On-slide thesis

The router’s learned surfaces depend on a maintained training stack, not only
hard-coded heuristics.

### Key beats

- Explain the role of specialized classifiers.
- Mention why ModernBERT is used for classification tasks.
- Tie training outputs back into routing decisions and runtime modules.

### Source anchors

- `website/docs/training/training-overview.md`
- `docs/agent/repo-map.md`

## Slide 26 — Core Design Principles and Seams

### Time

`3 min`

### On-slide thesis

The architecture stays understandable because capability is placed through narrow seams:

- `signal` extracts facts
- `decision` combines facts
- `algorithm` chooses among candidates
- `plugin` owns post-decision behavior
- `global` stays intentionally cross-cutting

### Key beats

- Make the architecture guardrails explicit rather than leaving them implicit.
- Explain why the architecture insists on contract-owned seams between router runtime,
  CLI, dashboard, and operator.
- Explain why schema declaration, migration, semantic validation, and transport
  translation are kept on separate owners.
- Translate the guardrails into failure modes:
  when detector logic leaks into decisions, when schema and validation collapse
  into one owner, or when product surfaces bind to runtime internals, the
  system becomes harder to evolve and reason about.

### Source anchors

- `docs/agent/architecture-guardrails.md`
- `website/docs/intro.md`

## Slide 27 — Platform Shape and Close

### Time

`4 min`

### On-slide thesis

The platform needs explicit boundaries and contracts because runtime, config,
UI, deployment, and learned assets evolve together.

### Key beats

- Walk through the main subsystems: router, CLI, dashboard, operator, fleet-sim, training, and shared contracts.
- Explain why a system with this many surfaces needs explicit boundaries and validation.
- Close with the demo bridge:
  - `vllm-sr serve`
  - dashboard
  - topology
  - playground
  - `balance.yaml`

### Source anchors

- `docs/agent/repo-map.md`
- `docs/agent/context-management.md`
- `src/vllm-sr/README.md`
- `dashboard/README.md`
- `deploy/amd/README.md`

## Session Planning

Target runtime: `47` to `58` minutes, plus optional Q&A or appendix material.
