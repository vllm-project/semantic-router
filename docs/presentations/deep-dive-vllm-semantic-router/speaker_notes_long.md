# Speaker Notes — Long Form

These notes are intentionally longer than the slide script. Use them for
rehearsal, trimming, or adapting the talk for different audiences.

## Slide 01 — Cover

- Open by saying this is not “just a semantic router implementation.”
- It is a system-level attempt to turn multi-model inference into a control
  plane with explicit policy, product surfaces, and deployment contracts.
- Move from motivation to internals to operations to overall system shape.

## Slide 02 — Agenda

- Set expectation that this is a `45` to `55` minute architecture talk, not a
  quick product tour.
- Explain the four acts so the audience can park questions in the right place:
  problem, internals, operations, and platform shape.
- Say there is a demo bridge at the end so the architecture has something
  concrete to land on.

## Slide 03 — Why Semantic Routing Exists

- Start from the common anti-pattern: one powerful model behind one endpoint.
- Explain the hidden costs of that pattern:
  - overpaying for easy traffic
  - under-serving specialist traffic
  - scattering security or compliance logic
  - making routing behavior opaque
- Make the point that once there are multiple models, the real question becomes
  system policy, not single-model capability.
- Name the four core claims from the intro page explicitly:
  - control plane for LLM routing
  - practical token economy layer
  - governance in the request path
  - research surface that can ship
- Keep the framing anchored in the project’s own claims, not in generic routing
  language.

## Slide 04 — The Project’s Five Big Questions

- Use the five questions from the docs as the design requirements for the
  system.
- Show that each major subsystem answers one of those questions:
  - signals answer missing information
  - projections and decisions answer composition
  - algorithms and model routing answer collaboration
  - plugins answer security and route-local behavior
  - observability and feedback answer learning and improvement
- These questions matter because they translate directly into the value story:
  better routing, lower waste, stronger governance, and a system that can
  evolve.

## Slide 05 — Where the Router Lives

- Explain the significance of `Envoy External Processor`.
- The router is in-band with traffic, but still modular enough to stay outside
  every individual application.
- This placement lets teams add routing, safety, header mutation, cache, and
  replay behavior without embedding those concerns into every client.

## Slide 06 — Request Lifecycle

- Walk slowly through one request:
  - ingress at gateway
  - signal extraction
  - projection coordination
  - decision match
  - algorithm or candidate selection
  - plugin execution
  - backend call
  - response-side handling and observability
- Use one maintained probe so the audience sees a real path instead of only an
  abstract diagram:
  `Verify the claim and answer with citations: why did the Roman Republic collapse?`
  Explain that the balance profile expects this kind of request to trigger
  history or verification evidence, produce `verification_required`, match
  `verified_explainer`, and then dispatch through the expected alias while
  keeping replayability.
- Clarify that some product surfaces observe or edit the config around the path,
  but the runtime path itself stays in the router and gateway boundary.

## Slide 07 — The Routing Stack in One View

- Give the audience a stable vocabulary before going deeper.
- Be explicit about ownership:
  - signals detect
  - projections coordinate
  - decisions declare policy
  - algorithms rank or orchestrate candidates
  - plugins execute route-local behavior
- Give the audience three anti-confusion rules up front:
  - detector output is not route policy
  - route policy is not candidate selection
  - candidate selection is not request or response behavior
- Stress that these are not arbitrary abstractions; they prevent policy from
  collapsing into one unreadable file or one giant config object.

## Slide 08 — Signal Families

- Explain the value of grouping signals by cost and dependency model.
- Heuristic signals are cheap, deterministic, and easy to reason about.
- Learned signals are more expressive and help with domain, difficulty,
  modality, and safety interpretation.
- Mention the doc drift briefly and move on; do not let the talk get stuck on
  documentation inconsistency.

## Slide 09 — Signal Tradeoffs

- Explain why a single classifier is not enough.
- Heuristics are fast and useful for strong lexical, identity, or context
  signals, but they are brittle.
- Learned signals catch paraphrase, softer intent, and safety pressure, but they
  introduce model dependence and latency.
- The point of the architecture is that the router can mix both instead of
  choosing one ideology.

## Slide 10 — Projections

- Explain that projections are the layer many people skip when they first think
  about routing systems.
- Use plain language:
  - partitions choose one winner in a competing lane
  - scores combine evidence
  - mappings turn scores into named facts that decisions can reuse
- Stress why this matters: without projections, numeric policy leaks into every
  decision and becomes impossible to maintain.

## Slide 11 — Decisions

- Explain that decisions stay boolean on purpose.
- They express policy in a reviewable way: “if this domain and this pressure
  band, then this route.”
- Mention that decisions are easier to audit when they reference named signals
  and projection outputs instead of embedding calculations inline.

## Slide 12 — Algorithms

- Many audiences conflate route matching and model selection. Separate them.
- A route can match first, and only then decide among candidate backends.
- Show that the system supports both one-model ranking and multi-model
  orchestration patterns.
- Say clearly that the router is not only about “which domain is
  this query,” but also about “what selection policy should govern this route.”

## Slide 13 — Plugins

- Use plugins to explain why the architecture is more than a classifier graph.
- Route-local behavior matters because not every route needs the same prompt,
  cache, retrieval, or safety controls.
- Explain the architectural benefit:
  global services can exist in `global:`, while per-route behavior remains
  attached to the route that owns it.

## Slide 14 — Collective Intelligence

- Reframe the earlier layers as one bigger idea: system-level intelligence.
- No single component decides everything.
- The system becomes smarter because different detectors, policies, and model
  specialists contribute different pieces of evidence and capability.
- Use this moment to justify the architecture as a whole.

## Slide 15 — Canonical Config Contract

- Explain the five sections and their intent.
- Treat config as a product surface, not as incidental YAML.
- Avoid calling the YAML “just configuration.” It is the portable control-plane
  contract across local runtime, dashboard, Helm, and operator.

## Slide 16 — Authoring and Validation Flow

- Show that authoring is treated seriously:
  - YAML
  - DSL
  - Dashboard
  - fragment catalogs
  - CLI validation and migration
- Make the point that the architecture is not only runtime code; it is also the
  workflow by which people safely author and evolve routing policy.

## Slide 17 — Local Runtime CLI

- After several abstract slides, move into something practical.
- Explain how `vllm-sr serve` becomes the gateway into the local product.
- Mention persistent dashboard state, stack isolation, and the unified command
  surface.
- If the audience includes operators or product engineers, this is where they
  start seeing the system as deployable software, not only research code.

## Slide 18 — Split Local Topology and Observability

- Explain why the local split topology matters: it brings local behavior closer
  to the actual multi-service reality.
- The dashboard, router, and Envoy are separate managed containers on one shared
  network; this keeps service URLs, status collection, and runtime apply flows
  honest.
- Mention embedded Grafana and Jaeger as evidence that observability is not an
  afterthought.

## Slide 19 — Dashboard as a Real Control Plane

- Walk through the six user-facing surfaces.
- Emphasize that the dashboard backend proxies several upstreams under stable
  same-origin routes, which matters for auth, embedding, and usability.
- The topology page is especially important for a talk because it turns policy
  into something the audience can see.

## Slide 20 — Fleet Sim

- Clarify what Fleet Sim does:
  planning, cost comparison, strategy evaluation, what-if analysis.
- Clarify what it does not do:
  it is not the live router path or a hidden autoscaler.
- This explains why a simulator belongs beside the router without collapsing
  runtime and planning into one component.

## Slide 21 — Kubernetes via the Same CLI

- Explain that the product goal is lifecycle continuity: the same verbs should
  work locally and in Kubernetes.
- Mention secret handling to show the CLI owns more than process startup.
- If the audience is infra-heavy, briefly call out that the system wants Docker
  and K8s to feel like two backends behind the same operational story.

## Slide 22 — Operator and Cluster Discovery

- Explain the operator as the cluster-native version of the control-plane
  contract.
- The three discovery modes matter because they show the router is designed to
  sit in front of different serving ecosystems.
- Mention semantic cache backends to show the operator surface is not only about
  starting pods; it also configures runtime integrations.

## Slide 23 — AMD Balance Profile

- Use this as the strongest concrete case study in the talk.
- Explain the one-backend / many-aliases trick clearly:
  the hardware stays fixed, but the router exports semantic tiers to the caller.
- That means the policy surface can remain expressive even when the physical
  serving substrate is simpler than the routing space.

## Slide 24 — AMD Calibration and Decision Design

- Use this section to prove the profile is not just a marketing example.
- The profile has:
  - maintained signals
  - maintained projections
  - explicit decision ordering
  - probe manifests
  - a calibration loop
- Use this moment to show how authoring, evaluation, and deployment connect.

## Slide 25 — Training Stack

- Explain why the system has a training stack at all.
- Some routing surfaces depend on learned assets, and those assets need their
  own lifecycle.
- Mention ModernBERT and the rationale in the docs:
  lower latency, self-hostability, specialization, and deterministic inference
  compared with API-based general LLM routing.

## Slide 26 — Core Design Principles and Seams

- Close the gap between “what the system contains” and “how it stays
  understandable as it grows.”
- Make the capability-placement rule explicit:
  - `signal` extracts facts
  - `decision` combines facts
  - `algorithm` chooses among candidate models
  - `plugin` owns post-decision or post-selection behavior
  - `global` is only for intentionally cross-cutting behavior
- Explain the seam rules in practical terms:
  - one file, one main responsibility
  - deep modules, narrow entrypoints
  - schema, migration, and semantic validation should not collapse into one owner
  - CLI, dashboard, and operator should prefer contract-owned seams over deep runtime internals
- Translate those rules into concrete failure cases the audience will recognize:
  - if detector logic leaks into decisions, policy becomes opaque and hard to audit
  - if schema and validation collapse into one owner, config evolution slows down
  - if dashboard or operator bind directly to runtime internals, product surfaces become fragile across change
- End with the sense that the architecture is intentional rather than
  accidental.

## Slide 27 — Platform Shape and Close

- Explain the platform as a system with several product surfaces:
  router runtime, CLI, dashboard, operator, fleet-sim, training, and shared
  contracts.
- Emphasize that a system with this many surfaces needs explicit boundaries and
  validation.
- Close with the strongest claim:
  what makes this project interesting is not one detector or one YAML shape, but
  the fact that it connects research, runtime, UI, and deployment in one
  maintained system.

## Closing Demo Bridge

- If time allows, segue directly into the demo runbook.
- Keep the demo simple:
  - start or show `vllm-sr serve`
  - open the dashboard
  - show topology
  - use playground
  - map one example request back to the `balance` profile
