# Deck Brief

## Topic

`Deep Dive into vLLM Semantic Router`

## Mode

Expanded persisted long-form route based on the earlier `ppt-as-code` quick
draft.

Reason:

- the user now wants a `40` to `60` minute technical talk
- the deck needs deeper structure, not only visual polish
- the repo already contains enough source material to build a reference-heavy
  presentation without external web research
- the output should feel like a product-and-architecture deep dive, not a short
  intro walkthrough

## Audience

- contributors new to the repository
- platform or infra engineers evaluating multi-model routing
- research engineers who want to connect routing ideas to runtime and product surfaces
- technical leads who need to explain why this repo spans runtime, control
  plane, UI, and deployment concerns in one place

## Delivery Context

- primary talk: `45` to `55` minutes
- optional Q&A / appendix: `5` to `10` minutes
- suitable for screen share or async browser viewing
- includes a clean live-demo bridge at the end instead of requiring a long live
  build sequence

## Artifact Bundle

- main deck: `index.html`
- backup deck: `appendix.html`
- primary script: `deck_script.md`
- expanded presenter notes: `speaker_notes_long.md`
- evidence register: `source_map.md`
- demo execution guide: `demo_runbook.md`

## Core Message

vLLM Semantic Router is not just a request router.
It is a control plane for Mixture-of-Models that combines:

- signal extraction
- projection-based coordination
- policy decisions
- post-match algorithm policy
- request and response plugins
- operational surfaces such as CLI, dashboard, observability, fleet-sim,
  operator, and AMD deployment profiles
- contributor-facing repo structure and harness rules that keep those surfaces
  aligned

## Visual Lock

- white or warm-white background
- black and charcoal typography
- minimal page furniture
- thin rules, quiet borders, generous whitespace
- mono labels for commands and repo paths
- no stock photography
- diagrams and structure do the visual work

## Narrative Acts

1. Why this project exists and what problem shape it addresses
2. How the routing system works internally
3. How the system is operated across local, dashboard, CLI, Kubernetes, and AMD
4. Why the repository shape and harness rules matter

## Time Budget

- act 1: `8` to `10` minutes
- act 2: `16` to `20` minutes
- act 3: `14` to `18` minutes
- act 4: `7` to `10` minutes
- demo bridge and closing: `5` minutes

## Planned Slide Count

- main deck: `27` slides
- appendix deck: `9` slides

## Persistence Strategy

Persist artifacts in-repo because the user explicitly asked for a generated deck.

Chosen output folder:

`docs/presentations/deep-dive-vllm-semantic-router/`

## Source Anchors

- `website/docs/intro.md`
- `website/docs/overview/goals.md`
- `website/docs/overview/semantic-router-overview.md`
- `website/docs/overview/collective-intelligence.md`
- `website/docs/tutorials/signal/overview.md`
- `website/docs/tutorials/projection/overview.md`
- `website/docs/tutorials/plugin/overview.md`
- `website/docs/tutorials/algorithm/overview.md`
- `website/docs/installation/configuration.md`
- `website/docs/training/training-overview.md`
- `website/docs/fleet-sim/overview.md`
- `website/docs/fleet-sim/dashboard-integration.md`
- `src/vllm-sr/README.md`
- `dashboard/README.md`
- `deploy/operator/README.md`
- `deploy/amd/README.md`
- `docs/agent/repo-map.md`
- `docs/agent/context-management.md`
- `docs/agent/architecture-guardrails.md`

## Source Drift Note

The deck uses the current maintained count of `16` signal families from the intro and signal tutorial docs.
`website/docs/overview/signal-driven-decisions.md` still contains an older `15` count, so it was treated as stale for that specific number.

## Demo Posture

- demo the already-integrated product path instead of rebuilding the stack live
- optimize for one stable story: `serve -> dashboard -> topology -> playground -> balance profile`
- keep appendix and backup screenshots or talking points ready in case the live
  environment is unstable
