# Source Map

This file maps the long-form deck to repository-native sources so the talk can
be updated without drifting away from the current repo.

## Canonical Positioning

- `website/docs/intro.md`
  - research-driven project framing
  - Mixture-of-Models control-plane language
  - maintained signal-family count of `16`
  - core benefit claims: control plane, token economy, governance, and a
    research surface that can ship
- `website/docs/overview/goals.md`
  - the five system questions that motivate the architecture
- `website/docs/overview/semantic-router-overview.md`
  - problem/solution framing
  - signal -> projection -> decision -> plugins + model dispatch pipeline
- `website/docs/overview/collective-intelligence.md`
  - system-level “collective intelligence” framing

## Routing Stack

- `website/docs/tutorials/signal/overview.md`
  - heuristic versus learned split
  - maintained signal families and fragment-tree mapping
- `website/docs/tutorials/projection/overview.md`
  - why `routing.projections` exists
  - `partitions`, `scores`, `mappings`
  - balance recipe as the maintained example
- `website/docs/tutorials/algorithm/overview.md`
  - post-match algorithm layer
  - selection versus looper algorithms
- `website/docs/tutorials/plugin/overview.md`
  - route-local behavior model
- `docs/agent/architecture-guardrails.md`
  - capability placement boundaries that reinforce the routing-stack distinctions

## Config and Authoring

- `website/docs/installation/configuration.md`
  - canonical v0.3 config shape
  - section ownership
  - fragment catalog and validation posture
- `src/vllm-sr/README.md`
  - `validate`, `config migrate`, `config import`
  - primary CLI lifecycle
- `dashboard/README.md`
  - config editor, topology, playground, ML setup, monitoring, and proxy model

## Runtime and Deployment

- `src/vllm-sr/README.md`
  - local runtime lifecycle
  - Kubernetes lifecycle through the same CLI
  - stack name and port offset behavior
- `docs/agent/environments.md`
  - repo-native environment selection
  - local versus AMD versus K8s posture
- `docs/agent/plans/pl-0020-local-runtime-topology-separation.md`
  - split local topology contract
- `deploy/operator/README.md`
  - operator quick start
  - backend discovery types
  - semantic cache backend options

## AMD Case Study

- `deploy/amd/README.md`
  - single ROCm backend + semantic aliases
  - five routing tiers
  - thirteen routing decisions
  - projection overview and calibration loop
- `deploy/recipes/balance.yaml`
  - maintained canonical recipe
- `deploy/recipes/balance.dsl`
  - maintained authoring surface for the same profile
- `deploy/recipes/balance.probes.yaml`
  - maintained executable request examples with expected decisions and aliases

## Fleet Sim and Planning

- `website/docs/fleet-sim/overview.md`
  - simulator scope and non-goals
- `website/docs/fleet-sim/dashboard-integration.md`
  - dashboard proxy integration and local sidecar behavior

## Training and Learned Assets

- `website/docs/training/training-overview.md`
  - ModernBERT-based training overview
  - rationale for specialized classification models
- `docs/agent/repo-map.md`
  - training stack location in the repository

## Design Principles and Seams

- `docs/agent/architecture-guardrails.md`
  - one-file-one-responsibility rule
  - capability-placement rules across signal / decision / algorithm / plugin / global
  - contract-owned seam guidance across CLI, dashboard, and operator
  - separation of schema, migration, validation, request phases, and response phases

## Repo Shape and Harness

- `docs/agent/repo-map.md`
  - subsystem map
  - hotspots and main entry points
- `docs/agent/context-management.md`
  - why the repo uses task-first context packs
- `AGENTS.md`
  - entrypoint and canonical command list

## Slide-to-Source Map

### Slides 01-04

- `website/docs/intro.md`
- `website/docs/overview/goals.md`
- `website/docs/overview/semantic-router-overview.md`

### Slides 05-07

- `website/docs/intro.md`
- `website/docs/overview/semantic-router-overview.md`
- `deploy/local/envoy.yaml`
- `docs/agent/architecture-guardrails.md`
- `deploy/recipes/balance.yaml`
- `deploy/recipes/balance.probes.yaml`

### Slides 08-14

- `website/docs/tutorials/signal/overview.md`
- `website/docs/tutorials/projection/overview.md`
- `website/docs/tutorials/algorithm/overview.md`
- `website/docs/tutorials/plugin/overview.md`
- `website/docs/overview/collective-intelligence.md`

### Slides 15-16

- `website/docs/installation/configuration.md`
- `src/vllm-sr/README.md`
- `dashboard/README.md`

### Slides 17-22

- `src/vllm-sr/README.md`
- `docs/agent/environments.md`
- `docs/agent/plans/pl-0020-local-runtime-topology-separation.md`
- `dashboard/README.md`
- `website/docs/fleet-sim/overview.md`
- `website/docs/fleet-sim/dashboard-integration.md`
- `deploy/operator/README.md`

### Slides 23-24

- `deploy/amd/README.md`
- `deploy/recipes/balance.yaml`
- `deploy/recipes/balance.dsl`

### Slides 25-27

- `website/docs/training/training-overview.md`
- `docs/agent/architecture-guardrails.md`
- `docs/agent/repo-map.md`
- `docs/agent/context-management.md`

## Known Caveats

- Signal-family count drift:
  - use `16` from `website/docs/intro.md` and `website/docs/tutorials/signal/overview.md`
  - do not use the stale `15` count from `website/docs/overview/signal-driven-decisions.md`
- Some architecture docs are higher-level marketing or overview pages.
  - When a claim becomes operational or configuration-specific, prefer the tutorial or README source over the overview page.
