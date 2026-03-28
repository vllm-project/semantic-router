---
name: fleet-sim-change
category: primary
description: Modifies the fleet simulator package, API service, release wiring, or simulator-owned docs and assets as one maintained subsystem. Use when changing src/fleet-sim, simulator release workflow, or fleet-sim-owned docs and assets under website/.
---

# Fleet Sim Change

## Trigger

- Change the fleet simulator package, API service, CLI entrypoints, or release workflow
- Change simulator-owned docs or static assets that must stay aligned with the maintained subsystem

## Workflow

1. Read the fleet-sim local rules and change-surface contract
2. Modify the simulator package, workflow, or owned docs and assets as one subsystem
3. Run `make agent-report ENV=cpu CHANGED_FILES="..."` to verify routing and validation scope
4. Run the simulator validation path and any affected harness checks
5. Confirm simulator runtime and owned docs or assets still describe the same maintained surface

## Gotchas

- `src/fleet-sim/**` is a first-class subsystem now; do not route it through generic fallback reasoning or treat its docs as repo-generic prose.
- Moves inside `fleet_sim` still require public import and entrypoint stability, even when the visible change looks like a small API-handler edit.

## Must Read

- [src/fleet-sim/AGENTS.md](../../../../src/fleet-sim/AGENTS.md)
- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/repo-map.md](../../../../docs/agent/repo-map.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make vllm-sr-sim-test`
- `make agent-ci-gate CHANGED_FILES="..."`

## Acceptance

- Simulator package, API entrypoints, and owned docs or assets stay aligned
- Simulator release or runtime contract changes update the maintained workflow and validation path
