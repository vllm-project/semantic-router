---
name: fleet-sim-runtime
category: fragment
description: Modifies the fleet simulator package, API surface, release workflow, and simulator-owned docs or assets. Use when a primary skill touches src/fleet-sim or the simulator subsystem's owned website/release surfaces.
---

# Fleet Sim Runtime

## Trigger

- The primary skill touches the maintained fleet simulator subtree, release workflow, or owned docs/assets

## Workflow

1. Read the fleet-sim local rules for package and API-surface constraints
2. Modify the simulator package, workflow, or owned docs/assets
3. Run `make vllm-sr-sim-test` to verify the maintained simulator path
4. Confirm simulator code and owned docs/assets still describe the same subsystem

## Must Read

- [src/fleet-sim/AGENTS.md](../../../../src/fleet-sim/AGENTS.md)
- [docs/agent/repo-map.md](../../../../docs/agent/repo-map.md)

## Standard Commands

- `make vllm-sr-sim-test`

## Acceptance

- Simulator package, API entrypoints, and owned docs or assets stay aligned
