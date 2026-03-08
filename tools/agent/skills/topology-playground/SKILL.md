---
name: topology-playground
category: fragment
description: Topology visualization and playground reveal/display details for router metadata.
---

# Topology Playground

## Trigger

- The primary skill touches topology rendering, highlighted paths, or playground reveals

## Required Surfaces

- `topology_visualization`
- `playground_reveal`

## Conditional Surfaces

- `response_headers`

## Stop Conditions

- UI reveal behavior depends on backend metadata that is not available in the current change

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)

## Standard Commands

- `make dashboard-check`
- `make agent-report ENV=cpu CHANGED_FILES="..."`

## Acceptance

- Topology and playground expose the intended user-visible metadata without drift
