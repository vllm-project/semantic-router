---
name: topology-visualization
category: fragment
description: Modifies topology graph APIs, data parsers, layout algorithms, and frontend rendering components that visualize routing topology. Use when changing how topology graphs are fetched, parsed, laid out, or rendered in the dashboard.
---

# Topology Visualization

## Trigger

- The primary skill touches topology graph APIs, parsers, layout, or topology rendering

## Workflow

1. Read change surfaces doc to understand topology visualization dependencies
2. Modify topology graph APIs, parsers, layout, or rendering components
3. Run `make dashboard-check` to validate UI consistency
4. Verify topology renderers and data sources agree on the same graph semantics

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)

## Standard Commands

- `make dashboard-check`

## Acceptance

- Topology renderers and data sources agree on the same graph semantics
