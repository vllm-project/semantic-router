# Technical Debt Entries

This directory stores the detailed debt records referenced by [../tech-debt-register.md](../tech-debt-register.md). The register is the landing page. The files here are the durable per-item source of truth.

## When to Create or Update a Debt Entry

- Create a new entry when the repo knows about a durable architecture or harness gap that will survive beyond the current change.
- Update an existing entry when the scope, evidence, desired end state, or exit criteria materially changed.
- Keep this inventory aligned with the detailed entry files in this directory.

## What Belongs in a Debt Entry

- one durable unresolved gap per file
- stable ID, scope, and summary
- concrete evidence links
- why the gap matters
- the desired end state
- exit criteria that define when the item can be retired

## What Does Not Belong in a Debt Entry

- branch-local cleanup notes
- one-off bug triage with no durable architectural consequence
- active execution state that belongs in a plan
- durable decisions that belong in an ADR

## Debt Entry Versus Other Governance Files

- `docs/agent/tech-debt-register.md`
  - landing page and policy for technical debt workflow
- `docs/agent/tech-debt/*.md`
  - detailed per-item debt records and the only source of truth for debt metadata
- `docs/agent/adr/*.md`
  - durable harness decisions
- `docs/agent/plans/*.md`
  - active long-horizon execution loops

Rule of thumb:

- if the repo knows the gap but has not retired it, use a debt entry
- if the repo has already decided, use an ADR
- if the repo is still executing a long loop, use a plan

## Debt Entry Template

Every debt entry should include:

- `# TDxxx: <title>`
- `## Status`
- `## Scope`
- `## Summary`
- `## Evidence`
- `## Why It Matters`
- `## Desired End State`
- `## Exit Criteria`

Use file names such as `TD001-example.md`.

## Current Debt Entries

- [TD001 Config Surface Fragmentation Across Router, CLI, K8s, and Dashboard](TD001-config-surface-fragmentation.md)
- [TD002 Config Portability Gap Between Local Docker and Kubernetes Deployments](TD002-config-portability-gap-local-vs-k8s.md)
- [TD003 Package Topology, Naming, and Hotspot Layout Debt](TD003-package-topology-hotspot-layout-debt.md)
- [TD004 Python CLI and Kubernetes Workflow Separation](TD004-python-cli-kubernetes-workflow-separation.md)
- [TD005 Dashboard Lacks Enterprise Console Foundations](TD005-dashboard-enterprise-console-foundations.md)
- [TD006 Structural Rule Target Still Exceeds Reality in Key Legacy Hotspots](TD006-structural-rule-target-vs-legacy-hotspots.md)
- [TD007 End-to-End and Integration Test Surfaces Are Split Across Parallel Frameworks](TD007-e2e-integration-surfaces-split-across-frameworks.md)
- [TD008 E2E Profile Matrix Repeats Shared Router Testcases Without Clear Coverage Ownership](TD008-e2e-profile-matrix-shared-router-coverage-ownership.md)
- [TD009 E2E Profile Inventory, Naming, and Documentation Have Drifted Out of Sync](TD009-e2e-profile-inventory-and-documentation-drift.md)
