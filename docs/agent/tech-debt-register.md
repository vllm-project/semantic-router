# Technical Debt Register

This document is the summary index for durable gaps between the repository's desired architecture and the current implementation. It is the canonical place to discover tracked debt that should survive beyond one PR, one contributor, or one chat thread.

Detailed records for each item live under [tech-debt/README.md](tech-debt/README.md).

## Why This Exists

- Some architectural gaps are too broad to fix in the same change that discovers them.
- If those gaps stay only in PR text, chat, or memory, agents and contributors will miss them.
- A durable debt register lets the harness distinguish:
  - canonical rules we want to converge toward
  - known implementation debt that has not been retired yet

## Policy

- When current code materially diverges from the desired architecture or harness rules and the gap is not fully closed in the same change, add or update a summary here and the matching detailed entry under `docs/agent/tech-debt/`.
- Use stable IDs (`TD001`, `TD002`, ...) so PRs and follow-up work can point to the same debt item.
- Keep each summary concrete:
  - status
  - scope
  - one-paragraph summary
  - link to the detailed entry
- Keep the detailed entry responsible for evidence, desired end state, and exit criteria.
- Do not use this file for one-off branch tasks or temporary debugging notes.

## Open Debt Items

### TD001 Config Surface Fragmentation Across Router, CLI, K8s, and Dashboard

- Status: open
- Scope: configuration architecture
- Summary: The same conceptual router configuration is represented across Go router config, Python CLI models, dashboard editing UI, and Kubernetes/operator schemas, so common config changes still require synchronized multi-surface edits.
- Entry: [TD001-config-surface-fragmentation.md](tech-debt/TD001-config-surface-fragmentation.md)

### TD002 Config Portability Gap Between Local Docker and Kubernetes Deployments

- Status: open
- Scope: environment and deployment configuration
- Summary: Local Docker startup, repo config examples, and Kubernetes deployment paths do not share one portable config story, so environment changes still depend on special-case files and validation behavior.
- Entry: [TD002-config-portability-gap-local-vs-k8s.md](tech-debt/TD002-config-portability-gap-local-vs-k8s.md)

### TD003 Package Topology, Naming, and Hotspot Layout Debt

- Status: open
- Scope: code organization and file/module structure
- Summary: The codebase still depends on oversized hotspot files and uneven package seams, so the structure rules describe the target architecture more than the common case for several high-risk areas.
- Entry: [TD003-package-topology-hotspot-layout-debt.md](tech-debt/TD003-package-topology-hotspot-layout-debt.md)

### TD004 Python CLI and Kubernetes Workflow Separation

- Status: open
- Scope: environment orchestration and user workflow
- Summary: The Python CLI is centered on local container lifecycle management and does not provide an equally first-class Kubernetes workflow, which forces users onto different control surfaces across environments.
- Entry: [TD004-python-cli-kubernetes-workflow-separation.md](tech-debt/TD004-python-cli-kubernetes-workflow-separation.md)

### TD005 Dashboard Lacks Enterprise Console Foundations

- Status: open
- Scope: dashboard product architecture
- Summary: The dashboard already supports setup, deploy, and readonly flows, but it still lacks the persistent config, authentication, and session foundations expected from an enterprise console.
- Entry: [TD005-dashboard-enterprise-console-foundations.md](tech-debt/TD005-dashboard-enterprise-console-foundations.md)

### TD006 Structural Rule Target Still Exceeds Reality in Key Legacy Hotspots

- Status: open
- Scope: architecture ratchet versus current code
- Summary: The structure rules intentionally ratchet the repo toward smaller modules, but several legacy hotspots still need explicit exceptions, so the common path has not yet caught up with the stated target.
- Entry: [TD006-structural-rule-target-vs-legacy-hotspots.md](tech-debt/TD006-structural-rule-target-vs-legacy-hotspots.md)

### TD007 End-to-End and Integration Test Surfaces Are Split Across Parallel Frameworks

- Status: open
- Scope: test architecture and harness coverage
- Summary: Integration and E2E coverage is split across Go profile-based tests, standalone Python suites, and workflow-driven coverage, which weakens affected-test selection and ownership clarity.
- Entry: [TD007-e2e-integration-surfaces-split-across-frameworks.md](tech-debt/TD007-e2e-integration-surfaces-split-across-frameworks.md)

### TD008 E2E Profile Matrix Repeats Shared Router Testcases Without Clear Coverage Ownership

- Status: open
- Scope: test matrix efficiency and coverage design
- Summary: Many expensive E2E profiles repeatedly run shared router assertions without a clear ownership model for baseline smoke versus profile-specific behavior, so cost and coverage intent drift together.
- Entry: [TD008-e2e-profile-matrix-shared-router-coverage-ownership.md](tech-debt/TD008-e2e-profile-matrix-shared-router-coverage-ownership.md)

### TD009 E2E Profile Inventory, Naming, and Documentation Have Drifted Out of Sync

- Status: open
- Scope: test discoverability and profile inventory
- Summary: The documented E2E profile inventory, runner registration, and harness profile map have drifted apart, making supported-profile discovery and naming less reliable than the repo's governance model intends.
- Entry: [TD009-e2e-profile-inventory-and-documentation-drift.md](tech-debt/TD009-e2e-profile-inventory-and-documentation-drift.md)

## How to Retire Debt

- Close an item only when the underlying architectural gap is materially reduced, not just renamed.
- When a debt item is retired:
  - update the relevant canonical docs and executable rules first
  - update the summary in this register and the matching entry file in the same change
  - mark the item as closed or remove it from both places when appropriate
  - reference the retiring PR or change in the entry if useful
