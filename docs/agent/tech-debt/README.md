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

Use file names such as `td-001-example.md`.
Keep the numeric index unique within `docs/agent/tech-debt/`.

## Current Debt Entries

None. The indexed debt register is currently empty.

## Architecture Review Coverage Map

Use this map when turning scale-out architecture findings into debt work. Reuse the matching entry first instead of opening a duplicate item for the same gap.

- Router core
  - resolved runtime bootstrap and shared runtime state registry cleanup: [TD031](td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md)
  - resolved classification subsystem boundary collapse: [TD020](td-020-classification-subsystem-boundary-collapse.md)
  - extproc request and response phase collapse: [TD023](td-023-extproc-request-pipeline-phase-collapse.md), [TD029](td-029-extproc-response-pipeline-phase-collapse.md)
  - resolved restart-sensitive runtime state taxonomy, vector-store restart recovery, and active-config projection semantics: [TD034](td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md)
  - resolved custom Chat Completions struct consolidation: [TD038](td-038-custom-chat-completions-structs.md)
- Dashboard frontend and backend
  - resolved enterprise console browser-session hardening and explicit browser-local chat-history classification: [TD005](td-005-dashboard-enterprise-console-foundations.md)
  - resolved frontend route shell, editor control plane, and large UI containers: [TD030](td-030-dashboard-frontend-config-and-interaction-slice-collapse.md)
  - resolved dashboard backend training and evaluation artifact contract seams: [TD032](td-032-training-evaluation-artifact-contract-drift.md)
  - historical dashboard runtime-control handler split: [TD025](td-025-dashboard-backend-runtime-control-slice-collapse.md)
  - resolved cross-stack console state taxonomy, durability baseline, and active-config projection semantics: [TD034](td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md)
- Cross-stack control planes
  - resolved router-kernel versus control-plane contract ownership across CLI, dashboard backend, operator, and shared runtime/config seams: [TD039](td-039-control-plane-contract-ownership-collapse.md)
- Python CLI
  - resolved shared Docker/Kubernetes lifecycle backend surface: [TD004](td-004-python-cli-kubernetes-workflow-separation.md)
  - resolved dev integration environment ownership and shared-suite topology across Docker, Kind, and CI: [TD037](td-037-dev-integration-env-ownership-and-shared-suite-topology.md)
  - prior CLI config-contract consolidation work: [TD022](td-022-cli-config-contract-boundary-collapse.md)
  - resolved workspace-backed local state taxonomy, vector-store recovery seam, and active-config projection baseline: [TD034](td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md)
- Test architecture and CI
  - resolved default dev integration flow, environment ownership, and PR matrix reduction for shared dashboard/core-routing coverage: [TD037](td-037-dev-integration-env-ownership-and-shared-suite-topology.md)
- Training src
  - resolved training and evaluation artifact contract drift across dashboard, runtime, and scripts: [TD032](td-032-training-evaluation-artifact-contract-drift.md)
- Candle binding and ONNX binding
  - resolved backend capability, parity, and lifecycle contract gap: [TD033](td-033-native-binding-runtime-parity-and-lifecycle-gap.md)
- Shared config, schema, contracts, utilities, and tooling
  - resolved contract-first control-plane boundary between router internals and external control surfaces: [TD039](td-039-control-plane-contract-ownership-collapse.md)
  - resolved shared Milvus lifecycle seam across runtime stores: [TD021](td-021-milvus-adapter-duplication-across-runtime-stores.md)
  - resolved dashboard/DSL weak-typing cleanup across editor, setup, and emit helpers: [TD015](td-015-weakly-typed-config-and-dsl-contracts.md)
  - resolved operator-side contract ownership and canonical translation boundaries: [TD028](td-028-operator-config-contract-boundary-collapse.md)
  - resolved projection-partition fallback and centroid-validation contract work: [TD035](td-035-signal-group-default-coverage-contract-gap.md)
  - resolved paired-source contract for `DECISION_TREE` / `IF ELSE` authoring versus canonical runtime config: [TD036](td-036-decision-tree-authoring-roundtrip-gap.md)
  - resolved state taxonomy, durability ownership baseline, and restart-aware telemetry semantics across router/dashboard/CLI: [TD034](td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md)
  - resolved predecessor work for repo-wide canonical contract consolidation: [TD001](td-001-config-surface-fragmentation.md), [TD026](td-026-go-config-contract-boundary-collapse.md)

## Closed / Historical Debt Entries

- [TD003 Package Topology, Naming, and Hotspot Layout Debt](td-003-package-topology-hotspot-layout-debt.md)
- [TD005 Dashboard Lacks Enterprise Console Foundations](td-005-dashboard-enterprise-console-foundations.md)
- [TD001 Config Surface Fragmentation Across Router, CLI, K8s, and Dashboard](td-001-config-surface-fragmentation.md)
- [TD002 Config Portability Gap Between Local Docker and Kubernetes Deployments](td-002-config-portability-gap-local-vs-k8s.md)
- [TD004 Python CLI and Kubernetes Workflow Separation](td-004-python-cli-kubernetes-workflow-separation.md)
- [TD006 Structural Rule Target Still Exceeds Reality in Key Legacy Hotspots](td-006-structural-rule-target-vs-legacy-hotspots.md)
- [TD007 End-to-End and Integration Test Surfaces Are Split Across Parallel Frameworks](td-007-e2e-integration-surfaces-split-across-frameworks.md)
- [TD008 E2E Profile Matrix Repeats Shared Router Testcases Without Clear Coverage Ownership](td-008-e2e-profile-matrix-shared-router-coverage-ownership.md)
- [TD009 E2E Profile Inventory, Naming, and Documentation Have Drifted Out of Sync](td-009-e2e-profile-inventory-and-documentation-drift.md)
- [TD010 E2E Framework Extension Paths Still Rely on Script-Style Stack Composition and Low-Level Test Fixtures](td-010-e2e-framework-extension-paths.md)
- [TD011 API Server Runtime State Is Split Between a Live Service Handle and a Stale Config Snapshot](td-011-apiserver-runtime-state-split.md)
- [TD012 Canonical v0.3 Routing Contract Still Lacks a LoRA Catalog Surface](td-012-canonical-lora-catalog-gap.md)
- [TD013 Legacy IntelligentPool and IntelligentRoute Controller Bypasses Canonical v0.3 Config](td-013-legacy-k8s-controller-bypasses-canonical-v0-3.md)
- [TD014 Candle Binding Crate-Wide Clippy Gate Blocks Diff-Scoped Validation](td-014-candle-binding-crate-wide-clippy-gate.md)
- [TD015 Weak Typing Still Leaks Through Dashboard Editor Models and DSL Serialization Helpers](td-015-weakly-typed-config-and-dsl-contracts.md)
- [TD016 Fleet Sim Subtree Still Diverges from the Shared Ruff Contract](td-016-fleet-sim-shared-ruff-contract-gap.md)
- [TD017 Fleet Sim Migration Still Depends on Relaxed Structure Gates](td-017-fleet-sim-structure-gate-migration-gap.md)
- [TD018 Skill Surface Taxonomy Has Drifted Away from Active Module Boundaries](td-018-skill-surface-coverage-drift.md)
- [TD019 Behavior Contract E2E Tests Still Encode Report-Only Thresholds](td-019-e2e-behavior-contract-thresholds.md)
- [TD020 Classification Subsystem Boundaries Have Collapsed Into Hotspot Orchestrators](td-020-classification-subsystem-boundary-collapse.md)
- [TD021 Milvus Lifecycle Logic Is Duplicated Across Runtime Stores](td-021-milvus-adapter-duplication-across-runtime-stores.md)
- [TD022 CLI Config Contract Knowledge Is Collapsed Across Schema, Migration, and Validation Hotspots](td-022-cli-config-contract-boundary-collapse.md)
- [TD023 Extproc Request Pipeline Phases Have Collapsed Across Request Filters](td-023-extproc-request-pipeline-phase-collapse.md)
- [TD024 OpenClaw Feature Slice Still Collapses Page, Transport, and Proxy Control Boundaries](td-024-openclaw-feature-slice-boundary-collapse.md)
- [TD025 Dashboard Backend Runtime-Control Slice Still Collapses Handler Transport, Config Persistence, and Status Collection](td-025-dashboard-backend-runtime-control-slice-collapse.md)
- [TD026 Go Router Config Contract Knowledge Still Collapses Across Schema Families, Canonical Conversion, and Validation Hotspots](td-026-go-config-contract-boundary-collapse.md)
- [TD027 Fleet Sim Optimizer and Public Surface Boundaries Still Collapse Analytical Sizing, DES Verification, and Export Policy](td-027-fleet-sim-optimizer-and-public-surface-boundary-collapse.md)
- [TD028 Operator Config Contract Still Collapses Across CRD Schema, Webhook Validation, Canonical Translation, and Sample Fixtures](td-028-operator-config-contract-boundary-collapse.md)
- [TD029 Extproc Response Pipeline Phases Still Collapse Normalization, Streaming, Replay, and Response-Side Warnings](td-029-extproc-response-pipeline-phase-collapse.md)
- [TD030 Dashboard Frontend Config and Interaction Surfaces Still Collapse Route Shell, Page Orchestration, and Large UI Containers](td-030-dashboard-frontend-config-and-interaction-slice-collapse.md)
- [TD031 Router Runtime Bootstrap and Shared Service Registry Still Depend on Process-Wide Globals](td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md)
- [TD032 Training and Evaluation Artifact Contracts Still Drift Across Dashboard, Runtime, and Scripts](td-032-training-evaluation-artifact-contract-drift.md)
- [TD033 Native Binding Runtime Parity and Lifecycle Still Diverge Across Candle and ONNX Backends](td-033-native-binding-runtime-parity-and-lifecycle-gap.md)
- [TD034 Runtime and Dashboard State Surfaces Still Lack a Coherent Durability, Recovery, and Telemetry Contract](td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md)
- [TD035 Projection Partition Default Coverage Contract Is No Longer Declarative Only](td-035-signal-group-default-coverage-contract-gap.md)
- [TD036 Decision Tree Authoring Cannot Round-Trip Through Runtime Config](td-036-decision-tree-authoring-roundtrip-gap.md)
- [TD037 Dev Integration Environment Ownership and Shared-Suite Topology Still Diverge Across CLI, Kind, and CI](td-037-dev-integration-env-ownership-and-shared-suite-topology.md)
- [TD038 Custom Chat Completions Structs Duplicate Official OpenAI SDK Types](td-038-custom-chat-completions-structs.md)
- [TD039 Control Planes Still Depend on Router Runtime Internals Instead of Contract-Owned Seams](td-039-control-plane-contract-ownership-collapse.md)
