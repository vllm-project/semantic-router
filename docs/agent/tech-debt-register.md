# Technical Debt Register

This document tracks durable gaps between the repository's desired architecture and the current implementation. It is the canonical place to record debt that should survive beyond one PR, one contributor, or one chat thread.

## Why This Exists

- Some architectural gaps are too broad to fix in the same change that discovers them.
- If those gaps stay only in PR text, chat, or memory, agents and contributors will miss them.
- A durable debt register lets the harness distinguish:
  - canonical rules we want to converge toward
  - known implementation debt that has not been retired yet

## Policy

- When current code materially diverges from the desired architecture or harness rules and the gap is not fully closed in the same change, add or update an entry here.
- Use stable IDs (`TD001`, `TD002`, ...) so PRs and follow-up work can point to the same debt item.
- Keep each item concrete:
  - what is wrong now
  - where the evidence lives
  - why it matters
  - what a good end state looks like
  - what exit criteria would allow the item to be retired
- Do not use this file for one-off branch tasks or temporary debugging notes.

## Open Debt Items

### TD001 Config Surface Fragmentation Across Router, CLI, K8s, and Dashboard

- Status: open
- Scope: configuration architecture
- Evidence:
  - [src/semantic-router/pkg/config/config.go](../../src/semantic-router/pkg/config/config.go)
  - [src/vllm-sr/cli/models.py](../../src/vllm-sr/cli/models.py)
  - [src/vllm-sr/cli/parser.py](../../src/vllm-sr/cli/parser.py)
  - [src/vllm-sr/cli/validator.py](../../src/vllm-sr/cli/validator.py)
  - [src/vllm-sr/cli/merger.py](../../src/vllm-sr/cli/merger.py)
  - [src/semantic-router/pkg/dsl/emitter_yaml.go](../../src/semantic-router/pkg/dsl/emitter_yaml.go)
  - [deploy/operator/api/v1alpha1/semanticrouter_types.go](../../deploy/operator/api/v1alpha1/semanticrouter_types.go)
  - [dashboard/frontend/src/pages/ConfigPage.tsx](../../dashboard/frontend/src/pages/ConfigPage.tsx)
- Why it matters:
  - The same conceptual router configuration is represented in multiple schemas and translated between them.
  - A single config feature can require synchronized edits in Go router config, Python CLI models, merge/translation logic, dashboard editing UI, and Kubernetes CRD paths.
  - This increases drift risk and makes feature delivery slower and less reliable.
- Desired end state:
  - One canonical config contract with thinner adapters for CLI, dashboard, and Kubernetes deployment.
  - Translation layers exist only where representation changes are unavoidable.
- Exit criteria:
  - Adding a config feature no longer requires parallel structural changes across several independent schemas for the common path.
  - Router, CLI, dashboard, and operator paths share a clearer single source of truth for config shape.

### TD002 Config Portability Gap Between Local Docker and Kubernetes Deployments

- Status: open
- Scope: environment and deployment configuration
- Evidence:
  - [src/vllm-sr/cli/templates/config.template.yaml](../../src/vllm-sr/cli/templates/config.template.yaml)
  - [src/vllm-sr/cli/templates/router-defaults.yaml](../../src/vllm-sr/cli/templates/router-defaults.yaml)
  - [config/config.yaml](../../config/config.yaml)
  - [deploy/kubernetes/llmd-base/llmd+public-llm/config.yaml.local](../../deploy/kubernetes/llmd-base/llmd+public-llm/config.yaml.local)
  - [src/semantic-router/pkg/config/validator.go](../../src/semantic-router/pkg/config/validator.go)
- Why it matters:
  - Local Docker startup, repo config examples, and Kubernetes/operator deployment paths do not share one portable config story.
  - The `config/` directory mixes legacy and environment-specific examples that are not consistently reusable across local and Kubernetes flows.
  - Kubernetes mode currently needs special validation and loading behavior instead of looking like the same config model deployed differently.
- Desired end state:
  - A clearer split between canonical portable config, environment overlays, and legacy examples.
  - Local Docker, AMD, and Kubernetes paths can consume the same conceptual config with predictable adapters.
- Exit criteria:
  - The primary local and Kubernetes workflows can start from the same canonical config structure or a formally defined overlay system.
  - Legacy-only examples are either retired or explicitly isolated from the default path.

### TD003 Package Topology, Naming, and Hotspot Layout Debt

- Status: open
- Scope: code organization and file/module structure
- Evidence:
  - [src/semantic-router/pkg/config/config.go](../../src/semantic-router/pkg/config/config.go)
  - [src/semantic-router/pkg/extproc/processor_req_body.go](../../src/semantic-router/pkg/extproc/processor_req_body.go)
  - [src/semantic-router/pkg/extproc/processor_res_body.go](../../src/semantic-router/pkg/extproc/processor_res_body.go)
  - [src/vllm-sr/cli/docker_cli.py](../../src/vllm-sr/cli/docker_cli.py)
  - [dashboard/frontend/src/pages/BuilderPage.tsx](../../dashboard/frontend/src/pages/BuilderPage.tsx)
  - [dashboard/frontend/src/components/ChatComponent.tsx](../../dashboard/frontend/src/components/ChatComponent.tsx)
- Why it matters:
  - The desired structure rules say files should stay narrow and packages should reflect clear seams, but the codebase still contains several oversized hotspots and a pkg layout that is partly too flat and partly too fragmented.
  - Some packages carry only a tiny amount of code while other high-complexity areas are still concentrated in large orchestration files.
  - Naming and package boundaries do not always reflect the current architectural layers.
- Desired end state:
  - Package boundaries reflect real subsystems and runtime seams.
  - Legacy hotspot files continue shrinking until the main orchestration files stop acting as catch-all modules.
- Exit criteria:
  - The highest-risk hotspots have been reduced enough that new work can follow the standard modularity rules without exception-heavy local guidance.
  - Package names and directory structure align more closely with stable subsystem boundaries.

### TD004 Python CLI and Kubernetes Workflow Separation

- Status: open
- Scope: environment orchestration and user workflow
- Evidence:
  - [src/vllm-sr/cli/core.py](../../src/vllm-sr/cli/core.py)
  - [src/vllm-sr/cli/docker_cli.py](../../src/vllm-sr/cli/docker_cli.py)
  - [docs/agent/environments.md](environments.md)
  - [deploy/operator/api/v1alpha1/semanticrouter_types.go](../../deploy/operator/api/v1alpha1/semanticrouter_types.go)
- Why it matters:
  - The Python CLI is strongly oriented around local container lifecycle and does not provide a comparable first-class orchestration path for Kubernetes environments.
  - This creates an environment split where local users and Kubernetes users learn different control surfaces and config flows.
  - It also makes it harder to provide one consistent product story across local dev, cluster deployment, and dashboard operations.
- Desired end state:
  - The CLI and environment management model expose a more consistent experience across local and Kubernetes workflows.
  - Environment differences are treated as deployment backends, not separate product surfaces.
- Exit criteria:
  - Kubernetes deployment and lifecycle management have a coherent path within the shared CLI or a clearly unified orchestration interface.
  - Users do not need to mentally switch between unrelated environment management models for common operations.

### TD005 Dashboard Lacks Enterprise Console Foundations

- Status: open
- Scope: dashboard product architecture
- Evidence:
  - [dashboard/README.md](../../dashboard/README.md)
  - [dashboard/backend/config/config.go](../../dashboard/backend/config/config.go)
  - [dashboard/backend/evaluation/db.go](../../dashboard/backend/evaluation/db.go)
  - [dashboard/backend/router/router.go](../../dashboard/backend/router/router.go)
- Why it matters:
  - The dashboard already provides readonly mode, proxying, setup/deploy flows, and a small evaluation database, but it does not yet provide a unified persistent config store, user login/session management, or stronger enterprise security controls.
  - The README explicitly treats OIDC, RBAC, and stronger proxy/session behavior as future work.
  - This limits the dashboard's role as a real enterprise console.
- Desired end state:
  - Dashboard state and config persistence move toward a clearer control-plane model.
  - Authentication, authorization, and user/session management become first-class capabilities instead of future notes.
- Exit criteria:
  - The dashboard has a coherent persistent storage model for console state and config workflows.
  - Auth, login/session, and user/role controls exist as supported product features rather than roadmap notes.

### TD006 Structural Rule Target Still Exceeds Reality in Key Legacy Hotspots

- Status: open
- Scope: architecture ratchet versus current code
- Evidence:
  - [docs/agent/architecture-guardrails.md](architecture-guardrails.md)
  - [docs/agent/repo-map.md](repo-map.md)
  - [tools/agent/structure-rules.yaml](../../tools/agent/structure-rules.yaml)
- Why it matters:
  - The harness correctly states that large hotspot files are debt, not precedent, but several code areas still depend on hotspot-specific exceptions and ratchets.
  - This is the right governance posture, but it remains a real code/spec gap until the worst hotspots no longer need special handling.
- Desired end state:
  - The global structure rules become the common case rather than something many hotspot directories can only approach gradually.
- Exit criteria:
  - The highest-risk files no longer need special ratchet treatment to stay within the intended modularity envelope.

### TD007 End-to-End and Integration Test Surfaces Are Split Across Parallel Frameworks

- Status: open
- Scope: test architecture and harness coverage
- Evidence:
  - [docs/agent/testing-strategy.md](testing-strategy.md)
  - [docs/agent/playbooks/e2e-selection.md](playbooks/e2e-selection.md)
  - [tools/agent/task-matrix.yaml](../../tools/agent/task-matrix.yaml)
  - [tools/agent/e2e-profile-map.yaml](../../tools/agent/e2e-profile-map.yaml)
  - [e2e/cmd/e2e/main.go](../../e2e/cmd/e2e/main.go)
  - [e2e/testing/run_all_tests.py](../../e2e/testing/run_all_tests.py)
  - [.github/workflows/integration-test-memory.yml](../../.github/workflows/integration-test-memory.yml)
  - [.github/workflows/integration-test-vllm-sr-cli.yml](../../.github/workflows/integration-test-vllm-sr-cli.yml)
- Why it matters:
  - The repository currently maintains multiple durable integration and E2E entrypoints: the Go profile-based harness under `e2e/`, standalone Python suites under `e2e/testing/`, and workflow-driven integration coverage outside the harness selection path.
  - The testing docs tell contributors to use `task-matrix.yaml` and `e2e-profile-map.yaml` as the selection contract, but those files do not model the full set of workflow-driven integration suites.
  - This makes it unclear which suite is canonical for a given behavior change, weakens affected-test selection, and allows some integration or E2E changes to sit outside the normal harness validation story.
- Desired end state:
  - One explicit repository-wide taxonomy for unit, integration, local E2E, and CI-only system tests.
  - Every durable integration or E2E suite is either represented in the harness selection rules or intentionally declared as an exception with documented ownership and invocation.
- Exit criteria:
  - A contributor can change any integration or E2E file and get a deterministic `agent-report` result with named validation commands and affected suites.
  - Workflow-only suites are either absorbed into the canonical harness or documented as bounded exceptions instead of parallel hidden paths.

### TD008 E2E Profile Matrix Repeats Shared Router Testcases Without Clear Coverage Ownership

- Status: open
- Scope: test matrix efficiency and coverage design
- Evidence:
  - [e2e/README.md](../../e2e/README.md)
  - [e2e/profiles/ai-gateway/profile.go](../../e2e/profiles/ai-gateway/profile.go)
  - [e2e/profiles/dynamic-config/profile.go](../../e2e/profiles/dynamic-config/profile.go)
  - [e2e/profiles/istio/profile.go](../../e2e/profiles/istio/profile.go)
  - [e2e/profiles/llm-d/profile.go](../../e2e/profiles/llm-d/profile.go)
  - [e2e/profiles/production-stack/profile.go](../../e2e/profiles/production-stack/profile.go)
  - [e2e/testing/vllm-sr-cli/test_integration.py](../../e2e/testing/vllm-sr-cli/test_integration.py)
- Why it matters:
  - Many heavy profiles rerun the same shared router assertions. Today `chat-completions-request` is wired into nine profiles, the two chat stress cases into six profiles, and `domain-classify` into six profiles.
  - Some profiles explicitly reuse shared router cases rather than asserting profile-specific semantics. `llm-d`, for example, currently reuses shared router testcases and expects llm-d-specific HA or traffic behavior to be covered elsewhere.
  - The CLI integration suite shows the same pattern at a smaller scale: several methods each repeat the full `vllm-sr serve` startup path instead of separating common setup cost from distinct contract checks.
  - Without a durable coverage ownership map, expensive environments spend time re-verifying baseline behavior while unique environment contracts remain under-specified.
- Desired end state:
  - A coverage matrix that distinguishes baseline router smoke, cross-profile contract tests, and environment-specific assertions.
  - Shared behaviors run in one small baseline path or a parameterized matrix, while expensive profiles focus on the semantics that only they can validate.
- Exit criteria:
  - Each profile's `GetTestCases()` list is explainable in terms of unique contract coverage rather than inherited habit.
  - Repeated shared testcases are reduced to a deliberate smoke subset or a single parameterized multi-profile matrix with clear purpose.
  - Environment-specific profiles add assertions for their own semantics instead of mostly replaying generic router cases.

### TD009 E2E Profile Inventory, Naming, and Documentation Have Drifted Out of Sync

- Status: open
- Scope: test discoverability and profile inventory
- Evidence:
  - [e2e/README.md](../../e2e/README.md)
  - [e2e/cmd/e2e/main.go](../../e2e/cmd/e2e/main.go)
  - [tools/agent/e2e-profile-map.yaml](../../tools/agent/e2e-profile-map.yaml)
  - [e2e/profiles/response-api/profile.go](../../e2e/profiles/response-api/profile.go)
  - [e2e/profiles/responseapi/redis_profile.go](../../e2e/profiles/responseapi/redis_profile.go)
- Why it matters:
  - The documented supported profile list in `e2e/README.md` does not match the actual runner inventory in `e2e/cmd/e2e/main.go`, and the harness profile map has a different set again.
  - The same surface uses mixed naming conventions such as `response-api` for a public profile directory and `responseapi` for shared helper code, while some profiles are present in the runner but absent from the harness map.
  - Inventory drift makes it harder to discover supported profiles, reason about affected coverage, and reliably wire new profiles into docs, CI, and agent selection.
- Desired end state:
  - One authoritative inventory of supported profiles, their ownership, and their classification in local versus CI coverage.
  - Naming between profile directories, helper packages, docs, and harness manifests follows a consistent convention.
- Exit criteria:
  - `e2e/README.md`, `e2e/cmd/e2e/main.go`, and `tools/agent/e2e-profile-map.yaml` stay synchronized for the active profile set.
  - Profile additions or removals can be mechanically validated against the canonical inventory.

## How to Retire Debt

- Close an item only when the underlying architectural gap is materially reduced, not just renamed.
- When a debt item is retired:
  - update the relevant canonical docs and executable rules first
  - mark the item as closed or remove it in the same change
  - reference the retiring PR or change in the item body if useful
