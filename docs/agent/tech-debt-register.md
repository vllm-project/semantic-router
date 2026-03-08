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

## How to Retire Debt

- Close an item only when the underlying architectural gap is materially reduced, not just renamed.
- When a debt item is retired:
  - update the relevant canonical docs and executable rules first
  - mark the item as closed or remove it in the same change
  - reference the retiring PR or change in the item body if useful
