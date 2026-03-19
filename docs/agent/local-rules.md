# Local Rules

These local `AGENTS.md` files are first-class supplements to the shared harness. They do not replace the canonical contract, but they provide hotspot-specific boundaries near the code.

## Indexed Local `AGENTS.md` Files

- [../../src/vllm-sr/cli/AGENTS.md](../../src/vllm-sr/cli/AGENTS.md)
  - CLI orchestration and `docker_cli.py` hotspot rules
- [../../src/fleet-sim/AGENTS.md](../../src/fleet-sim/AGENTS.md)
  - simulator package boundaries, service surface, and public export rules
- [../../src/fleet-sim/fleet_sim/optimizer/AGENTS.md](../../src/fleet-sim/fleet_sim/optimizer/AGENTS.md)
  - optimizer hotspot boundaries, export policy, and extraction-first rules
- [../../src/semantic-router/pkg/config/AGENTS.md](../../src/semantic-router/pkg/config/AGENTS.md)
  - config schema and `config.go` hotspot rules
- [../../src/semantic-router/pkg/classification/AGENTS.md](../../src/semantic-router/pkg/classification/AGENTS.md)
  - classification bootstrap, family-boundary, and hotspot rules
- [../../src/semantic-router/pkg/extproc/AGENTS.md](../../src/semantic-router/pkg/extproc/AGENTS.md)
  - extproc processor and router hotspot rules
- [../../deploy/operator/api/v1alpha1/AGENTS.md](../../deploy/operator/api/v1alpha1/AGENTS.md)
  - operator CRD schema and admission-validation hotspot rules
- [../../deploy/operator/controllers/AGENTS.md](../../deploy/operator/controllers/AGENTS.md)
  - operator controller translation and discovery hotspot rules
- [../../dashboard/frontend/src/AGENTS.md](../../dashboard/frontend/src/AGENTS.md)
  - dashboard frontend app-shell, auth/setup routing, and layout-boundary rules
- [../../dashboard/frontend/src/pages/AGENTS.md](../../dashboard/frontend/src/pages/AGENTS.md)
  - dashboard page orchestration and large-page hotspot rules
- [../../dashboard/frontend/src/components/AGENTS.md](../../dashboard/frontend/src/components/AGENTS.md)
  - dashboard component-level hotspot rules
- [../../dashboard/backend/handlers/AGENTS.md](../../dashboard/backend/handlers/AGENTS.md)
  - dashboard backend handler transport and runtime-control hotspot rules
- [../../e2e/testcases/AGENTS.md](../../e2e/testcases/AGENTS.md)
  - testcase contract and acceptance-threshold rules

## Policy

- These files are local supplements, not alternate sources of truth.
- Durable cross-cutting guidance belongs in `docs/agent/*` or the executable harness manifests.
- Local rules should stay narrow and directory-specific.
- Changes to these files are harness-doc changes. They should resolve through the harness entrypoints, not through the surrounding business-code task rules.
- Every indexed local rule should use the same three sections:
  - `## Scope`
  - `## Responsibilities`
  - `## Change Rules`
