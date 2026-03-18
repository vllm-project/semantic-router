# Local Rules

These local `AGENTS.md` files are first-class supplements to the shared harness. They do not replace the canonical contract, but they provide hotspot-specific boundaries near the code.

## Indexed Local `AGENTS.md` Files

- [../../src/vllm-sr/cli/AGENTS.md](../../src/vllm-sr/cli/AGENTS.md)
  - CLI orchestration and `docker_cli.py` hotspot rules
- [../../src/fleet-sim/AGENTS.md](../../src/fleet-sim/AGENTS.md)
  - simulator package boundaries, service surface, and public export rules
- [../../src/semantic-router/pkg/config/AGENTS.md](../../src/semantic-router/pkg/config/AGENTS.md)
  - config schema and `config.go` hotspot rules
- [../../src/semantic-router/pkg/extproc/AGENTS.md](../../src/semantic-router/pkg/extproc/AGENTS.md)
  - extproc processor and router hotspot rules
- [../../dashboard/frontend/src/pages/AGENTS.md](../../dashboard/frontend/src/pages/AGENTS.md)
  - dashboard page orchestration and large-page hotspot rules
- [../../dashboard/frontend/src/components/AGENTS.md](../../dashboard/frontend/src/components/AGENTS.md)
  - dashboard component-level hotspot rules

## Policy

- These files are local supplements, not alternate sources of truth.
- Durable cross-cutting guidance belongs in `docs/agent/*` or the executable harness manifests.
- Local rules should stay narrow and directory-specific.
- Changes to these files are harness-doc changes. They should resolve through the harness entrypoints, not through the surrounding business-code task rules.
- Every indexed local rule should use the same three sections:
  - `## Scope`
  - `## Responsibilities`
  - `## Change Rules`
