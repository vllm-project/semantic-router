# TD006: Structural Rule Target Still Exceeds Reality in Key Legacy Hotspots

## Status

Open

## Scope

architecture ratchet versus current code

## Summary

The harness correctly ratchets the repo toward smaller modules, but several legacy hotspots still depend on explicit exceptions. The v0.3 config-contract rollout also touched additional hotspot files that still need extraction-first follow-up before they can satisfy the structural target directly.

## Evidence

- [docs/agent/architecture-guardrails.md](../architecture-guardrails.md)
- [docs/agent/repo-map.md](../repo-map.md)
- [tools/agent/structure-rules.yaml](../../../tools/agent/structure-rules.yaml)
- [tools/linter/go/.golangci.agent.yml](../../../tools/linter/go/.golangci.agent.yml)
- [tools/agent/scripts/agent_doc_validation.py](../../../tools/agent/scripts/agent_doc_validation.py)
- [dashboard/frontend/src/pages/OpenClawPage.tsx](../../../dashboard/frontend/src/pages/OpenClawPage.tsx)
- [dashboard/frontend/src/components/ClawRoomChat.tsx](../../../dashboard/frontend/src/components/ClawRoomChat.tsx)
- [dashboard/backend/handlers/openclaw.go](../../../dashboard/backend/handlers/openclaw.go)
- [dashboard/backend/handlers/openclaw_rooms.go](../../../dashboard/backend/handlers/openclaw_rooms.go)
- [dashboard/backend/handlers/openclaw_teams.go](../../../dashboard/backend/handlers/openclaw_teams.go)
- [dashboard/backend/handlers/openclaw_workers.go](../../../dashboard/backend/handlers/openclaw_workers.go)
- [dashboard/backend/handlers/openclaw_provision.go](../../../dashboard/backend/handlers/openclaw_provision.go)
- [dashboard/backend/handlers/openclaw_test.go](../../../dashboard/backend/handlers/openclaw_test.go)
- [dashboard/backend/handlers/config.go](../../../dashboard/backend/handlers/config.go)
- [dashboard/backend/handlers/deploy.go](../../../dashboard/backend/handlers/deploy.go)
- [deploy/operator/controllers/semanticrouter_controller.go](../../../deploy/operator/controllers/semanticrouter_controller.go)
- [src/vllm-sr/cli/models.py](../../../src/vllm-sr/cli/models.py)
- [src/vllm-sr/cli/commands/show_config.py](../../../src/vllm-sr/cli/commands/show_config.py)
- [docs/agent/plans/pl-0003-v0-3-config-contract-rollout.md](../plans/pl-0003-v0-3-config-contract-rollout.md)

## Why It Matters

- The harness correctly states that large hotspot files are debt, not precedent, but several code areas still depend on hotspot-specific exceptions and ratchets.
- The harness-side validation layer still includes at least one oversized script that remains above the warning threshold even after related changes land.
- OpenClaw management and dashboard UI now sit on the same debt boundary: the feature surface is active and maintained, but the implementation still spans oversized page/component/handler files that cannot yet satisfy the global structure target directly.
- The agent-specific Go complexity gate also needs explicit legacy exclusions for the same OpenClaw handlers/tests until those modules are decomposed enough to meet the global `cyclop`, `funlen`, `gocognit`, and `nestif` thresholds.
- The config-contract rollout now depends on additional large orchestrator files in dashboard/backend, operator generation, and CLI schema compatibility code; until those are decomposed, the structure ratchet will continue to surface exceptions when config work lands.
- This is the right governance posture, but it remains a real code/spec gap until the worst hotspots no longer need special handling.

## Desired End State

- The global structure rules become the common case rather than something many hotspot directories can only approach gradually.
- Config contract rollout work can land by extending narrower helper modules instead of growing dashboard handler, operator controller, or CLI hotspot files.

## Exit Criteria

- The highest-risk files no longer need special ratchet treatment to stay within the intended modularity envelope.
- Config rollout follow-up extracts stable schema/export/merge helpers out of the current hotspot files enough for the relevant lint and structure gates to pass without bespoke exceptions.
