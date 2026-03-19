# Dashboard Backend Handlers Notes

## Scope

- `dashboard/backend/handlers/**`
- local rules for dashboard backend config/deploy/status hotspots

## Responsibilities

- Handler files should own HTTP transport concerns: method guards, request decoding, response encoding, and delegation.
- Config persistence, backup/version inventory, runtime propagation, and status collection belong in adjacent helpers or services instead of growing inline inside handlers.
- Keep config edit/deploy/rollback flows distinct from system-status collection; shared runtime helpers should expose narrow seams.

## Change Rules

- `config.go` is a hotspot. Keep request and response orchestration there, but move canonical config read/write, endpoint validation, rollback, and runtime-apply helpers into sibling modules when extending it.
- `deploy.go` is a hotspot. Keep deploy endpoint orchestration and response types there, but move preview merge, backup/version inventory, direct-write, and rollback/apply behavior into support modules.
- `status.go` and `status_modes.go` are runtime-status hotspots. Keep top-level status response shaping and mode dispatch there, but move Docker or supervisor probing, log parsing, router-runtime synthesis, and model-info fetch helpers into sibling collectors or support files.
- Do not add new YAML mutation, container probing, or long-lived runtime side-effect helpers inline in handler files; extract the seam first.
- If a change touches both config/deploy and status collection, treat that as a design smell and look for a narrower shared helper instead of growing another handler.
