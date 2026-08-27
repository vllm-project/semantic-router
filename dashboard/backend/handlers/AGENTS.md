# Dashboard Backend Handlers Notes

## Scope

- `dashboard/backend/handlers/**`
- local rules for dashboard backend API-client, status, and console-only hotspots

## Responsibilities

- Handler files should own HTTP transport concerns: method guards, request decoding, response encoding, and delegation.
- Dashboard handlers expose the reference control-plane and optional Agent APIs over
  narrow services. They must not become an inference proxy, quota executor, ExtProc
  implementation, or direct runtime-store mutation path.
- Console-only integrations and status collection belong in adjacent helpers or services instead of growing inline inside handlers.
- Keep Router API-client plumbing distinct from system-status collection; shared runtime helpers should expose narrow seams.

## Change Rules

- Desired-state handlers delegate to control-plane domain services and snapshot
  publishers. Browser-session exchange, request signing, and transport policy stay in
  dedicated auth/client packages; handlers must not acquire data-plane authority.
- `status.go` and `status_modes.go` are runtime-status hotspots. Keep top-level status response shaping and mode dispatch there, but move Docker or supervisor probing, log parsing, router-runtime synthesis, and model-info fetch helpers into sibling collectors or support files.
- Do not add YAML mutation, inference forwarding, access enforcement, container probing, or long-lived runtime side-effect helpers inline in handler files; use the owning Router API or extract the console-only seam first.
- If a change touches both Router API-client behavior and status collection, treat that as a design smell and look for a narrower shared helper instead of growing another handler.
