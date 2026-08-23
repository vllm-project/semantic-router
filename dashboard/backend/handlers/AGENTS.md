# Dashboard Backend Handlers Notes

## Scope

- `dashboard/backend/handlers/**`
- local rules for dashboard backend API-client, status, and console-only hotspots

## Responsibilities

- Handler files should own HTTP transport concerns: method guards, request decoding, response encoding, and delegation.
- Router access and routing state use versioned Router APIs. Dashboard handlers must not become an authoritative store, inference proxy, quota engine, or runtime-config mutation path.
- Console-only integrations and status collection belong in adjacent helpers or services instead of growing inline inside handlers.
- Keep Router API-client plumbing distinct from system-status collection; shared runtime helpers should expose narrow seams.

## Change Rules

- Router Management forwarding is owned by `dashboard/backend/router/` and
  `dashboard/backend/routerauth/`. Keep browser-session exchange, request
  signing, and transport policy there; handlers must not acquire or project
  Router authority.
- `status.go` and `status_modes.go` are runtime-status hotspots. Keep top-level status response shaping and mode dispatch there, but move Docker or supervisor probing, log parsing, router-runtime synthesis, and model-info fetch helpers into sibling collectors or support files.
- Do not add YAML mutation, inference forwarding, access enforcement, container probing, or long-lived runtime side-effect helpers inline in handler files; use the owning Router API or extract the console-only seam first.
- If a change touches both Router API-client behavior and status collection, treat that as a design smell and look for a narrower shared helper instead of growing another handler.
