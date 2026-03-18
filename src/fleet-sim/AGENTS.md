# Fleet Sim Notes

## Scope

- `src/fleet-sim/**`
- `src/fleet-sim/fleet_sim/**`

## Responsibilities

- Keep the simulator package runnable as both the `vllm-sr-sim` CLI and the HTTP service used by the dashboard sidecar.
- Keep core simulation logic, API transport, and catalog data separated so package moves do not break imports.

## Change Rules

- Preserve `fleet_sim` import stability; when moving files, update package exports and package data together.
- Keep API route modules thin. Put simulation and catalog logic in package modules, not FastAPI handlers.
- Treat `fleet_sim/__init__.py` as the public surface. If symbols are added or moved, update the exports there deliberately.
