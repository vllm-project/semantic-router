# Fleet Simulator Migration Execution Plan

## Goal

- Move the fleet simulator from `bench/fleet-simulator` into a maintained `src/fleet-sim` subtree.
- Repackage the simulator as a first-class Python CLI named `vllm-sr-sim`.
- Integrate simulator workflows into the dashboard through backend-managed service proxying and frontend pages.
- Move simulator documentation into the website and retire the standalone simulator frontend.

## Scope

- Python package layout, entrypoints, test paths, and data paths for the simulator
- Root make targets and local dev flow integration
- PyPI and GitHub release publishing for the simulator package and service image
- Dashboard backend APIs and proxy support for simulator operations
- Dashboard frontend routes, navigation, and simulator pages
- Website docs and static assets for simulator documentation and guide artifacts
- Cleanup of obsolete simulator-local frontend assets and stale references
- Harness manifest and repo map updates so `src/fleet-sim` is a first-class maintained surface

## Exit Criteria

- The simulator no longer lives under `bench/fleet-simulator` as its source-of-truth implementation.
- `vllm-sr-sim` installs and runs from the maintained `src/fleet-sim` package.
- Root development flow can build/install both `vllm-sr` and `vllm-sr-sim`.
- `vllm-sr serve` starts a sibling `vllm-sr-sim` container by default unless an external simulator URL is configured.
- Dashboard users can access simulator capabilities through the shared dashboard shell and backend APIs.
- Simulator docs live under `website/` and the standalone simulator frontend is removed.
- Relevant repository validation passes for the touched Python, dashboard, docs, and workflow surfaces.

## Task List

- [x] `F001` Create the durable migration plan and keep loop progress in-repo.
- [x] `F002` Move simulator source, tests, examples, and data from `bench/fleet-simulator` to `src/fleet-sim`.
- [x] `F003` Repackage the simulator as `vllm-sr-sim` with stable console entrypoints and updated metadata.
- [x] `F004` Add simulator make targets, local dev wiring, and release/publish workflow support alongside `vllm-sr`.
- [x] `F005` Add dashboard backend proxy routes and runtime wiring for simulator operations exposed by the service.
- [x] `F006` Add dashboard simulator pages plus a top-nav dropdown that fits the current layout and manager-style page shell.
- [x] `F007` Migrate simulator docs into the website and move guide artifacts into website-managed locations.
- [ ] `F008` Remove obsolete simulator-local frontend assets, validate the affected surfaces, and close remaining harness drift.

## Current Loop

- In progress: finish audit fixes after the main migration by restoring the missing model catalog package, publishing the simulator container image, closing harness drift, and rerunning the affected validation paths.

## Decision Log

- Use `src/fleet-sim` as the maintained home for simulator implementation, tests, data, and packaging assets.
- Keep the simulator as a Python-first CLI package with an HTTP service entrypoint and let the dashboard backend proxy that service instead of reviving the standalone simulator frontend.
- Start `vllm-sr-sim` as a sibling container on the same local runtime network during `vllm-sr serve` unless `TARGET_FLEET_SIM_URL` points to an external simulator service.
- Reuse existing dashboard patterns:
  - Go backend config/settings/proxy layers for optional external services
  - shared `Layout` navigation and `ConfigPageManagerLayout`-style page shells for the frontend

## Implementation Notes

- `src/fleet-sim/` is now the maintained simulator subtree, including the `fleet_sim` package, tests, examples, data, Dockerfile, and package metadata.
- The dashboard backend exposes `/api/fleet-sim/*` and forwards it to `TARGET_FLEET_SIM_URL`.
- The dashboard frontend adds a conditional `Fleet Sim` dropdown with Overview, Workloads, Fleets, and Runs pages.
- Simulator docs, guide assets, and long-form overview material now live under `website/`.
- Harness metadata now needs to recognize `src/fleet-sim` as a maintained subsystem so `agent-report` and future task routing stay accurate.

## Follow-up Debt / ADR Links

- [../tech-debt-register.md](../tech-debt-register.md)
- [../tech-debt/README.md](../tech-debt/README.md)
- [../adr/README.md](../adr/README.md)
