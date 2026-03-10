# Go Router Playbook

- Start with `src/semantic-router/` and the matching package under `pkg/`
- Before implementing a new router-side capability, classify it first:
  - text extraction goes to `signal`
  - boolean matching and control logic goes to `decision`
  - per-decision multi-model choice goes to `algorithm`
  - post-decision processing goes to `plugin`
  - only truly cross-cutting behavior belongs at the global level
- Run `make agent-ci-gate CHANGED_FILES="..."` before heavier validation
- If routing behavior or config semantics changed, run `make agent-feature-gate ENV=cpu CHANGED_FILES="..."`
- Add or update E2E coverage when request handling, routing decisions, or config behavior changes
- Keep handlers, selectors, and config logic in separate units; do not merge them into one file
