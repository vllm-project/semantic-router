# vLLM Semantic Router agent entry

vLLM Semantic Router is an Envoy ExtProc router for LLM inference. A public
entrypoint resolves to an isolated recipe; its signals and projections feed a
decision and algorithm, which select a backend and recipe-scoped plugins.

## Start here

- Read the nearest `AGENTS.md` only for directories you change.
- Use `tools/agent/docs/change-surfaces.md` when a user-visible router contract
  crosses configuration, runtime, deployment, or documentation.
- `tools/agent/domains.yaml` is the single changed-path registry for ownership,
  minimum checks, CI jobs, images, and E2E profiles.

```bash
make impact ENV=cpu CHANGED_FILES="path/one path/two"
make check CHANGED_FILES="path/one path/two"
make verify DOMAIN=<domain>       # explicit integration check
make verify PROFILE=<e2e-profile>
make ci-full                      # complete local PR baseline
```

Use `make harness-check` after changing the registry, workflows, harness code,
or these instructions. `impact` reports facts; it does not select a skill,
prescribe a work loop, or decide when the task is complete.

## Repository map

- `src/semantic-router/`: Go router, ExtProc runtime, routing, and APIs
- `src/vllm-sr/`: Python CLI and local stack orchestration
- `config/`: canonical configuration, fragments, schemas, and recipes
- `candle-binding/`, `ml-binding/`, `nlp-binding/`, `onnx-binding/`: inference bindings
- `dashboard/`: React frontend and Go management backend
- `deploy/`: deployment artifacts and operator
- `e2e/`: end-to-end framework and profiles
- `tools/`: build, development, release, security, and harness tooling
- `website/`: public documentation

## Durable constraints

- Use the existing `vllm-sr serve` local-image flow for local runtime behavior.
- Keep steady-state configuration canonical. The public document is
  `version/listeners/providers/evaluation/routing/entrypoints/recipes/global`;
  legacy layouts belong in explicit migration tooling, not the runtime parser.
- A behavior-visible routing, startup, config, Docker, CLI, or API change needs
  an appropriate integration or E2E assertion. Pure refactors do not.
- Generated artifacts and public docs change with their source contract.
- Numeric file, function, nesting, and interface limits are review signals, not
  architecture. Forbidden dependencies, new cycles, generated invariants, and
  unowned root files remain blocking checks.
- Prefer cohesive modules. Split or extract when ownership becomes mixed, not
  to satisfy a line-count target.
- Use an execution plan only for genuinely resumable multi-session work. Prefer
  GitHub issues for tracked debt; keep repository-only architectural risks in
  `tools/agent/docs/architecture-risks.md`.
- PR commits use `git commit -s`. Keep unrelated changes out of the branch.
- Never publish credentials, private hostnames, or private fleet details.

## Optional repository skills

Choose a skill by the semantics of the task, not by an automatic path match:

- `router-contract-change`: request-visible config/signal/decision/algorithm/plugin changes
- `routing-calibration`: live maintained-recipe probe and calibration work
- `maintainer-ops`: reviewed issue, PR, release, or GitHub mutation workflows

Canonical contributor and GitHub metadata remain in `CONTRIBUTING.md`, the
issue and PR templates, and `.prowlabels.yaml`.
