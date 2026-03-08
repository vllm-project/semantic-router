# GitHub Copilot Instructions

This repository uses an agent harness. Treat reviews and code suggestions as part of that harness instead of as isolated code edits.

## Source of Truth Order

When reviewing or generating changes, follow the repository rule layers in this order:

1. `AGENTS.md`
2. `docs/agent/README.md`
3. relevant docs under `docs/agent/*`
4. executable rule sources usnder `tools/agent/*`, `tools/make/agent.mk`, and `.github/workflows/*`
5. nearest local `AGENTS.md` for hotspot directories

Interpret them this way:

- `AGENTS.md` is a short entrypoint, not the full handbook.
- `docs/agent/*` is the human-readable system of record.
- manifests, scripts, Make targets, and CI workflows are the executable rule layer.
- local `AGENTS.md` files are narrow hotspot supplements, not alternate truth.

Do not invent a second source of truth in reviews or code suggestions.

## Review Priorities

When reviewing a change, first check whether it follows the repository harness:

- the change starts from the correct project-level skill and impacted surfaces
- code, docs, and executable rules stay aligned when the change affects more than one layer
- contributor-facing wrappers such as `README.md`, `.github/PULL_REQUEST_TEMPLATE.md`, issue templates, and this file stay aligned with the canonical harness
- repeated governance requirements are enforced mechanically when possible instead of only described in prose

Then review the code change itself for bugs, regressions, missing validation, and maintainability issues.

## Harness-Specific Things to Check

- For harness changes, verify alignment with:
  - `docs/agent/governance.md`
  - `docs/agent/testing-strategy.md`
  - `docs/agent/feature-complete-checklist.md`
  - `docs/agent/skill-catalog.md`
  - `tools/agent/repo-manifest.yaml`
  - `tools/agent/task-matrix.yaml`
  - `tools/agent/skill-registry.yaml`
- For hotspot changes, verify the nearest local `AGENTS.md` was respected.
- For long-horizon work, verify active loop execution lives under `docs/agent/plans/*.md` instead of ad hoc notes.
- For durable unresolved code/spec mismatches, verify the gap is promoted to an indexed debt entry under `docs/agent/tech-debt/`.
- For durable governance or architecture decisions, verify they are recorded in `docs/agent/adr/` when needed.

## Architecture and Modularity Checks

Review changes against `docs/agent/architecture-guardrails.md` and `docs/agent/module-boundaries.md`.

Flag issues when a change:

- grows a legacy hotspot without reducing responsibility elsewhere
- adds unrelated responsibilities into the same file
- uses large manager/god-object patterns where extraction or composition should be used
- introduces interfaces away from true seams
- ignores the repository layer model:
  - `signal`: extract information from request/response text
  - `decision`: boolean control logic using signals and conditions
  - `algorithm`: per-decision model selection
  - `plugin`: post-decision processing
  - `global`: intentionally cross-cutting behavior

## Validation Expectations

Review whether the change matches the repository validation ladder in `docs/agent/testing-strategy.md`.

Flag missing validation when:

- behavior-visible routing, startup, config, Docker, CLI, or API changes do not update/add E2E unless the change is clearly a pure refactor
- harness-only changes do not update the relevant source-of-truth docs or executable rules
- contributor-interface changes drift away from the canonical harness entrypoints

## Review Output Style

Prioritize findings about:

1. bugs or behavioral regressions
2. harness violations or source-of-truth drift
3. missing validation or E2E coverage
4. architecture, modularity, and hotspot-ratchet violations

Keep findings concrete, file-specific, and actionable.

If there are no findings, say so explicitly and briefly note any residual risks or validation gaps.
