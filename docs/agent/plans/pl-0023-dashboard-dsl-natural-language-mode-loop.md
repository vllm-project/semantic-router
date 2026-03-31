# Dashboard DSL Natural-Language Mode Execution Plan

## Goal

- Deliver the real `nl` mode for dashboard DSL authoring so users can describe routing behavior in natural language, generate DSL, review the result, and deploy it through the existing config-management flow.
- Reuse the repository NL-to-DSL engine, the existing DSL store validation/compile path, and the current deploy preview/runtime-apply flow instead of inventing parallel dashboard-only contracts.
- Keep the work resumable from repository state alone and retire the loop only when frontend UX, backend generation and verification, deploy gating, tests, and docs all converge on the same contract.

## Scope

- `dashboard/frontend/src/pages/{BuilderPage.tsx,builderPageToolbar.tsx}` and adjacent builder support modules
- `dashboard/frontend/src/stores/{dslStore.ts,dslStoreTypes.ts}`
- new or updated dashboard NL-mode support components under `dashboard/frontend/src/pages/**` or `dashboard/frontend/src/components/**`
- `dashboard/backend/router/core_routes.go`
- new dashboard backend generation and verification handlers or support services under `dashboard/backend/handlers/**`
- any extracted dashboard backend support modules needed to keep handler transport seams narrow
- `src/semantic-router/internal/nlgen/**` and any narrow shared support needed to reuse the NL-to-DSL engine cleanly from the dashboard backend
- `src/semantic-router/cmd/dsl/generate.go` only if shared request or service seams need alignment
- docs and tests that must move with the feature contract
- nearest local `AGENTS.md` files for `dashboard/frontend/src/**`, `dashboard/frontend/src/pages/**`, `dashboard/frontend/src/components/**`, and `dashboard/backend/handlers/**`

## Exit Criteria

- `BuilderPage` ships a real `nl` mode instead of the placeholder state.
- The dashboard supports two generation connection modes:
  - default VSR-routed `MoM`
  - custom OpenAI-compatible connection settings with explicit verification
- Generated DSL is loaded into the existing DSL editor state and must pass the same validate, compile, preview, and deploy checks as hand-authored DSL.
- Deploy is blocked on final parse failure or compile errors, and still requires explicit user confirmation after preview.
- Backend generation and verification logic reuses `internal/nlgen` and does not duplicate prompt or repair semantics in the frontend.
- Focused frontend, backend, and DSL-surface tests cover the new contract.
- Repo-native docs and indexes reflect the final contract, and any residual durable gap is promoted to indexed tech debt in the same loop.

## Task List

- [x] `NLM001` Create the canonical ADR and execution plan for dashboard NL-to-DSL mode and align them with the current Builder, deploy, and generation surfaces.
- [ ] `NLM002` Define the dashboard backend API contract for connection verification and NL-to-DSL generation, including the default `MoM` preset and custom OpenAI-compatible request fields.
- [ ] `NLM003` Extract a backend generation service seam that wraps `internal/nlgen`, separates VSR `MoM` transport from custom endpoint transport, and keeps handlers transport-only.
- [ ] `NLM004` Replace the Builder `nl` placeholder with a dedicated NL-mode UI component that collects the instruction, shows connection settings, supports verification, and presents generation results without growing `BuilderPage.tsx` into another hotspot.
- [ ] `NLM005` Extend `dslStore` with narrow NL-mode state and actions so generated DSL hydrates the existing editor, validation, compile, preview, and deploy flow instead of forking a second authoring pipeline.
- [ ] `NLM006` Tighten generated-DSL deploy semantics so final parse failure and compile errors block deploy, warnings stay visible, and deploy still requires merged-preview confirmation.
- [ ] `NLM007` Add focused tests for backend verification and generation handlers, DSL-store state transitions, Builder NL-mode UX, and any `internal/nlgen` integration seams touched by dashboard reuse.
- [ ] `NLM008` Update user-facing docs, README indexes, and any residual debt or follow-up governance artifacts needed to close or narrow the loop cleanly.

## Current Loop

- Loop status: opened on 2026-03-31.
- Completed in this loop:
  - audited the existing Builder `nl` placeholder, DSL store state model, deploy preview and deploy handlers, router chat proxy path, and dashboard model-inventory status seams
  - reviewed the repository-local `internal/nlgen` / `sr-dsl generate` pipeline as the intended generation core for dashboard reuse
  - decided the initial feature shape: Builder-owned NL mode, backend-owned generation, default `MoM`, optional custom OpenAI-compatible connection, explicit verification, and human-confirmed deploy
  - created and indexed the canonical ADR and this execution plan
- Next loop focus:
  - execute `NLM002` first by locking the backend request/response schema and the exact verification semantics before any UI or handler implementation starts
- Commands run:
  - startup doc reads for `AGENTS.md`, `docs/agent/{README.md,governance.md,adr/README.md,plans/README.md}`, and nearest local dashboard rules in `dashboard/frontend/src/{AGENTS.md,pages/AGENTS.md,components/AGENTS.md}` plus `dashboard/backend/handlers/AGENTS.md`
  - broad `codebase-retrieval` for governance conventions, Builder and DSL-store ownership, deploy preview/apply seams, router chat proxy and model-inventory paths, and the NL generation surfaces under `src/semantic-router/internal/nlgen/**`
  - focused file reads across `BuilderPage.tsx`, `builderPageToolbar.tsx`, `dslStore.ts`, `deploy.go`, `core_routes.go`, `proxy_routes.go`, and the current ADR/plan inventories
- Key findings:
  - `BuilderPage` already has a typed `nl` mode and a visible placeholder, so the product seam already exists in the main DSL authoring flow
  - `dslStore` already owns the canonical frontend checks and deploy orchestration, so generated DSL should re-enter that store instead of creating a second deploy path
  - `deploy.go` already provides merge-preview and runtime-apply behavior that should remain the only dashboard deploy contract
  - `PlaygroundPage` and `ChatComponent.tsx` are the wrong seam for deployable DSL authoring because they are optimized for chat transport, tool orchestration, and general routed inference rather than config lifecycle
  - the repo-local `internal/nlgen` flow already captures the generation semantics the dashboard needs: schema-guided prompting, sanitize, parse or repair retry, and AST validation
  - custom connection support introduces secret-handling and verification concerns that should stay request-scoped by default unless the repository later makes a separate durable storage decision

## Decision Log

- 2026-03-31: keep this as an execution plan rather than an ADR-only change because the repository has a clear design direction but still needs a multi-step implementation loop across dashboard frontend, dashboard backend, DSL state, and generation seams.
- 2026-03-31: put NL-to-DSL authoring in `BuilderPage` rather than `PlaygroundPage` because the builder already owns editor modes, DSL lifecycle state, and deploy preview/apply semantics.
- 2026-03-31: default to VSR-routed `MoM` for zero-config generation, but allow a custom OpenAI-compatible endpoint because the dashboard should support controlled override flows without forking the main authoring UX.
- 2026-03-31: keep generation backend-owned so the frontend never becomes the source of truth for prompt construction, repair semantics, provider wiring, or secret transport.
- 2026-03-31: do not auto-deploy generated DSL. The repository should preserve the existing human-reviewed preview and deploy confirmation flow for generated and hand-authored config alike.
- 2026-03-31: treat raw API keys as request-scoped or session-scoped only in the first implementation loop. If the repository later needs durable shared credential storage, promote that to a separate explicit decision instead of hiding it inside this feature rollout.

## Follow-up Debt / ADR Links

- [ADR 0003: Put Dashboard Natural-Language DSL Authoring in Builder Mode with Backend-Owned Generation](../adr/adr-0003-dashboard-dsl-natural-language-mode.md)
- [ADR README](../adr/README.md)
- [Plans README](README.md)
- [Tech Debt README](../tech-debt/README.md) (no dedicated debt entry yet; promote one in-loop if the implementation reveals a durable secret-storage, determinism, or boundary gap)
