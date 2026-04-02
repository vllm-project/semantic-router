# ADR 0003: Put Dashboard Natural-Language DSL Authoring in Builder Mode with Backend-Owned Generation

## Status

Proposed

## Context

The dashboard already has most of the runtime and authoring surfaces needed for a natural-language-to-DSL workflow, but they currently stop short of a durable product contract.

- `dashboard/frontend/src/pages/BuilderPage.tsx` and `dashboard/frontend/src/stores/dslStore.ts` already model three editor modes (`visual`, `dsl`, `nl`), but the `nl` mode is only a placeholder.
- The DSL builder path already has the durable post-authoring checks that matter for deployable config: frontend validation, compile-to-YAML, deploy preview, deploy confirmation, and backend runtime apply.
- `dashboard/backend/handlers/deploy.go` already provides the merge-preview and runtime-apply semantics that the dashboard should continue to use for any generated DSL.
- The general playground is intentionally a chat surface. `dashboard/frontend/src/components/ChatComponent.tsx` is already a transport and UI orchestration hotspot, and `/api/router/v1/chat/completions` is a routed inference path, not a deployable DSL-authoring contract.
- The repository is adding a dedicated NL-to-DSL generation pipeline under `src/semantic-router/internal/nlgen/**` and `src/semantic-router/cmd/dsl/generate.go`. That path already captures the right generation semantics: schema-guided prompting, output sanitization, parse/repair retries, and AST validation.
- The requested dashboard feature needs two connection modes:
  - a zero-config default that uses the VSR-routed `MoM` model from the dashboard environment
  - an optional custom OpenAI-compatible connection with explicit base URL, model, API key, and connection verification

Without a durable decision, the repository risks duplicating prompt logic in the frontend, coupling deployable authoring to the general chat playground, or letting secret handling and verification semantics drift between CLI, dashboard, and runtime paths.

## Decision

Adopt the following product and architecture contract for dashboard NL-to-DSL authoring.

- Put natural-language DSL authoring in `BuilderPage` as the real implementation of the existing `nl` editor mode, not in `PlaygroundPage`.
- Keep generation backend-owned. The frontend must call dedicated dashboard endpoints for generation and connection verification instead of calling LLM providers directly.
- Reuse the repository NL-to-DSL engine under `src/semantic-router/internal/nlgen/**` as the single generation core for both CLI and dashboard flows instead of maintaining a second prompt or repair implementation in the dashboard.
- Use the VSR-routed `MoM` model as the default generation preset in the dashboard environment.
- Support an optional custom OpenAI-compatible connection profile with explicit request-supplied fields for:
  - base URL
  - model
  - API key
  - timeout
  - temperature
  - max retries
- Provide a dedicated connection-verification endpoint before generation so the dashboard can check reachability, authentication, and model availability without requiring a full deploy or generation run.
- Treat custom credentials as request-scoped or session-scoped by default. The dashboard must not introduce silent durable storage of raw API keys in frontend local storage or router config.
- After generation succeeds, load the returned DSL into the existing DSL editor state and rerun the established dashboard checks:
  - validate the DSL
  - compile it to the canonical YAML fragment
  - surface warnings and diagnostics in the existing editor and deploy UX
- Tighten deploy semantics for generated DSL:
  - final parse failure blocks deploy
  - compile errors block deploy
  - warnings remain visible and reviewable, but do not silently bypass preview
- Keep deploy human-confirmed. Generation does not auto-deploy; the user must still review the generated DSL and the merged deploy preview before runtime apply.
- Respect existing local boundary rules while implementing the feature:
  - `BuilderPage.tsx` stays the page orchestration seam, but new NL-mode UI belongs in adjacent support components
  - `dslStore.ts` owns editor state and actions, not page-local JSX
  - backend handlers stay transport shells and delegate provider or generation logic into narrower support modules

## Consequences

- The dashboard gets one coherent authoring path: natural language can produce DSL, but the result still passes through the same validation, compile, preview, and deploy gates as hand-authored DSL.
- The repository avoids turning the general chat playground into a second configuration-management surface.
- CLI and dashboard generation stay aligned because both depend on the same `internal/nlgen` contract rather than separate prompt copies.
- The default `MoM` preset gives the feature a zero-config path, but generation behavior will still reflect the deployed router environment. If the repository later needs stronger determinism than the routed `MoM` contract provides, a future ADR may supersede this one with a dedicated `dsl-generator` alias or similar stable provider contract.
- The feature introduces new backend API surface area and state-management responsibilities in the dashboard store, so focused tests and narrow extraction seams are mandatory to avoid further hotspot growth.
- Custom connection support increases secret-handling and verification responsibilities. If the repository later needs durable shared credential storage, that should be handled as a separate explicit architecture decision rather than added implicitly during the first implementation loop.
