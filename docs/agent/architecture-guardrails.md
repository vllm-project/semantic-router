# Architecture Guardrails

## File and Function Shape

- Prefer files under 400 lines; 800 lines is a hard stop
- Prefer functions under 40-60 lines; 100 lines is a hard stop
- Keep nesting shallow; 4 levels is the maximum

## Module Design

- One file, one main responsibility
- Split long flows into orchestrator + helper functions
- Prefer deep modules with narrow entrypoints over shallow managers with broad shared state
- Keep backend bootstrap and request-time execution on separate seams
- Keep shared storage lifecycle policy in one owner; let domain packages own schema and query semantics
- Keep schema declaration, compatibility migration, and semantic validation on separate seams; canonical field inventories should have one clear owner
- Keep transport translation, semantic evaluation, and request enrichment on separate seams; request pipelines should compose phases instead of hiding them in one filter
- Keep response normalization, streaming finalization, replay or cache persistence, and response-side warning shaping on separate seams
- Keep dashboard backend handler transport, config persistence/deploy control, and runtime status collection on separate seams
- Keep runtime config schema families, canonical import/export, semantic validation, and plugin-family contracts on separate seams
- Keep cross-product control planes on contract-owned seams; CLI, dashboard backend, and operator code should not depend on deep router-runtime internals when a versioned contract or public service can own the boundary
- Keep fleet-sim analytical sizing, DES verification, power or flexibility analysis, and export policy on separate seams
- Keep operator CRD schema declaration, admission validation, controller-side canonical translation, and sample or generated contract upkeep on separate seams
- Keep dashboard frontend route shell, page orchestration, and large interaction containers on separate seams
- Put interfaces at seams:
  - package boundaries
  - external dependency boundaries
  - multiple implementation boundaries

## Deep Modules and Narrow Seams

- If a change needs config loading, backend discovery, metrics wiring, and runtime decision logic at once, the design is already too shallow; split the responsibilities before adding more.
- Keep per-family or per-backend specialization behind dedicated helpers or adapters instead of expanding one giant constructor or switch tree.
- Shared integration policy such as connection setup, retries, collection/index bootstrap, or provider bootstrapping should be implemented once and reused.
- Service composition files should assemble collaborators, not duplicate discovery or lifecycle logic that belongs in the underlying package.

## Preferred Patterns

- Composition over inheritance
- Strategy for routing or selection variation
- Adapter for external APIs and providers
- Factory for runtime-specific construction

## Capability Placement

- Keep extraction and processing responsibilities separate.
- New heuristics, semantic matching, or learned text-understanding features belong in the signal layer first.
- New boolean composition, route gating, or other control logic that combines signals belongs in the decision layer first.
- New per-decision model-choice logic belongs in the algorithm layer first.
- New cache, rewrite, enrichment, or other decision-driven processing belongs in the plugin layer first.
- Use the global level only for behavior that is intentionally cross-cutting and not owned by a specific signal, decision, algorithm, or plugin.
- If a feature touches signal, decision, algorithm, plugin, and global config at once, choose one primary layer as the owner and keep the rest as thin integration seams.
- Treat the flow as ordered responsibilities when possible:
  - `signal` extracts facts
  - `decision` combines facts with boolean control logic
  - `algorithm` chooses among candidate models for a matched decision
  - `plugin` performs post-decision or post-selection processing
  - `global` carries intentionally cross-cutting behavior

## API Type Contracts

- Use official SDK types for OpenAI and Anthropic request/response handling:
  - `github.com/openai/openai-go` for OpenAI-shaped data
  - `github.com/anthropics/anthropic-sdk-go` for Anthropic-shaped data
- Do not define custom structs that duplicate what the SDK provides
  (e.g., custom `ChatCompletionRequest`, `ChatCompletionResponse`)
- Exceptions: packages that intentionally avoid the SDK dependency for
  isolation (e.g., E2E fixtures, standalone training tools) should document
  the reason in a comment and keep their custom types minimal
- When the SDK type does not cover a field the router needs, extend via
  composition (`type ExtendedReq struct { openai.ChatCompletionNewParams; Extra string }`)
  rather than reimplementing the whole struct

## Avoid

- giant managers
- giant switches with many unrelated responsibilities
- files that mix config parsing, orchestration, I/O, and domain logic
- shallow wrapper layers that simply re-express the same backend bootstrap logic in multiple packages
- leaking test or docs dependencies into production code

## Legacy Hotspots

These files are existing debt, not acceptable targets for new growth:

- `src/semantic-router/pkg/config/config.go`
- `src/semantic-router/pkg/config/validator.go`
- `src/semantic-router/pkg/config/rag_plugin.go`
- `src/semantic-router/pkg/classification/classifier.go`
- `src/semantic-router/pkg/extproc/processor_req_body.go`
- `src/semantic-router/pkg/extproc/processor_res_body.go`
- `src/vllm-sr/cli/core.py`
- `src/vllm-sr/cli/commands/runtime.py`
- `src/fleet-sim/fleet_sim/optimizer/base.py`
- `deploy/operator/api/v1alpha1/semanticrouter_types.go`
- `deploy/operator/controllers/canonical_config_builder.go`
- `dashboard/backend/handlers/config.go`
- `dashboard/backend/handlers/deploy.go`
- `dashboard/backend/handlers/status.go`
- `dashboard/backend/handlers/status_modes.go`
- `dashboard/frontend/src/App.tsx`
- `dashboard/frontend/src/pages/DashboardPage.tsx`
- `dashboard/frontend/src/pages/BuilderPage.tsx`
- `dashboard/frontend/src/pages/ConfigPage.tsx`
- `dashboard/frontend/src/pages/SetupWizardPage.tsx`
- `dashboard/frontend/src/components/ChatComponent.tsx`
- `dashboard/frontend/src/components/ExpressionBuilder.tsx`

When touching one of these files:

- prefer extraction-first edits that move types, helpers, or display-only code into adjacent modules
- do not add a second major responsibility into the same hotspot file
- treat any net reduction in file size or complexity as part of the acceptance bar for the change
- the structural gate applies a ratchet here: these files may still be over global limits, but they must not grow, and touched code should move toward the standard shape
- for config hotspots, keep schema families, canonical conversion, plugin-family contracts, and semantic validation on separate seams
- for classifier hotspots, keep model discovery, family-specific mapping or backend logic, and request-time orchestration on separate seams
- for CLI orchestration hotspots, keep top-level command routing and user-facing flow in the main orchestration seams but move docker/runtime helpers, container wiring, and support types into adjacent modules; keep compatibility barrels such as `docker_cli.py` thin
- for extproc response hotspots, keep provider normalization, streaming finalization, replay/cache helpers, and response warning mutations on separate seams
- for fleet-sim optimizer hotspots, keep analytical kernels, DES verification, power/flex helpers, and package exports on separate seams
- for operator hotspots, keep CRD schema, admission validation, controller-side canonical translation, and generated/sample upkeep on separate seams
- for dashboard frontend shell hotspots, keep routing/auth/layout composition above page-specific and interaction-specific support logic
- for dashboard backend handler hotspots, keep HTTP transport thin and move file mutation, runtime apply, container probing, and status collection into sibling services or helpers
- for dashboard pages, keep route state and async orchestration in the page but move pure config builders, validation tables, and section panels out
- for large dashboard components, keep ReactFlow or transport orchestration in the container and move pure data-model helpers, constants, and display fragments out
