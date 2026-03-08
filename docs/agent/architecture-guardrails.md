# Architecture Guardrails

## File and Function Shape

- Prefer files under 400 lines; 600 lines is a hard stop
- Prefer functions under 40-60 lines; 80 lines is a hard stop
- Keep nesting shallow; 4 levels is the maximum

## Module Design

- One file, one main responsibility
- Split long flows into orchestrator + helper functions
- Put interfaces at seams:
  - package boundaries
  - external dependency boundaries
  - multiple implementation boundaries

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

## Avoid

- giant managers
- giant switches with many unrelated responsibilities
- files that mix config parsing, orchestration, I/O, and domain logic
- leaking test or docs dependencies into production code

## Legacy Hotspots

These files are existing debt, not acceptable targets for new growth:

- `src/semantic-router/pkg/config/config.go`
- `src/semantic-router/pkg/extproc/processor_req_body.go`
- `src/semantic-router/pkg/extproc/processor_res_body.go`
- `src/vllm-sr/cli/docker_cli.py`
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
- for CLI orchestration hotspots, keep top-level command routing and user-facing flow in the main file but move docker/runtime helpers, container wiring, and support types into adjacent modules
- for dashboard pages, keep route state and async orchestration in the page but move pure config builders, validation tables, and section panels out
- for large dashboard components, keep ReactFlow or transport orchestration in the container and move pure data-model helpers, constants, and display fragments out
