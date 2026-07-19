# PL-0038 Entrypoints and Multi-Recipe Routing

## Goal

Implement issue #2331: let one router configuration carry multiple named routing
recipes, with an `entrypoints` table mapping request-facing virtual model names
to recipes. Follow-up issue #2354 (normalizing `auto_model_name(s)` and
algorithm virtual slugs through entrypoints) builds on this plan but is not part
of it.

## Scope

- Canonical schema: top-level `entrypoints` and `recipes` beside the existing
  `routing` block. The top-level `routing` profile is the `default` recipe;
  `recipes` adds named additional profiles. A recipe named `default` may only
  appear when the top-level `routing` block carries no profile of its own.
- Internal representation: normalized `Recipes`/`Entrypoints` on `RouterConfig`;
  the default recipe stays bridged into the existing flat routing fields so
  current read sites keep working unchanged.
- Request path: resolve model name → entrypoint → recipe before signal
  evaluation in `pkg/extproc`; decision evaluation reads the per-request recipe.
- Validation: structural recipe/entrypoint checks first, then cross-surface
  collision checks (entrypoint names versus configured model names and aliases).
- Surfaces after the core lands: canonical export/dump, `/v1/models` listing,
  DSL round-trip, CLI/operator/dashboard, docs and reference config, E2E.

Non-goals (from #2331): recipes do not own provider definitions or global model
assets; the global signal registry stays outside recipes; no breaking change for
single-profile configs.

## Open Decisions

- Recipe-level `strategy` / `model_selection` with `global.router` as the
  default source: proposed to the maintainer in #2331, pending confirmation.
  Until confirmed, recipes inherit the global values and the canonical recipe
  schema does not expose the two fields.
- Entrypoint names colliding with configured model names or aliases: proposed
  in #2331 as a load-time validation error, pending confirmation.

## Exit Criteria

- Different virtual model names select different recipes at request time.
- Existing `vllm-sr/auto` / `auto` behavior is unchanged for single-profile
  configs.
- `routing.signals`, `routing.projections`, and `routing.decisions` remain the
  recipe-level profile shape.
- Config validation rejects duplicate recipe names, unknown recipe references,
  duplicate entrypoint model names, recipe-owned model cards, and a
  `default`-named recipe conflicting with a non-empty top-level `routing` block.
- Reference config, config docs, and tutorials show a multi-recipe example.
- E2E covers two entrypoints selecting different recipes and the `/v1/models`
  listing.

## Task List

- [ ] T1 execution plan committed (this file)
- [ ] T2 canonical `entrypoints`/`recipes` schema and normalization with the
      default-recipe bridge
- [ ] T3 cross-surface validation for entrypoint and recipe mappings
- [ ] T4 canonical export and config dump emit normalized entrypoints and
      recipes
- [ ] T5 extproc request-entry entrypoint resolution before signal evaluation
- [ ] T6 `/v1/models` lists entrypoint model names
- [ ] T7 multi-recipe unit tests in config and extproc
- [ ] T8 config-contract docs, reference config, and tutorials
- [ ] T9 E2E coverage for multi-recipe entrypoints
- [ ] T10 DSL round-trip for recipes and entrypoints (follow-up PR)
- [ ] T11 CLI and operator surfaces (follow-up PR)
- [ ] T12 dashboard surfaces (follow-up PR)

## Next Action

Land T2 on the `feat/entrypoints-recipes-2331` branch.

## Operating Rules

- `config.go` stays a schema table; recipe/entrypoint contracts live in
  `recipes.go` and canonical normalization in `canonical_recipes.go`.
- `processor_req_body.go` stays an orchestrator; entrypoint resolution lives in
  a dedicated `req_filter_entrypoint.go`.
- The flat routing fields on `RouterConfig` must always equal the default
  recipe; runtime code moves to recipe reads, never the reverse.
- Every step passes `make agent-lint` and the affected package tests before the
  next step starts.

## Related Docs

- Issue #2331, follow-up #2354, roadmap #2287
- `docs/agent/architecture-guardrails.md`
- `src/semantic-router/pkg/config/AGENTS.md`
- `src/semantic-router/pkg/extproc/AGENTS.md`
