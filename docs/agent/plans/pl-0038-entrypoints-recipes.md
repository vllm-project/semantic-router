# PL-0038 Entrypoints and Multi-Recipe Routing

## Goal

Let one router configuration expose several request-facing entrypoints backed
by isolated routing recipes. Preserve the original single-profile contract by
normalizing the top-level `routing` block as the `default` recipe selected by
`vllm-sr/auto` and the other configured auto aliases.

## Final Architecture

- `entrypoints[].model_names` are virtual request model IDs. They select one
  named recipe and never reach a backend.
- The top-level `routing` block is the `default` recipe. A recipes-only config
  may declare an explicit recipe named `default`, but not alongside a non-empty
  top-level profile.
- A recipe owns its `signals`, `projections`, `decisions`, `strategy`, decision
  algorithms, and route plugins. PII, jailbreak, and authz role-binding rules
  are policy and therefore recipe-local.
- Signal, projection, and decision names are local identifiers. Names may repeat
  in different recipes; every reference must resolve inside its owning recipe.
- Model cards, provider/backend bindings, model artifacts, service transports,
  and store implementations remain shared infrastructure.
- Request-time classifiers and selectors are compiled per recipe. Semantic
  cache, replay, learning/session state, metric identities, and handoff
  penalties carry the recipe namespace so identical local names cannot mix
  state.
- Concrete backend model and LoRA requests do not select a recipe and bypass
  recipe evaluation and recipe-local state/plugins. Direct ReMoM/Fusion/Flow
  aliases select the default recipe and then constrain the eligible decisions
  to the requested looper type.

## Request Resolution

1. Resolve the request model before signal evaluation.
2. Entrypoint model → mapped named recipe.
3. Auto alias → default recipe.
4. Direct looper alias → default recipe plus looper-type decision filtering.
5. Concrete model/LoRA → provider passthrough with no recipe.
6. Evaluate only the selected recipe's classifier graph, projections,
   decisions, selector registry, and plugins.
7. Emit `x-vsr-selected-recipe`; record the same scope in replay/Insights and
   all internal state keys.

## Validation Contract

Config loading rejects:

- duplicate recipe names and conflicting default-profile declarations
- unknown recipe references, empty entrypoints, duplicate virtual model claims,
  and collisions with models, LoRAs, auto aliases, or looper aliases
- recipe-owned model cards
- duplicate local signal, projection, or decision names within one recipe
- cross-recipe or otherwise unresolved signal/projection dependencies

The Go loader, Python CLI validator, and dashboard editor enforce the same
local-namespace contract.

## Compatibility and User-Visible Changes

- Existing single-profile files keep their routing behavior; their flat profile
  is exposed internally as `default` even when configs are built
  programmatically rather than through the canonical YAML loader.
- `routing.strategy` and `recipes[].routing.strategy` accept `priority` or
  `confidence`; named recipes inherit the router default when omitted.
- Configurations that previously relied on a named recipe referencing a signal
  or projection from another profile now fail startup validation and must copy
  the definition into that recipe.
- Duplicate local names across recipes, previously rejected, are now valid.
- Concrete backend requests no longer inherit the default recipe's signal,
  plugin, cache, learning, or session-routing state.
- Responses expose `x-vsr-selected-recipe`. Replay APIs and Insights expose a
  recipe field/filter, and metrics/state keys distinguish recipe identity.

## Completion Checklist

- [x] canonical `entrypoints`/`recipes` schema, normalization, export, and
      default-profile compatibility bridge
- [x] local-namespace validation in Go and the CLI
- [x] entrypoint resolution before signal evaluation and concrete passthrough
- [x] per-recipe classifiers, decisions, strategies, selectors, and plugins
- [x] recipe-scoped cache, replay, learning/session state, metrics, and handoff
      lookup state
- [x] `/v1/models` entrypoint metadata and `x-vsr-selected-recipe`
- [x] dashboard recipe strategy editing and recipe-filtered Insights
- [x] unit, contract, and E2E coverage, including different definitions with
      the same local signal name
- [x] reference config and user documentation
- [ ] multi-profile DSL syntax/round-trip; this is a separate language-design
      extension because the current DSL compiles one routing profile per file

## Operating Rules

- `config.go` stays a schema table; recipe contracts live in `recipes.go`,
  canonical normalization in `canonical_recipes.go`, and scoped validation in
  `validator_recipe_scope.go`.
- `processor_req_body.go` stays an orchestrator; model-to-recipe resolution
  lives in `req_filter_entrypoint.go`.
- The flat `Signals`, `Projections`, and `Decisions` fields mirror only the
  default recipe for backward compatibility. Whole-config discovery uses
  explicit recipe-aware helpers; request paths use `ConfigForRecipe` and the
  selected recipe object.
- Internal state uses `(recipe, local name)` identity. Public surfaces expose
  the two fields separately instead of leaking delimiter-encoded keys.

## Related Docs

- Issue #2331 and follow-up #2354
- `website/docs/tutorials/global/entrypoints-and-recipes.md`
- `docs/agent/architecture-guardrails.md`
- `src/semantic-router/pkg/config/AGENTS.md`
- `src/semantic-router/pkg/extproc/AGENTS.md`
