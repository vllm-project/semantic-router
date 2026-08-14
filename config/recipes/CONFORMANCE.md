# Adding recipe conformance tests

Every maintained recipe directory is discovered automatically. It must contain
exactly:

- `config.yaml`
- `recipe.dsl`
- `probes.yaml`
- `README.md`

The router may create a temporary `.vllm-sr/` directory while running; it is not
part of the maintained contract.

## Add or change a recipe

1. Add the four files under `config/recipes/<name>/` and add the recipe to the
   catalog in [README.md](README.md).
2. Set `schema_version: v1`, matching `name`, and correct `routing_assets` in
   `probes.yaml`.
3. Add at least one probe for every decision and every request-facing model
   entrypoint. Default recipes use `global.router.auto_model_names`; named
   recipes set `model` and `expected_recipe`.
4. Declare `expected_algorithm` for every decision and `expected_plugins` when
   the decision configures plugins.
5. Use `expected_signals` or `forbidden_signals` for signal and projection
   evidence that the prompt is intended to exercise.
6. Preserve or raise the checked-in `coverage` minima. New routing surfaces
   must not reduce an existing percentage or robustness count.

Each variant must contain exactly one of `query` or `messages`. Add `tools` when
tool shape is part of the contract.

## Coverage tiers

- **T0 — structural:** four files, JSON Schema, canonical config, and YAML/DSL
  symmetry. Blocking.
- **T1 — reachability:** every decision, fallback, entrypoint, and required
  request shape. Blocking at 100%.
- **T2 — routing fidelity:** recipe, decision, alias, algorithm, plugins,
  signals, projections, and exact EvalTrace. Blocking and ratcheted.
- **T3 — robustness:** minimum language, negative, collision, fallback, stress,
  tool, and multi-turn counts plus their live pass rates. Blocking.
- **T4 — stress/SLO:** framing expansion, generation, GPU, and latency
  baselines. Nightly or manual reporting only.

Use these tag forms for T3 coverage:

- `language:<iso-code>`
- `class:negative` or `negative`
- `collision`
- `fallback`
- `stress:<kind>`
- `shape:tools`
- `shape:messages` or `multi-turn`

Existing short language tags such as `zh` remain recognized, but new probes
should use `language:zh`.

## Run locally

```bash
# Schema, coverage ratchets, canonical config/DSL, and decision contracts.
make recipe-conformance-static

# Evaluate one recipe against an already running router.
make recipe-conformance-eval \
  RECIPE_CONFORMANCE_RECIPE=<name> \
  RECIPE_CONFORMANCE_ROUTER_URL=http://127.0.0.1:8080

# Build the CPU router and run every maintained recipe.
make recipe-conformance-live-cpu-all
```

CI publishes coverage in the job summary and uploads the consolidated
`recipe-conformance-report` artifact for 30 days. `inventory.json` contains the
configured, asserted, and uncovered surfaces; per-recipe `eval-report.json`
contains exact live results and T3 receipts.
