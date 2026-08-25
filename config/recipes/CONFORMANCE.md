# Recipe Authoring and Conformance

Every standalone maintained Recipe directory is discovered automatically. It
must contain exactly:

- `config.yaml`
- `metadata.yaml`
- `recipe.dsl`
- `probes.yaml`
- `README.md`

The router may create a temporary `.vllm-sr/` directory while running; it is not
part of the maintained contract.

`config/recipes/built-in/` is the reserved versioned distribution container,
not a Recipe. The default standalone inventory skips that directory and CI runs the
same five-file conformance checks separately for every bundle under
`built-in/latest/` and every release snapshot.

## Write the Model Card

`README.md` is a reader-facing Model Card, not a test log or deployment
runbook. Use these sections, in this order:

1. `Overview`
2. `Model details`
3. `Intended use`
4. `Routing behavior`
5. `Requirements`
6. `Data handling and safety`
7. `Quick start`
8. `Evaluation`
9. `Limitations`
10. `References`

Describe stable, user-visible behavior in the present tense. A Model Card
should answer what the recipe does, who should use it, what backends and routing
assets it needs, how it handles data, and where it can fail or trade quality for
cost or latency.

Keep route examples short and representative. The complete scenario inventory
belongs in `probes.yaml`; current counts and pass rates belong in generated
conformance reports. Do not include CI receipts, tuning phases, branch state,
private qualification hosts, private paths, or promotion checklists. Put
authoring mechanics here and release operations in the maintainer guide.

## Add or change a recipe

1. Add the five files under `config/recipes/<name>/` and add the recipe to the
   index in [README.md](README.md).
2. Set `schema_version: vllm-sr/recipe-metadata/v1` in `metadata.yaml`. Its
   stable `id` must match the directory name; declare a semantic `version`,
   authorship, license, tags, and at least the source link. Bump the version for
   a stable semantic release. The mutable latest channel may replace the current
   content for an existing `id` + `version`; stores advance that reference while
   retaining every immutable `recipe_digest` object and any active old digest.
3. Set `schema_version: v1`, matching `name`, and correct `routing_assets` in
   `probes.yaml`.
4. Add at least one probe for every decision and every request-facing model
   Entrypoint. Every request-facing alias belongs to `entrypoints[].model_names`;
   named probes set `model` and `expected_recipe`.
5. Declare `expected_algorithm` for every decision and `expected_plugins` when
   the decision configures plugins.
6. Use `expected_signals` or `forbidden_signals` for signal and projection
   evidence that the prompt is intended to exercise.
7. Preserve or raise the checked-in `coverage` minima. New routing surfaces
   must not reduce an existing percentage or robustness count.

Each variant must contain exactly one of `query` or `messages`. Add `tools` when
tool shape is part of the contract.

## Synthetic request fixtures

Keep large request boundaries declarative and reviewable:

- Use `padding` for query-shaped probes.
- Use `generated_text` with zero-based `message_index`, `content_index`, and an
  exact `target_text_bytes` for message-shaped probes.
- Use a named `image_fixture` for repeated binary media. Declare its human
  description, media type, canonical base64 payload, and SHA-256 once under
  `fixtures.images`; materialization constructs the data URI. Admission reads
  the encoded static-image container and requires it to match the declared
  PNG, JPEG, GIF, or WebP media type. Each image is limited to 8192 pixels per
  side and 16,777,216 canvas pixels in addition to the 4 MiB encoded-payload
  limit; animated containers are rejected.

The evaluator materializes these fields only in memory. Reports retain the
compact specification, materialized text/JSON byte counts, and a SHA-256
receipt—not expanded filler or fixture binary. Do not check in repeated
content objects, duplicate data URIs, YAML anchors, aliases, merge keys, or
explicit tags; conformance and package admission reject YAML indirection.

Dashboard Validate sends the same materialized messages to Router Eval without
rendering binary payloads. Dashboard Run preserves that exact request while
Playground presents text parts as text and verified inline image parts as real
images. Edit is offered only when the terminal user message has one editable
text part; changing it must retain every non-text part unchanged.

## Coverage tiers

- **T0 — structural:** five files, metadata and probe JSON Schemas, canonical
  config, and YAML/DSL
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

# A distribution containing several Recipes has no bundled Models or Entrypoints.
# Bind each Recipe to the published Entrypoint used by this live environment.
make recipe-conformance-eval \
  RECIPE_CONFORMANCE_RECIPE=mom-v1 \
  RECIPE_CONFORMANCE_ROUTER_URL=http://127.0.0.1:8080 \
  RECIPE_CONFORMANCE_ENTRYPOINTS="balance=vllm-sr/blend speed=vllm-sr/lite cost=vllm-sr/cost accuracy=vllm-sr/ultra vault=vllm-sr/vault"

# Build the CPU router and run every maintained recipe.
make recipe-conformance-live-cpu-all
```

CI publishes coverage in the job summary and uploads the consolidated
`recipe-conformance-report` artifact for 30 days. `inventory.json` contains the
configured, asserted, and uncovered surfaces; per-recipe `eval-report.json`
contains exact live results and T3 receipts.
