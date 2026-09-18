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

`config/recipes/built-in/` is the reserved versioned catalog container, not a
Recipe. Live CI discovers standalone recipes and every bundle under
`built-in/latest/` as separate sources. Their static inventories, live shards,
and reports have distinct namespaces. Immutable release snapshots retain their
catalog/release validation and do not multiply the live CPU matrix.

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
   catalog in [README.md](README.md).
2. Set `schema_version: vllm-sr/recipe-metadata/v1` in `metadata.yaml`. Its
   stable `id` must match the directory name; declare a semantic `version`,
   authorship, license, tags, and at least the source link. Bump the version for
   a stable semantic release. The mutable latest channel may replace the current
   content for an existing `id` + `version`; stores advance that reference while
   retaining every immutable `recipe_digest` object and any active old digest.
3. Set `schema_version: v1`, matching `name`, and correct `routing_assets` in
   `probes.yaml`.
4. Add at least one probe for every decision and every request-facing model
   entrypoint. Default recipes use `global.router.auto_model_names`; named
   recipes set `model` and `expected_recipe`.
5. Declare `expected_algorithm` for every decision and `expected_plugins` when
   the decision configures plugins. A model-free `fast_response` decision has no
   algorithm: omit `expected_algorithm`, assert the plugin, and set
   `expected_selection_status: not_required` on its probe group.
6. Use `expected_signals` or `forbidden_signals` for signal and projection
   evidence that the prompt is intended to exercise.
7. Preserve or raise the checked-in `coverage` minima. New routing surfaces
   must not reduce an existing percentage or robustness count.

Each variant must contain exactly one of `query` or `messages`. Add `tools` when
tool shape is part of the contract.

### Raw signal values

Use `expected_signal_values` when a projection consumes a raw score even below
the signal's match threshold. Keys are exact runtime keys from the Preview
response's top-level `signal_values`, rather than the plural families in
`expected_signals`:

```yaml
expected_signal_values:
  embedding:informational: {gte: -1, lte: 1}
  structure:question_count: {gte: 2}
```

Each interval needs at least one inclusive `gte` or `lte` bound. Bounds must be
finite numbers, not booleans, with `gte <= lte` when both are present. A group
provides defaults; a variant replaces the whole mapping when specified, and
`{}` clears it. Missing, nonnumeric or nonfinite values and corresponding
`signal_errors` fail the assertion. An observed score below the match threshold
can pass its interval without satisfying `expected_signals`; existing expected,
forbidden, projection, decision and trace assertions remain independent.

Scores use the evaluator's own units and are **not necessarily probabilities**.
For example, cosine similarities may be negative, and structure values may be
counts. A broad valid range proves observation, not semantic correctness; use
decision and projection assertions to verify the intended policy. Static
coverage resolves each value to a configured rule in that probe's recipe
(`embedding:name` covers `embeddings:name`); unknown rules are rejected. Live
reports retain observed values and interval errors in both evaluation scopes.

## Expected model selection

Selection checks use the expected algorithm's normal preview statuses by default.
An optional `expected_selection_status` on a decision declares one exact expected
status; a variant can override it for a specific boundary case. Supported values
are `selected`, `planned_final`, `fallback`, `execution_required`, `unavailable`,
and `failed`.

Use an explicit `unavailable` expectation in a deployment-specific probe when
its input exceeds every configured backend's context capacity. Such a negative
probe must return no `selected_model` and a nonempty `selection_reason`. It still
checks the requested model, recipe, decision, algorithm, signals, and trace.
Missing selection methods are allowed for `unavailable` and `failed`, since
eligibility checks can reject the request before a selector runs. Other probes
retain their normal positive selection requirements.

Record these deployment assumptions in the adapted probe's notes. Keep the
maintained recipe's original probes and report their outcomes separately; do
not silently turn an unexpected failure into an expected negative.

## Synthetic request fixtures

Keep large request boundaries declarative and reviewable:

- Use `padding` for query-shaped probes.
- Use `generated_text` with zero-based `message_index`, `content_index`, and an
  exact `target_text_bytes` for message-shaped probes. The target counts all
  UTF-8 text bytes in the selected message, including its explicit text parts.
  The inserted part repeats `character` (default `x`), or a nonempty `text`
  pattern; these two fields are mutually exclusive. `text` allows printable
  ASCII, tabs, and line breaks. Its final repetition is truncated to meet the
  exact target. Use representative records for meaningful long-context input;
  the request's JSON byte limit also accounts for escaped characters.
- Use a named `image_fixture` for repeated binary media. Declare its human
  description, media type, canonical base64 payload, and SHA-256 once under
  `fixtures.images`; materialization constructs the data URI. Admission reads
  the encoded static-image container and requires it to match the declared
  PNG, JPEG, GIF, or WebP media type. Each image is limited to 8192 pixels per
  side and 16,777,216 canvas pixels in addition to the 4 MiB encoded-payload
  limit; animated containers are rejected.

The evaluator materializes these fields only in memory. Reports retain the
compact request specification, materialized text/JSON byte counts, and a SHA-256
receipt. The raw Preview response is also preserved, including any text returned
by the server; keep generated reports out of recipe packages. Do not check in repeated
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

# Build the CPU router and run every CPU-compatible maintained recipe.
make recipe-conformance-live-cpu-all

# Run the current built-in bundle through the same live pipeline.
make recipe-conformance-live-cpu \
  RECIPE_CONFORMANCE_RECIPES_ROOT=config/recipes/built-in/latest \
  RECIPE_CONFORMANCE_RECIPES=mom-v1 \
  RECIPE_CONFORMANCE_REPORT_DIR=.agent-harness/recipe-conformance/built-in/latest
```

Portable built-in bundles are composed with the CLI's
`initialize_builtin_recipe` binding implementation. The runner binds every
recipe to its catalog entrypoint and supplies explicit Preview-only backend
fixtures, including capabilities, context limits, pricing, and synthetic
benchmark records for model selection. It preserves every authored decision,
signal, projection, algorithm, plugin, and probe. The current MoM bundle runs
all five entrypoints and all 315 authored probes; counts are checked against the
discovered inventory rather than used to select a smaller sample. These fixtures
do not measure backend LLM quality or call external providers.

Each invocation owns an isolated stack and runtime directory under
`.agent-harness/recipe-conformance-runtime/`; logs and reports are collected
before that stack is stopped. Configured bearer credentials are supplied to
startup, readiness, and evaluation requests. Shared downloaded models live in
the ignored `models/` cache. Use `VLLM_SR_PORT_OFFSET` to separate host ports when
running concurrent stacks. Missing reports, zero probes, and incomplete probe
execution fail the selected CPU gate.

### Policy and deployment scope

Live evaluation uses routing Preview. It runs the configured router classifiers
and embeddings, evaluates decisions and the model selector, and returns the
trace without calling a backend LLM. The default `--scope deployment` checks
both policy evidence and the expected selection result against the actual
assigned models. It does not certify backend answer quality or multi-model
execution.

For reusable policy evaluation across different model capacities, explicitly use
`--scope policy`:

```bash
python tools/calibration/recipe/recipe_conformance.py eval \
  --recipe <name> --router-url http://127.0.0.1:8080 --scope policy

python tools/calibration/recipe/router_calibration_loop.py eval \
  --probes path/to/probes.yaml --router-url http://127.0.0.1:8080 --scope policy
```

Both scopes retain the same decision, recipe, algorithm, plugin, signal,
projection, trace, and request-error checks. Policy scope can pass when no
assigned model fits, but still requires a valid selection response. Reports
show policy and deployment results separately, selection-status counts, and
each returned reason. A policy pass is not a deployment pass. Keep
`expected_algorithm`; omitting it is not a policy-only mode.

Explicit group-level `expected_selection_status: unavailable` and
`not_required` remain strict in both scopes: the status must match, no final
model may be fabricated, and a reason is required. `not_required` also requires
the `fast_response` selection method. Use `unavailable` only when the test binds
real assignment limits; a long request alone does not imply that every possible
model pool must reject it.

For deployment qualification, bind the intended recipe entrypoints and real
model metadata, then run deployment scope and backend execution checks. Input
capacity includes retained messages and media plus the effective output limit.
The router's neutral token estimate is not an exact tokenizer or media bound.
Router-classifier input limits are separate: an inference or HTTP failure fails
both scopes and cannot be reclassified as a successful capacity rejection.

CI publishes coverage in the job summary and uploads the consolidated
`recipe-conformance-report` artifact for 30 days. `inventory.json` contains the
configured, asserted, and uncovered surfaces; per-recipe `eval-report.json`
contains exact live results and T3 receipts.

### Explicit hardware requirements

Static checks always include every maintained recipe. Live CPU planning derives
hardware requirements from explicit `global.model_catalog.deployments.*.device`
values. Recipes with non-CPU devices stay in the inventory and are listed with
their required devices in `cpu-eligibility.json` and the consolidated report;
they do not count as CPU runtime passes. The CPU runner rejects an incompatible
explicit selection before stopping or starting any containers. It never rewrites
GPU bindings to CPU.

Use `list --platform cpu` to list compatible CPU recipes. For an AMD recipe,
start its unchanged configuration with the matching image and run
`make recipe-conformance-eval` against that active Router. Hardware-specific
execution remains an explicit runtime qualification.
