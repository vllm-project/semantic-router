# Maintained routing recipes

`config/recipes/` contains complete, executable use-case deliveries. Directory
names describe the user outcome; implementation names such as Router Flow,
SAARS, or MMLU belong inside the recipe documentation rather than in the
catalog taxonomy.

## Delivery contract

Every child directory has the same four files:

- `config.yaml` — canonical v0.3 runtime configuration.
- `recipe.dsl` — reviewable routing policy that compiles back into the same
  dynamic routing surface.
- `probes.yaml` — backend-independent `/api/v1/eval` correctness probes.
- `README.md` — intended use, routing policy, tradeoffs, and validation steps.

The repository contract tests reject incomplete directories, invalid YAML or
DSL, YAML/DSL drift, missing decision reachability, stale aliases, and loss of
YAML-only decision adaptation policy during DSL merge.

Probe manifests use `schema_version: v1` and are validated against
`tools/agent/schemas/recipe-probes-v1.schema.json`. The conformance inventory
discovers every immediate child directory automatically. Adding a recipe
therefore adds it to static and live CI without a workflow or Go allowlist
change.

See [CONFORMANCE.md](CONFORMANCE.md) for the short contributor checklist,
coverage tiers, tag conventions, and local commands.

Single-profile recipes expose their `routing` block through the default
`global.router.auto_model_names` entrypoint. Multi-profile configurations can
disable that default and expose named `entrypoints` instead. Conformance counts
and exercises both forms, so a default auto alias is not reported as zero
entrypoints.

## Catalog

| Use case | Purpose |
| --- | --- |
| [`accuracy`](accuracy/README.md) | Spend bounded multi-model orchestration only where it has an expected quality benefit; keep long context single-model. |
| [`agent`](agent/README.md) | Route agent, coding, specialist, privacy, and security work across local and frontier lanes. |
| [`balance`](balance/README.md) | General-purpose quality, latency, and cost balance for a single default routing profile. |
| [`feedback`](feedback/README.md) | Recover from corrections, repeated dissatisfaction, failed code, and verification requests. |
| [`knowledge`](knowledge/README.md) | Use KB evidence to decide whether a knowledge-domain question merits frontier escalation. |
| [`multi-objective`](multi-objective/README.md) | Expose isolated balanced, speed, cost, accuracy, and privacy recipes as request-facing entrypoints. |
| [`privacy`](privacy/README.md) | Keep sensitive, suspicious, and private-context traffic on policy-compatible models. |

`bounded-candidate-iteration.dsl` and other syntax-only demonstrations are not
deployable recipes. Their behavior is covered by DSL unit tests instead of
being mixed into this catalog.

## Maintained acceptance baseline

The blocking August 2026 baseline covers 275 base probes, 58 decisions, and 11
recipe-entrypoint bindings across all seven recipes. Decision, entrypoint,
fallback, algorithm, and plugin coverage is complete; signal and projection
assertions use checked-in per-recipe ratchets that cannot decrease:

- Accuracy: 13 probes, 4 decisions.
- Agent: 27 probes, 11 decisions.
- Balance: 57 probes, 14 decisions.
- Feedback: 23 probes, 7 decisions.
- Knowledge: 15 probes, 2 decisions.
- Privacy: 20 probes, 4 decisions.
- Multi-objective: 120 probes, 16 decisions, 5 named entrypoints.

The live CPU gate requires router readiness and exact `/api/v1/eval?trace=true`
results. Framing expansion, upstream generation, GPU parity, and latency SLOs
are T4 reporting concerns rather than PR acceptance criteria.

## Validate a recipe

From the repository root:

```bash
make recipe-conformance-static

vllm-sr validate --config config/recipes/<use-case>/config.yaml

(cd src/semantic-router && \
  go run ./cmd/dsl validate ../../config/recipes/<use-case>/recipe.dsl)

(cd src/semantic-router && \
  go run ./cmd/dsl compile \
    --base ../../config/recipes/<use-case>/config.yaml \
    -o /tmp/<use-case>.yaml \
    ../../config/recipes/<use-case>/recipe.dsl)
```

Run the backend-independent calibration suite against a live router:

```bash
make recipe-conformance-eval \
  RECIPE_CONFORMANCE_RECIPE=<use-case> \
  RECIPE_CONFORMANCE_ROUTER_URL=http://127.0.0.1:8080

# Build the local CPU image and evaluate every maintained recipe.
make recipe-conformance-live-cpu-all
```

Pull requests that touch recipes or their router semantics run the base probes
through the reusable Recipe Conformance CI domain. Framing/whitespace
expansion, real generation, GPU execution, and timing baselines remain
scheduled or manual workloads rather than PR blockers.

Each CI run publishes the coverage matrix in the GitHub Actions job summary and
uploads a `recipe-conformance-report` artifact for 30 days. The consolidated
artifact contains the inventory, per-recipe Eval reports, summaries, and
failure logs, including partial results when a live shard fails. Its inventory
lists configured, asserted, and uncovered signals, projections, algorithms,
and plugins; live summaries include the T3 robustness pass-rate receipts.

The multi-objective profile additionally checks requested model, selected
recipe, decision, algorithm, plugins, signal evidence, multilingual variants,
multi-turn/tool shapes, and long-context boundaries. Every maintained manifest
also declares bounded concurrency, so the same report records end-to-end p50,
p95, and p99 Eval latency, throughput, and transport errors alongside routing
accuracy.
