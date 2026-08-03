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
DSL, and YAML/DSL drift.

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

## Validate a recipe

From the repository root:

```bash
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
python tools/agent/scripts/router_calibration_loop.py \
  eval \
  --router-url http://127.0.0.1:8080 \
  --probes config/recipes/<use-case>/probes.yaml
```

The multi-objective profile additionally checks requested model, selected
recipe, decision, algorithm, plugins, signal evidence, multilingual variants,
multi-turn/tool shapes, and long-context boundaries. Every maintained manifest
also declares bounded concurrency, so the same report records end-to-end p50,
p95, and p99 Eval latency, throughput, and transport errors alongside routing
accuracy.
