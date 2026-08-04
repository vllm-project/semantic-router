# Multi-objective routing recipe

This is the complete entrypoints + recipes example. Clients choose a stable
request-facing model ID to select an isolated objective:

| Entrypoint | Recipe objective |
| --- | --- |
| `vllm-sr/mom-balanced-v1` | Adaptive quality, speed, and efficiency. |
| `vllm-sr/mom-flash-v1` | Lowest practical latency with a bounded heavy lane. |
| `vllm-sr/mom-economy-v1` | Local-first cost control. |
| `vllm-sr/mom-frontier-v1` | Direct frontier, ReMoM, Fusion, or Router Flow for accuracy. |
| `vllm-sr/mom-private-v1` | Local-only private and suspicious traffic handling. |

Signals, projections, decisions, algorithms, and decision plugins are owned by
one recipe and cannot match another recipe. Provider bindings, model cards, and
runtime services remain shared infrastructure.

## DSL and evaluation

`recipe.dsl` uses first-class `ENTRYPOINT` and `RECIPE` scopes. Compiling it
over `config.yaml` reproduces the same five mappings and five isolated routing
programs. The probe suite contains 100 multilingual, negative, collision,
multi-turn, tool-shape, PII/jailbreak, and long-input cases across all 15
decisions.

```bash
vllm-sr validate --config config/recipes/multi-objective/config.yaml
(cd src/semantic-router && go run ./cmd/dsl validate ../../config/recipes/multi-objective/recipe.dsl)
python tools/agent/scripts/router_calibration_loop.py \
  eval \
  --router-url http://127.0.0.1:8080 \
  --probes config/recipes/multi-objective/probes.yaml
```

The manifest runs these 100 cases with bounded concurrency and includes
end-to-end latency percentiles, throughput, and error count in the JSON report.
This exercises recipe isolation under load without invoking an inference
backend.

Use the management API to validate, create, update, or delete one named recipe.
Updates require `If-Match`; the default recipe and recipes referenced by an
entrypoint are protected from unsafe deletion.
