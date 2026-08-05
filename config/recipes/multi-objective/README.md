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

For a compact AMD ROCm deployment that exposes one physical model through
multiple logical aliases, see
[`website/blog/2026-08-05-multi-objective-mom-on-amd-developer-cloud.md`](../../../website/blog/2026-08-05-multi-objective-mom-on-amd-developer-cloud.md).

## Maintained capability assumptions

Request-facing model IDs are routing aliases, not capability declarations. The
maintained provider entries resolve to one shared backend, so every alias uses
the reasoning family supported by that backend. A deployment with distinct
physical backends should set this metadata from each backend's actual request
contract.

The balanced effort score gives explicit reasoning markers a `0.46` contribution
and limits the learned terse-preference penalty to `-0.10`. Explicit reasoning
therefore remains above the `0.32` deliberate threshold even when the terse
classifier also matches, while concise-only and simple-marker probes remain in
the standard band. The probe manifest covers this boundary with multilingual
preference-conflict positives and concise negatives.

Response-side hallucination plugins are intentionally absent from maintained
routes because this recipe does not include a supported detector runtime.
Evidence-sensitive routes retain fact-check signals and confidence-based
selection without advertising unavailable response analysis.

## DSL and evaluation

`recipe.dsl` uses first-class `ENTRYPOINT` and `RECIPE` scopes. Compiling it
over `config.yaml` reproduces the same five mappings and five isolated routing
programs. The probe suite contains 102 multilingual, negative, collision,
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

The manifest runs these 102 cases with bounded concurrency and includes
end-to-end latency percentiles, throughput, and error count in the JSON report.
This exercises recipe isolation under load without invoking an inference
backend.

Use the management API to validate, create, update, or delete one named recipe.
Updates require `If-Match`; the default recipe and recipes referenced by an
entrypoint are protected from unsafe deletion.
