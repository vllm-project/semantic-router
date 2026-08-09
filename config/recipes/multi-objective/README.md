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

For an AMD ROCm deployment that assigns all eight MI300X GPUs explicit serving
or routing responsibilities, see
[`website/blog/2026-08-05-multi-objective-mom-on-amd-developer-cloud.md`](../../../website/blog/2026-08-05-multi-objective-mom-on-amd-developer-cloud.md).

## Maintained capability assumptions

Request-facing model IDs are routing aliases, not capability declarations. The
maintained provider entries resolve to a measured local pool:

- `Qwen3.5-9B` has independent primary and replica lanes for cost/load-aware
  economy routing; the primary also owns the privacy alias.
- `Qwen3.6-35B-A3B-FP8` owns the flash lane.
- `Qwen3.6-27B` owns coding, planning, and structured-output work.
- `Gemma 4-26B-A4B-it` provides the fastest measured balanced lane and
  architecture diversity.
- `Qwen3.5-122B-A10B-FP8` remains the stable direct model and orchestration
  judge.
  Its vLLM service must use `--tool-call-parser qwen3_xml`; the model emits
  Qwen's tagged function/parameter protocol rather than the Qwen3 Coder JSON
  envelope.
- `DeepSeek-V4-Flash-0731` contributes a current, MIT-licensed analysis path
  behind the stable judge; it is not the direct default because the MI300X
  correctness gate scored below the stable models.

Only Qwen backends use the Qwen3 reasoning request contract. The configured 32K
limits for the smaller tiers are deployment limits chosen for predictable
single-GPU capacity, not the models' architectural maximum.

The balanced effort score gives explicit reasoning markers a `0.85`
contribution and limits deterministic terse markers to `-0.10`. Explicit
reasoning therefore remains above the `0.30` deliberate threshold even in
concise requests, while a dedicated negated-reasoning suppressor keeps “do not
analyze” requests direct. The probe manifest covers this boundary with
multilingual concise/reasoning conflicts and negation negatives.

Response-side hallucination plugins are intentionally absent from maintained
routes because this recipe does not include a supported detector runtime.
Evidence-sensitive routes require explicit verification language and use
confidence-based selection without letting a learned fact-check false positive
escalate an ordinary request.

Frontier orchestration is intentionally explicit and reachable with natural
requests: ordinary conversation stays direct; “verify with evidence” selects
confidence; “ask multiple models” selects Fusion; “think deeply and reason
through alternative paths” selects ReMoM; and investigate/implement/validate
requests select Workflow. Confidence evaluates candidates non-streaming so
log-probability thresholds are real, then publishes only the selected answer
as a client-compatible stream. Dashboard surfaces turn stable internal IDs
such as `unified_frontier_verified_answer` into readable labels such as
“Frontier Verified Answer” without changing replay or API identity.

Looper tool behavior is explicit: Workflow owns stateful tool interruption and
resume; Fusion and Confidence may publish only the final selected model's tool
call; ReMoM, Ratings, RL-driven, and fallback multi-model aggregation strip
callable schemas from private candidate requests because they cannot safely
merge multiple independent tool trajectories. A deep request with tools
attached can still use ReMoM, while a request that explicitly says to call a
tool and then synthesize is routed to Workflow.

## DSL and evaluation

`recipe.dsl` uses first-class `ENTRYPOINT` and `RECIPE` scopes. Compiling it
over `config.yaml` reproduces the same five mappings and five isolated routing
programs. The probe suite contains 120 multilingual, negative, collision,
multi-turn, tool-shape, PII/jailbreak, and long-input cases across all 16
decisions.

```bash
vllm-sr validate --config config/recipes/multi-objective/config.yaml
(cd src/semantic-router && go run ./cmd/dsl validate ../../config/recipes/multi-objective/recipe.dsl)
python tools/agent/scripts/router_calibration_loop.py \
  eval \
  --router-url http://127.0.0.1:8080 \
  --probes config/recipes/multi-objective/probes.yaml
```

The manifest runs these 120 cases with bounded concurrency and includes
end-to-end latency percentiles, throughput, and error count in the JSON report.
This exercises recipe isolation under load without invoking an inference
backend.

The AMD pool acceptance additionally expands these probes into 600 deterministic
framing/whitespace stress cases and runs 126 real generated requests across
English, Chinese, Spanish, French, Japanese, and German.

Use the management API to validate, create, update, or delete one named recipe.
Updates require `If-Match`; the default recipe and recipes referenced by an
entrypoint are protected from unsafe deletion.
