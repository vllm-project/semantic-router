---
title: Operations and troubleshooting
description: Check readiness, inspect real routing signals, and manage model updates.
---

Use this page after setting up [local models](in-process.md) or
[external services](external.md).

## Check startup

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
curl -fsS http://localhost:8080/ready
curl -fsS http://localhost:8080/startup-status
```

Validation checks configuration. Startup loads the required models, checks their
capabilities, and warms them before readiness. On AMD, first-time MIGraphX
compilation can take several minutes.

The CLI waits up to 1,800 seconds by default. Increase the budget if your measured
cold startup needs longer:

```bash
vllm-sr serve --config config.yaml --startup-timeout 7200
```

A timeout leaves containers running for inspection. Check their status and logs:

```bash
vllm-sr status
vllm-sr logs router
```

`--startup-timeout` controls readiness waiting; it does not change inference
request deadlines.

## Inspect the executed path

Route Preview runs your configured signals without calling a generation backend:

```bash
curl -fsS 'http://localhost:8080/api/v1/routing/preview?trace=true' \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","text":"Help me debug this Python program."}' \
  | jq '{decision_result, signal_confidences, signal_values, signal_errors, metrics, eval_trace}'
```

Replace `auto` with your public entrypoint name; the Vela AMD recipe uses
`vela-auto`. Inspect:

| Field | What it tells you |
| --- | --- |
| `decision_result` | Selected route and model |
| `signal_confidences`, `signal_values` | Actual classifier scores and extracted values |
| `signal_errors` | Failed or unavailable signal evaluations |
| `metrics` | Time spent evaluating the request and signals |
| `eval_trace` | How conditions produced the routing decision |

Preview does not run RAG retrieval or reranking. Test those with a real
`/v1/chat/completions` request after indexing documents. Its trace includes
`rag.rerank_candidates`, `rag.rerank_latency_seconds`, `rag.reranker_identity`,
and relevance scores. Cached RAG context may avoid a new reranker call.
See [Neural reranking](../../tutorials/plugin/rag.md#neural-reranking).

Compare warm inference latency separately from startup, retrieval, and backend
generation. Preview timing is not complete chat-response latency.

## Common problems

| Symptom | What to check |
| --- | --- |
| Model cannot load | Complete weights or ONNX files, tokenizer, labels, and mounted paths |
| Engine or device unavailable | Router image, host libraries, and GPU access match the deployment |
| Label mismatch | Checkpoint label order, rule labels, and mapping file |
| Input rejected | Deployment token budget and special tokens; [input policies](in-process.md#choose-an-input-budget) |
| Missing embedding layer or dimension | Model exports and requirements of caches, memory, or stored vectors |
| Remote inference fails | Endpoint, credentials, timeout, response format, and response-size limit |
| Preview returns 429 or 504 | Worker saturation or request timeout under `global.services.api.routing_preview` |
| Confidence is `null` | No model score was available; [failure policies](safety.md#handle-failures-and-missing-scores) |

A timed-out native Preview worker retains its slot until inference finishes.
Use a smaller input budget or reduce request concurrency before increasing
Preview deadlines.

## AMD startup problems

`rocm:N` selects the ROCm execution provider; `migraphx:N` selects MIGraphX.
A CK graph also requires `custom_ops_profile: ck_flash_attention` and its library
in the image. Use the graph and provider pair from the
[Vela AMD recipe](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/vela-amd/README.md).
Requested GPU inference rejects CPU fallback.

| Symptom | Action |
| --- | --- |
| Unsupported graph operator | Select an export compatible with the chosen execution provider |
| Missing GPU embedding budget | Set a positive deployment `input.max_tokens` |
| Slow cold startup | Allow compilation and warmup; consider the [MIGraphX compilation cache](in-process.md#advanced-migraphx-settings) |
| MIGraphX environment conflict | Remove the overrides below and configure precision/cache on the deployment |

Keep `MIGRAPHX_MLIR_USE_SPECIFIC_OPS=~attention` in the maintained MIGraphX image.
Leave the following variables unset, including values of `0`:

```text
ORT_MIGRAPHX_FP16_ENABLE
ORT_MIGRAPHX_BF16_ENABLE
ORT_MIGRAPHX_FP8_ENABLE
ORT_MIGRAPHX_INT8_ENABLE
ORT_MIGRAPHX_MODEL_CACHE_PATH
```

These process-wide overrides conflict with the deployment's explicit precision
or cache settings.

## Limit concurrent inference

For the `vela-domain` deployment in [In-process models](in-process.md), allow two
concurrent calls and a queue of eight with a one-second queue timeout:

```yaml
global:
  model_catalog:
    admission:
      vela-domain:
        max_concurrency: 2
        max_queue: 8
        queue_timeout_ms: 1000
        on_overflow: shed
```

`shed` rejects excess work, `wait` waits for a queue slot, and `fail_open`
bypasses the limit when full. `wait` requires a nonzero queue size. Omitting
admission settings leaves admission unbounded. Shared uses of a model share its
capacity, and request deadlines include queue time.

Restart the Router after changing a shared model's admission settings. A hot
reload with different settings is rejected while the active configuration
continues serving.

## Update a running model

Place the new revision in a new directory, update its deployment, and reload
through the Dashboard or management API. The Router prepares the replacement
before activating it; a failed load leaves the current configuration serving.
Existing requests finish before old model resources are released.

For a Dashboard update returning `202`, poll
`GET /api/router/api/v1/config/hash` until `active_runtime_hash` matches the
update's `generated_runtime_hash`. Another change returns `409` while an update
is pending. See the [management API reference](../../api/apiserver.md).

Changing an embedding space requires [reindexing affected documents](embeddings.md#change-a-model-without-mixing-vector-spaces).
Old knowledge-base asset versions remain on disk; remove them only after they
are no longer needed by active readers.

`serve` preserves configuration saved through the Dashboard or API. To replace
it with a local file, run:

```bash
vllm-sr serve --config config.yaml --replace-active-config
```
