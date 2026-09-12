---
title: Operations and troubleshooting
description: Check readiness, limit model concurrency, and update running models.
---

Use these checks after configuring an [in-process model](in-process.md) or
[external service](external.md).

## Check startup

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
curl -fsS http://localhost:8080/startup-status
```

Validation checks the configuration. Startup then loads or connects the models
and checks the capabilities needed by enabled features. GPU kernel compilation
can make the first startup slower than later requests.

| Problem | What to check |
| --- | --- |
| Model cannot load | Complete checkpoint or ONNX files, tokenizer, labels, and mounted paths |
| Engine or device unavailable | Image and host match the selected CPU/GPU runtime |
| Label mismatch | Checkpoint label order and the rule's labels or mapping file |
| Unsupported embedding layer or dimension | Export the requested layer and match the consumer or stored index |
| Remote inference fails | Endpoint, credentials, timeout, response format, and response-size limit |
| Confidence is `null` | The result has no model score; see [Safety models](safety.md#handle-failures-and-missing-scores) |

## AMD startup problems

Use the maintained ROCm image so ORT and MIGraphX libraries match. It includes
ORT 1.22.1 / MIGraphX 2.13 and sets `MIGRAPHX_MLIR_USE_SPECIFIC_OPS=~attention`.
Keep that setting with this image; it disables MLIR attention fusion.

| Error or symptom | Action |
| --- | --- |
| `IsNaN` unsupported in the SDPA graph | Use the compatible standard `onnx/model.onnx` graph |
| Missing GPU embedding budget | Set a positive deployment `input.max_tokens`; see [Embeddings](embeddings.md#amd-gpu) |
| Model fails during GPU preparation | Check the graph and vendor libraries; requested GPU execution does not fall back to CPU |
| Unexpectedly slow first startup | Allow time for compilation and warmup of every requested embedding layer |

Unset these process variables and configure precision in the deployment instead:

```text
ORT_MIGRAPHX_FP16_ENABLE
ORT_MIGRAPHX_BF16_ENABLE
ORT_MIGRAPHX_FP8_ENABLE
ORT_MIGRAPHX_INT8_ENABLE
ORT_MIGRAPHX_MODEL_CACHE_PATH
```

Any nonempty value, including `0`, is rejected because it can override the
configured precision or compiled model. The maintained images leave them unset.

## Limit concurrent inference

For the `email-risk-cpu` deployment in [In-process models](in-process.md),
this allows two calls and a queue of eight, with a one-second queue timeout:

```yaml
global:
  model_catalog:
    admission:
      email-risk-cpu:
        max_concurrency: 2
        max_queue: 8
        queue_timeout_ms: 1000
        on_overflow: shed
```

`shed` rejects excess work, `wait` waits for a queue slot, and `fail_open`
bypasses the limit when full. `wait` requires a nonzero queue size. Omit the
admission setting for unbounded admission. Uses of the same shared model also
share its capacity; request deadlines include queue time.

## Update a running model

Put a replacement model revision in a new directory, update its configuration,
then reload through the Dashboard or your existing management workflow.
The Router prepares the replacement before activating it. Failed preparation
leaves the current configuration running; existing requests finish before
old model resources are released.

A Dashboard change may be saved but still pending. For updates that return
`202`, poll `GET /api/router/api/v1/config/hash` through the Dashboard and wait
for `active_runtime_hash` to equal the update's `generated_runtime_hash`.
A competing change returns `409` while the first update is pending.

Knowledge-base updates keep old asset versions for active readers. Old versions
are retained on disk; automatic removal is not provided. See the
[management API reference](../../api/apiserver.md) for request and response details.
