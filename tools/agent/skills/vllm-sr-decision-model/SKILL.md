---
name: vllm-sr-decision-model
description: Choose and verify the decision model behind a vLLM Semantic Router recipe. Use when a route depends on model-judged signals, a decision answer is unknown, or a contributor needs evidence that separates model-runtime health from routing policy and backend delivery.
---

# Verify the decision model

The decision model answers Router judgments; an answer-model backend generates
the user's response. A process health check alone proves neither a loaded
decision model nor a correct route. Start with the active entrypoint, recipe,
configuration and one representative request. Keep this workflow read-only
until a reviewed configuration or runtime change is requested.

## 1. Resolve the model each question uses

Read the active canonical configuration (`vllm-sr config get --format yaml`)
and identify the public entrypoint's recipe. List the recipe's model-backed
application signals, custom `systemone` questions, System One selector and
model-backed plugins that the request can exercise. Request facts such as
`context` and `input_modality` do not themselves ask a decision model.

Resolve each question's deployment in this order: rule `deployment`, recipe
`model_bindings` or global signal-type binding, then
`global.model_catalog.system.decision_model` (`primary` by default). A custom
`systemone` rule or selector uses its rule deployment or the default. Embeddings
are separate recall resources. Check the exact precedence and span requirements
in [Choose a model for a signal](https://vllm-sr.ai/docs/installation/choose-a-model-for-a-signal).
Record the deployment name, provider (`model_runtime` or `systemone`),
artifact/model, device, question type and any explicit predicate. A deployment
declared but unused need not be running. The installed schema and
[model catalog](https://vllm-sr.ai/docs/installation/model-catalog) are the
authority when the configuration differs from an example.

For a new model choice, use the [supported model and hardware table](https://vllm-sr.ai/docs/model-runtime/get-started/choose-a-model#decision-models).
Keep `vllm-sr/vela3-fast` on CPU as the starting point unless the workload or
measurements require another choice. Check that a replacement answers every
question and any required PII or hallucination spans. A model without a
calibrated preset threshold can change routing even when it loads; record
`uncalibrated_preset` and the rule's predicate before proposing a change.

## 2. Prove readiness and one answer

For a local stack, use the [Serve skill](https://vllm-sr.ai/install/agent/vllm-sr-serve/SKILL.md)
for supported Router or Engine startup. For an attached or remote runtime,
follow [Connect the Router](https://vllm-sr.ai/docs/model-runtime/deploy/connect-the-router).
An existing stack is inspected rather than restarted.

```bash
vllm-sr status
curl -fsS "$ROUTER_ORIGIN/api/v1/instance"
curl -fsS "$ROUTER_ORIGIN/api/v1/inventory/model-runtime"
```

The inventory must show each *used local* deployment with `state: ready`,
`ready: true`, the intended artifact and device. `/health` can answer while
`/ready` is 503 during startup; on reload, `/ready` can stay 200 while the new
configuration is pending. Check `/startup-status` and config activation when
they matter. An attached runtime also exposes `GET /v1/models`; a remote
`provider: systemone` deployment may not appear in the Router's local
inventory. Use its model card and a bounded request to its System One endpoint
when authorized; do not copy an API key into the report. See
[Readiness and monitoring](https://vllm-sr.ai/docs/model-runtime/deploy/readiness-and-monitoring).

If Engine mode or an allowed Router listener exposes the selected model, ask
one question through `POST /v1/systemone` and require the expected answer field,
not merely HTTP 200. Use the exact selected model, question type and bounded
input from the recipe. The [System One API](https://vllm-sr.ai/docs/system-one/use/api)
defines the request and response. An unexposed model can still be tested through
Router preview; do not broaden a listener allowlist just for diagnosis.

## 3. Trace a real routing request

Use the [route verification procedure](https://vllm-sr.ai/install/agent/vllm-sr/references/route-verification.md)
to discover the management origin and the actual public entrypoint. Preserve
the request's messages and tool payloads with `--request-file` when those can
affect the signals. For a simple text case:

```bash
vllm-sr route preview --endpoint "$ROUTER_ORIGIN" --model "$ENTRYPOINT" \
  --prompt 'Define a readiness probe in one sentence.' --trace --json > preview.json
```

Read `eval_trace` for the question's signal state, answer/predicate, unknowns,
`signal_error`, decision and algorithm; read `signal_errors`,
`selection_status`, `routing_decision` and `selected_model`. `selected` is a
resolved preview. `execution_required` means the final model still depends on
execution. An unknown follows that rule's `on_unknown` policy; a lower-priority
route or the System One selector's first-model fallback can be expected policy,
but the unknown's cause still needs investigation. A preview proves selection,
not that the answer backend delivered a response.

When a full route is in scope, run a bounded `vllm-sr route probe` from the
same request and compare its selected recipe, decision and model headers to
the preview. A backend timeout, empty answer or `finish_reason: length` is a
delivery failure even if the decision model answered correctly. Preserve both
receipts and the config version; do not change the probe until it passes.

## 4. Classify and hand off

| Evidence | Likely owner and next check |
| --- | --- |
| Local deployment loading, failed or wrong device/artifact | Model runtime: startup status, worker logs, supported hardware and model card |
| Remote/attached call fails, times out or lacks a required capability | Runtime/provider: model card, endpoint, auth environment and request budget |
| Model answers but signal stays unknown or a different rule wins | Recipe/config: binding, question type, predicate, `on_unknown`, priority and trace |
| Preview selects the intended route but probe fails | Backend delivery: listener, upstream endpoint, response body and public headers |

Report the config version/hash, entrypoint and recipe, request fixture or safe
digest, resolved question-to-deployment map, model artifact/revision/device,
inventory or remote card evidence, direct answer if available, traced preview,
probe receipt if run, and the first failing boundary. Redact private request
text, credentials and backend hostnames before sharing. For a candidate signal
or threshold change, hand the fixed case to
[routing calibration](https://github.com/vllm-project/semantic-router/blob/main/tools/agent/skills/routing-calibration/SKILL.md)
and the existing [recipe tuning guide](https://vllm-sr.ai/install/agent/vllm-sr/references/recipe-tuning.md).
Do not tune on the holdout or promote a configuration from one probe.

For a maintained example, inspect the
[accuracy recipe](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/accuracy/README.md):
`accuracy_mode` is a Choice and `accuracy_difficulty` a Score; both use its
declared decision deployment unless a rule overrides it. Its request-fact
`accuracy_long_context` needs no model answer. A missing answer to either
model-backed question keeps a workstreams request out of the workflow rule;
the traced preview, rather than the final model name alone, explains why.
