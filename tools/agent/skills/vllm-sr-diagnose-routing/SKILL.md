---
name: vllm-sr-diagnose-routing
description: Explain why a vLLM Semantic Router request selected a recipe, decision and backend, fell back, or failed to deliver. Use real route preview, probe, trace, headers and model-runtime evidence for a bounded reproduction.
---

# Diagnose a Router request

Start from the request's public entrypoint, expected outcome and observed
symptom. Use the active configuration and a bounded reproduction. Keep the
diagnosis read-only; a suggested rule or backend fix is a separate reviewed
change. The [route verification procedure](https://vllm-sr.ai/install/agent/vllm-sr/references/route-verification.md)
defines the current CLI and header contract, and the
[troubleshooting table](https://vllm-sr.ai/install/agent/vllm-sr/references/troubleshooting.md)
maps known symptoms to specific checks.

## 1. Pin the request and active state

Record the request format, public model name, relevant messages/tools or a
safe digest, expected recipe/decision/model, gateway mode, and management and
inference origins. `GET /v1/models` shows published entrypoints; a concrete
provider model bypasses recipe routing. Read `vllm-sr status`, active
`config_version`/hash, and model-runtime inventory. Process health does not
prove a signal answered or an upstream generated a response.

Use a sanitized request file when history, tools, modalities or response
constraints matter. Preserve the failed request; do not simplify it until the
failure disappears and then present the simpler case as equivalent. Keep
credentials and private endpoints out of reports.

## 2. Preview selection

Run the same request through the management endpoint, asking for evaluation
evidence without sending it to an answer-model backend:

```bash
vllm-sr route preview --endpoint "$ROUTER_ORIGIN" --model "$ENTRYPOINT" \
  --request-file request.json --trace --json > preview.json
```

Read `selection_status`, `routing_decision`, `selected_model`, `signal_errors`
and `eval_trace`. Map each evaluated signal to its answer, predicate or
unknown/error state; then follow decision priority, rule `on_unknown`,
algorithm, fallback and recipe-scoped plugin bindings. A matched decision can
still report `execution_required` when execution selects the final model.
Preview cannot prove upstream delivery, tool execution, answer quality or a
successful response guard.

If a model-backed signal is missing, use the
[decision-model skill](https://vllm-sr.ai/install/agent/vllm-sr-decision-model/SKILL.md)
to resolve its deployment and check readiness, answer type and timeout. A
local inventory entry in `ready` state is necessary for that deployment but
does not prove this request received an answer. A remote System One deployment
may not appear in local inventory.

## 3. Probe delivery and plugin effects

Send a bounded request through the same public listener and compare its
receipt with preview. Pick a token cap large enough for the expected answer.
`route probe` accepts `--prompt` or `--messages` JSON, not a request file; if
tools, modalities or other payload fields are essential, send the original
request through the listener with an approved client and preserve its public
headers and body. Mark any simpler probe as a different request.

```bash
vllm-sr route probe --config config.yaml --base-url "$INFERENCE_BASE_URL" \
  --model "$ENTRYPOINT" --prompt 'Define a readiness probe in one sentence.' \
  --max-completion-tokens 256 > probe.json
```

Read the probe's `passed`, `assertions`, status, `response.body.delivery`,
selected recipe/decision/model headers, `x-vsr-response-path`, and warnings.
The public headers are available even when the debug trace is not. The
response-path distinguishes `upstream`, `cache`, `fast_response`, `looper`,
`blocked`, `rate_limited`, `error` and `fallback`; check the actual value before
attributing an answer to a backend. A cache hit or guard block can be HTTP
200 without an upstream answer. Debug `x-vsr-trace` requires both
`x-vsr-debug: true` and an allowed `global.router.headers.debug` policy;
absence of that header is not evidence that no routing occurred. Preserve
plugin outcomes and warnings that are visible in the receipt or authorized
trace; do not infer invisible plugin execution.

`route probe` exits 2 when an assertion fails. A 200 with empty or
reasoning-only output, malformed tool arguments, or `finish_reason: length`
does not prove delivery. Compare the backend directly only after the Router
evidence identifies an upstream boundary, and do not send private prompts to
an unapproved endpoint.

## Three-case diagnostic walkthrough

Use one frozen request per case, record preview and probe receipts, and write
the differing evidence. The cases are checks to perform on a maintained
fixture or the user's stack, not pre-claimed outcomes.

| Case | Evidence needed | Conclusion and next check |
| --- | --- | --- |
| Expected route | Preview `selected` with intended recipe, decision and model; no relevant signal error; probe matching headers, `response-path: upstream`, complete answer | Route and delivery worked for this request. Quality remains unmeasured; use a benchmark for that claim. |
| Unknown or fallback | `eval_trace` shows the signal state/error and decision `on_unknown`; preview selects a lower rule or reports fallback; probe headers agree | Identify model-runtime failure versus deliberate unknown policy. Check that signal's deployment and timeout before changing the rule. |
| Upstream delivery failure | Preview selects the intended route, while probe has an error path/status or failed `response.body.delivery` assertion | Selection worked; inspect the chosen backend endpoint, logs and response body. Do not retune the decision model to hide delivery failure. |

For a reproducible report, include active config identity, entrypoint/recipe,
sanitized request fixture or digest, expected route, preview fields and
`eval_trace` excerpt, inventory status for used deployments, probe status,
public headers, delivery assertion, plugin/warning evidence, the first failing
boundary, and one next check. If evidence is missing, say which observation
would settle the classification. Hand policy changes to the
[configuration workflow](https://vllm-sr.ai/install/agent/vllm-sr/references/configuration-loop.md)
or [routing calibration](https://github.com/vllm-project/semantic-router/blob/main/tools/agent/skills/routing-calibration/SKILL.md)
with the original failed case and a separate holdout.
