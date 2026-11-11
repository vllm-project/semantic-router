# Verify routing and delivery

Discover the management origin (`ROUTER_ORIGIN`), the inference base URL and
the public entrypoint of the stack you verify; `GET /v1/models` on the
listener lists the entrypoints. Installed `vllm-sr route preview --help` and
`vllm-sr route probe --help` list the supported inputs and assertions.

## What a response proves

Every routed response carries the public set, on upstream answers and on the
Router's own answers (cache hits, blocks, errors) alike:

| Header | Meaning |
| --- | --- |
| `x-vsr-schema-version` | The header contract, `3` |
| `x-vsr-config-version` | The configuration version that served the request |
| `x-vsr-response-path` | What answered: `upstream`, `cache`, `fast_response`, `looper`, `blocked`, `rate_limited`, `error` or `fallback` |
| `x-vsr-selected-recipe`, `x-vsr-selected-decision`, `x-vsr-selected-model` | The routing result; the decision is omitted when no decision matched |
| `x-vsr-warnings` | Codes of the `response_guard` rules that annotated the answer, and protocol losses |
| `x-vsr-replay-id`, `x-vsr-cost` | The replay record and the configured price, when enabled |

Everything else (matched signals, confidence, algorithm, ranking) rides in
`x-vsr-trace`, an RFC 8941 dictionary of at most 4 KB, sent only when the
request carries `x-vsr-debug: true` **and** `global.router.headers.debug`
allows the caller (`none` by default). No trace means no permission, not no
routing. The old `x-vsr-matched-*`, `x-vsr-cache-hit` and `x-selected-model`
headers are gone.

## Preview, then probe

Preview evaluates signals, projections and decisions without asking the
backend; probe sends a real request through the listener (the Router in
standalone mode, Envoy with `--gateway extproc`):

```bash
vllm-sr route preview --endpoint "$ROUTER_ORIGIN" --model "$ENTRYPOINT" \
  --prompt 'Define a readiness probe in one sentence.' --trace --json > preview.json
python3 -c '
import json
d = json.load(open("preview.json"))
print(d["routing_decision"], d["selected_model"], d["selection_status"], d.get("signal_errors", {}))'
vllm-sr route probe --config config.yaml --base-url "$INFERENCE_BASE_URL" \
  --model "$ENTRYPOINT" --prompt 'Define a readiness probe in one sentence.' \
  --max-completion-tokens 256 --expect-decision "$DECISION" > probe.json
echo "probe exit=$?"
python3 -c '
import json
r = json.load(open("probe.json"))
print(r["passed"], r["response"]["status"], r["response"]["routing"])'
```

Pass: the preview prints the intended decision and model, `selected`, and
`{}`; the probe exits 0 and prints `True 200` with the same decision under
`x-vsr-selected-decision`. The probe exits 2 when an assertion fails and
records each one under `assertions`.

- In the traced preview, each condition of `eval_trace` has a `state` and,
  when its signal got no answer, a `signal_error` (such as
  `systemone_failed`); `signal_errors` lists them by `type:name`. An unknown
  rule resolves through `on_unknown`, usually to no match, so the request
  takes a lower decision: that is a model failure, not a policy result.
- `selection_status` is `selected` for a resolved model; `execution_required`
  means selection depends on execution (Learning, multi-model workflows) and
  is not a completed choice. Learning previews read a snapshot without
  advancing it; don't disable Learning to make a preview deterministic.
- `--expect-recipe`, `--expect-decision` and `--expect-selected-model` read
  public headers; `--expect-algorithm` reads the trace, so it needs debug
  access. `--expect-response-model` checks the backend's own `model` field,
  which may differ from the routing alias.
- Without `--max-completion-tokens` the probe waits for the whole answer and
  a slow backend reaches the 120 s `--timeout`. With one, pick a prompt whose
  answer fits: an answer cut at the cap fails delivery.
- `route preview --request-file FILE` accepts chat messages, tools and
  response constraints. Preserve real history and tool payloads rather than a
  display prompt.

## The decision model

A decision-model rule only routes when its deployment answers:

```bash
curl -s "$ROUTER_ORIGIN/api/v1/inventory/model-runtime" | python3 -c '
import json, sys
for d in json.load(sys.stdin)["deployments"]:
    print(d["name"], d["state"], d["ready"], d.get("device"), d.get("repo"))'
curl -s "$ROUTER_ORIGIN/api/v1/instance"
```

Pass: each deployment a rule resolves to is `ready True`, on the intended
device. `vllm-sr status` names the decision model, and `GET /api/v1/instance`
its artifact. A remote `provider: systemone` deployment isn't in this list:
its failures appear as `systemone_deployment_failed` in `vllm-sr logs router`
and as `signal_errors` in a traced preview.

To ask the decision model directly, grant it on a listener
(`listeners[].systemone.models: [vllm-sr/vela3-fast]`) or serve it in engine
mode, then send a standard request (`state` and `questions`) or a batch
(`states`, each entry a standard request) to `POST /v1/systemone`; the answers
come back under `answers`, or under `states.NAME.answers` for a batch. See
[the System One API](https://vllm-sr.ai/docs/system-one/use/api).

## Delivery

A successful status is not delivery: check `response.body.delivery` in the
probe receipt and the final assistant output. Empty or reasoning-only output,
malformed tool arguments and `finish_reason: length` fail delivery. A refusal
or a valid tool call proves transport, not answer quality or tool execution.
Preserve failed attempts; don't rewrite bound probes to make them pass.

A decision whose `response_guard` can refuse or rewrite an answer holds a
streamed answer until the guard has read all of it: nothing arrives before
the model finishes, and a refused answer is HTTP 200 with only the
protocol's stream error. Its codes are in `x-vsr-warnings`, and, with trace
access, the trace's `response_guard` member carries the guard's outcome
(such as `blocked`), with `buffered` for a held stream.

## Images

The Images API reaches only a decision whose output is an image:

```bash
curl -s -w ' http=%{http_code}\n' http://127.0.0.1:8899/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model": "vllm-sr/auto", "prompt": "A red bicycle at dawn"}'
```

Pass, with an image decision: `http=200` and `x-vsr-response-path:
upstream`. Without one: `http=400` with
`"code":"image_route_required"`, before any model is called.

## Token boundaries and public errors

Test learned input capacity, candidate context and output eligibility, and
backend prompt-plus-generation limits independently. Tokenizers and chat
templates count differently. Keep other budgets within range to isolate the
acting limit. Inspect the actual public response and route rather than
assuming an API rejection, and compare direct and routed behavior when you
isolate a backend fault.

## Repeated API and UI checks

Choose a bounded matrix for the affected behavior: ordinary and boundary
cases, fallbacks, tools, multi-turn context or modalities as relevant. Test
through the public entrypoint; direct backend checks alone do not cover
routing. A tool cycle needs a matching tool result and a final answer, not
just a first call.

When UI work is requested, exercise the same entrypoint in the Dashboard and
report API-only coverage separately. For stability claims, state the load,
context and per-path success/total, latency and elapsed time; a short smoke is
not sustained-load evidence.

Use [recipe tuning](https://vllm-sr.ai/install/agent/vllm-sr/references/recipe-tuning.md) for policy changes and
[sr-bench](https://vllm-sr.ai/install/agent/vllm-sr/references/sr-bench.md) for datasets and measured comparisons. Probes are
delivery checks, not benchmark scores.
