# Verify routing and delivery

Discover the management origin, inference listener and public entrypoint from the
selected stack. Use installed `route preview --help` and `route probe --help` for
supported inputs, assertions and credential-variable options.

## Preview then probe

Preview evaluates signals, projections and decisions without backend generation.
Inspect recipe, decision, algorithm, selection status and trace evidence.
Learning-enabled preview uses a read-only snapshot: inspect
`selection_provenance`, including config/state identity and sampling seed.
A resolved preview model may differ from a later live choice; `execution_required`
is unresolved execution, not a completed selection. Do not disable Learning to
make a diagnostic look deterministic.

For chat context, `route preview --request-file FILE` accepts the supported request
subset with messages, tools and response constraints. Discover its schema;
arbitrary Chat Completions fields are not all accepted. Preserve actual history,
tool calls and payloads, rather than substituting a display prompt.

Probe sends a real request through Envoy. For example, with origins and entrypoint
already discovered:

```bash
vllm-sr route preview --endpoint "$ROUTER_ORIGIN" --model "$ENTRYPOINT" \
  --prompt 'Define a readiness probe.' --trace --json
vllm-sr route probe --config config.yaml --base-url "$INFERENCE_BASE_URL" \
  --model "$ENTRYPOINT" --prompt 'Define a readiness probe.'
```

Add installed `--expect-*` assertions for the intended route. Requested routing
identity and the backend's returned model name may differ; calibrate them
separately. Management and inference credentials are independent environment
references. Choose time/output budgets for the model and input, including
reasoning tokens.

A successful HTTP status is not sufficient: check `response.body.delivery` and
final assistant output. Empty/reasoning-only output, malformed tool arguments and
`finish_reason: length` fail delivery. A refusal or valid tool call proves
transported output, not answer quality or successful tool execution. Preserve
failed attempts; do not rewrite bound probes to make them pass.

## Token boundaries and public errors

Test learned input capacity, candidate context/output eligibility and backend
prompt-plus-generation limits independently. Tokenizers and chat templates may
count differently. Keep other budgets within range to isolate the acting limit.
A learned `overflow: reject` can become another decision under `on_unknown:
no_match`; inspect the actual public response and route rather than assuming an
API rejection. Compare direct and routed behavior when isolating backend faults.

## Repeated API and UI checks

Choose a bounded matrix for the affected behavior: ordinary and boundary cases,
fallbacks, tools, multi-turn context or modalities as relevant. Test through the
public entrypoint; direct backend checks alone do not cover routing. A tool cycle
needs a matching tool result and subsequent final answer, not just a first call.

When UI work is requested, exercise the same entrypoint and workflow in Dashboard.
Check completion, error handling, persistence and reload. Report API-only coverage
separately from UI coverage. For stability claims, state the tested load/context
and per-path success/total, latency and elapsed time; a short smoke is not sustained
load evidence. Signal timings can overlap and must not be summed as sequential.

Use [recipe tuning](https://vllm-sr.ai/install/agent/vllm-sr/references/recipe-tuning.md) for policy changes and
[sr-bench](https://vllm-sr.ai/install/agent/vllm-sr/references/sr-bench.md) for datasets, paired capability/cost comparisons
and the optimization loop. Individual route probes are delivery checks, not
benchmark scores.
