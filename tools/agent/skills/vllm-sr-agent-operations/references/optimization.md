# Optimize vLLM Semantic Router

Start from the objective the user states (answer quality, latency or cost) and
its acceptable trade-offs. Measure a baseline, make one change through the
[configuration workflow](https://vllm-sr.ai/install/agent/vllm-sr/references/configuration-loop.md#change-a-running-stack), measure again on the same
cases, and keep the change only when it wins. Set `ROUTER_ORIGIN` to the
stack's management origin.

## Measure the baseline

```bash
vllm-sr config get --format yaml --endpoint "$ROUTER_ORIGIN" > baseline.yaml
curl -s "$ROUTER_ORIGIN/api/v1/instance"
curl -s "$ROUTER_ORIGIN/api/v1/inventory/model-runtime" | python3 -c '
import json, sys
for d in json.load(sys.stdin)["deployments"]:
    print(d["name"], d.get("repo"), d.get("device"), d["ready"])'
```

Record the decision model, its device and the active `config_version`. Freeze
the cases before you look at candidate results: a short sr-bench dataset
(below) or the user's own requests. For latency, read `routing_latency_ms` in
`x-vsr-trace` (with debug access) and `latency_ms` in a probe receipt; for
cost, give each model its `pricing`, which buffered responses report in
`x-vsr-cost`.

## Choose the decision model

Every model-judged signal, the `systemone` selector and every judge ask the
decision model, so its size and device set the routing latency. On a CPU host
the default, vela3-fast, held about 7 GB and added a few hundred milliseconds
to each routed request: about 230 ms for a one-line prompt with one domain
rule, about 480 ms for an MMLU-Pro question.

| Need | Model | Runs on |
| --- | --- | --- |
| The default: every built-in question trained, with spans, calibrated thresholds | `vllm-sr/vela3-fast` | CPU, or a GPU with `device: auto` |
| Lower latency on a small GPU, multilingual | `vllm-sr/vela3-lite` (0.8B) | GPU |
| More accuracy, the built-in questions still trained | `vllm-sr/vela3-mini` (4B), `vllm-sr/vela3-flash` (9B), `vllm-sr/vela3-turbo` (27B) | GPU |
| Questions over images and video | the d3 sizes, `vllm-sr/d3-edge` to `vllm-sr/d3` | GPU; no spans |
| Your own questions on a CPU | `vllm-sr/Decision-2.0-Kai-0.6B` | CPU or GPU; no spans |
| No local model at all | a remote System One API (`provider: systemone`) | Someone else's hardware; the text it reads leaves the host |

Change the model or device by declaring `primary` again:

```yaml
global:
  model_catalog:
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/vela3-lite
        device: rocm:0
```

A remote API answers the questions nothing else binds; keep span-needing
signals on a local vela3 deployment, or the load fails with `span_required`:

```yaml
global:
  model_catalog:
    deployments:
      d3-remote:
        provider: systemone
        endpoint: https://systemone.example.com/v1/systemone
        model: vllm-sr/d3-ultra
        schema: vllm-sr/5.0
        auth: {api_key_env: SYSTEMONE_API_KEY}
        timeout_ms: 1500
    system:
      decision_model: {deployment: d3-remote}
    bindings:
      pii: primary
      hallucination: primary
```

Only vela3-fast and Vela 2.0 have calibrated thresholds; other models use 0.5
with an `uncalibrated_preset` warning until a rule writes its own `predicate`.
After the change: the inventory shows the new deployment `ready` on the
intended device, the dev cases still preview to the intended decisions, and
`routing_latency_ms` moved the way you wanted. The engine arguments
(`vllm-sr serve --max-num-seqs`, `--max-num-batched-tokens`,
`--batch-window-ms`, `--dtype`, `-dp N`) tune throughput; see
[Choose a model, size and hardware](https://vllm-sr.ai/docs/model-runtime/get-started/choose-a-model).

## Tune the recipe

Follow entrypoint, recipe, signals and projections, decision, algorithm and
model, plugins, and change the part responsible for the failure. The
[recipe tuning reference](https://vllm-sr.ai/install/agent/vllm-sr/references/recipe-tuning.md) covers System One questions and
predicates, unknown answers, selectors and executors, Learning, and how to
iterate from evidence.

## Run a short sr-bench evaluation

sr-bench runs frozen cases against single-model and MoM (mixture of models)
targets through the stack's sr-bench service (port 8090), which `--minimal`
leaves out. This smoke run prepares three MMLU-Pro cases, compares the
backend alone with `vllm-sr/auto`, and makes six model calls:

```bash
vllm-sr benchmark catalog
vllm-sr benchmark dataset prepare --benchmark mmlu-pro --profile smoke --limit 3 > dataset.json
curl -s "$ROUTER_ORIGIN/api/v1/config/hash" > hash.json
python3 - <<'PY'
import json
dataset, hashes = json.load(open("dataset.json")), json.load(open("hash.json"))
manifest = {
    "version": "sr-bench-1.0", "name": "smoke", "mode": "live",
    "profile": dataset["profile"], "seed": dataset["seed"],
    "dataset": {"path": dataset["path"], "sha256": dataset["sha256"]},
    "targets": [
        # The worker runs in a container: these are addresses as it sees them.
        {"id": "single", "kind": "single", "model": "MODEL",
         "base_url": "http://host.docker.internal:8000/v1"},
        {"id": "mom", "kind": "mom", "model": "vllm-sr/auto",
         "base_url": "http://vllm-sr-router-container:8899/v1",
         "config_hash": hashes["active_runtime_hash"]},
    ],
    "cost_policy": "capability_only",
    "limits": {"concurrency": 1, "total_timeout_s": 180, "idle_timeout_s": 30,
               "max_output_tokens": 4096, "max_run_seconds": 1800},
    "sampling": {"temperature": 0, "top_p": 1, "max_tokens": 4096},
}
json.dump(manifest, open("bench.json", "w"), indent=2)
PY
vllm-sr benchmark plan --manifest bench.json --output frozen.json
vllm-sr benchmark run --manifest frozen.json --idempotency-key smoke-1 > run.json
python3 -c '
import json
r = json.load(open("run.json"))
print(r["run_id"], r["status"], [(t["id"], t["completed"], t["failed"], t["complete"]) for t in r["summary"]["targets"]])'
```

Set the single target to the backend's own address and model name as the
worker container reaches it, and `NAME-vllm-sr-router-container` for a stack
named `NAME`. Pass: `plan` exits 0; `run` exits 0 with `completed` and every
target `complete` with 0 failed; `vllm-sr benchmark report RUN_ID --output
report.json` writes the quality, usage, latency and limitations.

- A MoM target's receipts ride in `x-vsr-trace`, so the stack must allow the
  worker's requests the trace. On a loopback stack, set
  `global.router.headers.debug: any` through the configuration workflow
  before you read the hash and plan; it changes the user's stack, so ask
  unless the user already allowed changes to it. Restore the previous value
  after the run when the user wants the trace off. Without it the run fails
  with `Frozen runtime identity verification failed`.
- `capability_only` permits unknown prices and so proves no savings; priced
  runs need complete `prices` on registered targets.
- Repeating the same idempotency key returns the same run without new model
  calls. `--detach` returns at once; `vllm-sr benchmark show RUN_ID` polls it.
- A smoke run proves the pipeline, not quality. For a comparison, follow
  [sr-bench](https://vllm-sr.ai/install/agent/vllm-sr/references/sr-bench.md): Quick datasets for tuning, a frozen Standard holdout,
  `candidate-plan` and `compare` against the best single model.

## Caches and judge caps

- **Response cache.** `mode: exact` reuses identical requests and asks no
  model. `semantic` recalls similar cached requests by embedding and lets the
  decision model's `same_request` judge confirm each one (at a `min_score` of
  0.8 by default), so a paraphrase with another meaning is a miss. Keep
  `recall.top_k` and `judge.max_candidates` small (3): every recalled
  candidate is one judge entry. Measure the hit rate and the judge's added
  latency before widening them.
- **Response cache with a guard.** On a decision with a `response_guard`,
  the cache stores only what the guard released, so its hit rate is lower,
  and a change to the guard starts its cache fresh. A hit skips the model and
  the guard: an exact hit is the same request, but a semantic hit serves the
  answer released for a similar request as is, without the guard's
  conditions on the new request. Use `mode: exact` on decisions whose guard
  depends on the request. After changing the threshold of a signal the guard
  reads, set a new `revision.policy_revision` in that decision's
  `response_cache` configuration, which starts its cache fresh; prefer it to
  a flush, which on a Redis-like store doesn't survive a configuration
  reload:

  ```yaml
  recipes:
    - name: default
      decisions:
        - name: guarded-route
          priority: 100
          rules: {operator: AND, conditions: []}
          modelRefs: [{model: MODEL}]
          plugins:
            - type: response_cache
              configuration:
                enabled: true
                mode: exact
                revision:
                  policy_revision: guard-2
  ```

- **Judge caps.** `judge.max_candidates` bounds every judge (255 at most for a
  choice judge such as `tool_rank`, 64 by default for `relevance` and
  `same_request`). Truncation never fails a request; it counts in
  `vsr_match_judge_truncated_total` on the metrics port. When it climbs, add a
  `recall` stage (`bm25` needs no model) rather than raising the cap.
- **Embeddings** load only when a recall stage uses them; `bm25` recall keeps
  the embedding model off entirely.

## Latency

- Move the decision model to a GPU, or to a smaller model, before tuning
  anything else; it answers every model-judged rule of a request.
- Rules that no decision, projection or plugin reads are never asked; remove
  rules you don't route on. Questions of one stage about one state share one
  call.
- `global.model_catalog.signal_timeout_ms` bounds all of a request's signals,
  and a rule's `timeout_ms` its own question. A late answer is unknown, not an
  error, so choose `on_unknown` deliberately.
- Compress what the signals read: `compress.signals` in the `context`
  plugin of `global.plugins` (or a recipe's) shortens a long request before
  the signals judge it, which cuts the decision model's share of latency and
  cost. A `compress` block with only `signals` leaves the request the model
  receives whole; the block's own fields (`mode`, `targets`) trim that.

  ```yaml
  global:
    plugins:
      - type: context
        configuration:
          enabled: true
          compress:
            signals:
              enabled: true
              max_tokens: 4096
              skip_signals: [jailbreak, pii]
  ```

  `skip_signals` keeps the original text for the signals that must read all
  of it. Check the routes again after enabling it: a judge reads less.
- A guard that can refuse or rewrite an answer (a `block` rule, an
  `annotate` rule with `details`, or `on_unknown: block`) makes the Router
  hold its decision's streamed answer until the guard has read all of it:
  the first token arrives only after the whole answer, and a held answer over
  64 MiB fails closed. That is the price of guarding a streamed answer; keep
  such rules on the decisions that need them. A guard that only records
  (`none`, or `annotate` without `details`) keeps live streaming, and
  `llm_response_guard_buffered_total` counts the held streams per decision.
- Measure cold start separately from warm median and p95 latency.

## Cost

Give each model its `pricing`, so responses carry `x-vsr-cost` and
sr-bench can price runs:

```yaml
providers:
  models:
    - name: small
      backend_refs: [{name: small, provider: vllm, endpoint: "small.example.com:8000"}]
      pricing: {currency: USD, prompt_per_1m: 0.1, completion_per_1m: 0.4}
```

Then let a decision choose the cheaper candidate: `multi_factor` weighs
quality, latency, cost and load (`vllm-sr config schema --surface
algorithm:multi_factor`), `escalate` tries a small model first and accepts its
answer when `answer_check` passes, and the `systemone` selector picks a
candidate from its description. At least two eligible models are needed to
show a trade-off; one model only proves delivery. Report savings only from a
priced, complete comparison on the same cases.
