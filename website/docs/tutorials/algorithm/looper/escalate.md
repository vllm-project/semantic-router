# Escalate

## Overview

`escalate` calls a decision's candidates one at a time, in order, and returns
the first answer it accepts. It can start with a small or cheap model and pay
for a stronger one only when the first answer is not good enough.

It is an executor on both APIs:

- **Chat**: one accept check judges each answer, by its token confidence, by
  the decision model's `answer_check` question, or by the `refusal` question.
- **System One**: each stage answers the original typed request; native
  acceptance rules check every answer before the next stage is tried, and the
  complete response is returned unchanged.

## Key Advantages

- Escalates small-to-large or cheap-to-expensive instead of fixing a winner.
- Makes the stopping condition explicit and reviewable per decision.
- Bounds the worst case with the executor's `budget`: a deadline and a call
  limit shared by every attempt.
- Keeps System One answers intact: it never turns a native answer into Chat
  text or invents confidence for a missing answer.

## What Problem Does It Solve?

Always calling the strongest model spends its full cost on easy requests.
Escalate tries the cheaper candidate first and moves on only when its answer
fails an explicit check, so stronger models are paid for on the requests that
need them. The policy lives in router configuration instead of application
code.

## When to Use

- A route has candidates of different size or cost, and a cheaper answer is
  often good enough.
- You can state what makes an answer acceptable: token confidence, an
  answer-quality check, a refusal check, or, on System One, native answer
  probabilities.
- The route can accept the extra latency of a second call when the first
  answer is rejected.

## Configuration

Escalate's settings are the `escalate` block. Like every executor, it also
reads `budget` and `on_error` beside its type.

### On Chat

```yaml
recipes:
  - name: default
    decisions:
      - name: verified-answers
        priority: 100
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: qwen3-8b
          - model: qwen3-32b
        algorithm:
          type: escalate
          budget: {deadline: 60s, max_calls: 2}
          on_error: last_attempt
          escalate:
            order: size
            accept:
              confidence:
                method: hybrid
                gte: 0.72
                hybrid_weights: {logprob_weight: 0.65, margin_weight: 0.35}
```

`order` sets the escalation order: `size` tries the smallest
`providers.models[].param_size` first, `cost` the cheapest priced model
first, and omitting it keeps the `modelRefs` order.

`accept` names exactly one check:

| Check | Accepts an answer when |
| --- | --- |
| `confidence: {method, gte}` | The answer's confidence is at least `gte` |
| `systemone: {gte}` | The decision model's `answer_check` question finds the answer acceptable with probability at least `gte` |
| `refusal: false` | The decision model's `refusal` question does not flag the answer as a refusal |

The confidence methods read token log probabilities from the candidate's
response, so the backend must return them:

| Method | Confidence |
| --- | --- |
| `avg_logprob` (default) | Average token log probability, mapped from `[-3, 0]` to `[0, 1]` |
| `margin` | Average margin between the top two token log probabilities, mapped to `[0, 1]` |
| `hybrid` | `logprob_weight × avg_logprob + margin_weight × margin`; the two weights must sum to `1` |
| `self_verify` | The same model grades its own answer in one extra call |

`gte` is required and lies in `[0, 1]`. `token_filter: tool_call_args`
scores only the generated tool-call arguments instead of every token;
`all` (the default) scores every token.

`systemone: {gte, deployment, timeout_ms}` asks the decision model, or the
named System One deployment, its `answer_check` preset question about the
request and the candidate's answer. `refusal: false` asks the `refusal`
preset question; `refusal: true` is rejected, because it would accept only
refusals. Both checks are one decision-model call per judged answer; the call
counts against the executor's `deadline` but not its `max_calls`.

`on_error` decides what happens when no answer is accepted:

- `last_attempt` (the default) serves the last answer whose check ran.
- `fail` fails the request.

A failed model call, or an answer whose check cannot run, such as one returned
without the log probabilities a confidence method needs, always moves on to the
next candidate. When no candidate returns a judged answer, the request fails.

`budget` bounds the run: `max_calls` caps the candidate calls (and
`self_verify`'s extra calls) below the Router's per-request limit, and
`deadline` caps their wall-clock time. A run that exhausts its budget fails.

#### Attempt observability

Escalate executions create a `looper.execute` trace span with one
`looper.attempt` child for each dispatched candidate or verifier call. Router
Replay stores the corresponding bounded, content-free attempt records under
`route_diagnostics.looper`, including status, threshold outcome, token usage,
latency, and the final attempt ordinal. Prompts, responses, reasoning, endpoint
URLs, credentials, and raw errors are never included in this structure.

Every escalation sends the request to another candidate model. Make sure every
candidate is allowed by the route's data policy, and bound latency and cost for
the worst-case chain.

### On System One

A recipe published with `api: systemone` runs escalate with `escalate.stages`.
This guide starts with **Decision 2.0 Kai 0.6B → Vega 27B**: Kai answers the
original request; escalate checks that answer before deciding whether to call
Vega. It does not make a separate mandatory Kai classification call or rerun the
recipe after the first answer. It supports `choice`, `score`, and `noul`
requests and keeps the original question bundle together.

#### Connect the models

Model aliases are shared resources under `providers.models`. A local alias refers to a model runtime deployment; a remote alias refers to a concrete Engine or compatible System One endpoint. Choose one transport per alias.

```yaml
providers:
  models:
    - name: kai
      api_format: systemone
      deployment: local-kai
    - name: vega
      api_format: systemone
      provider_model_id: vllm-sr/Decision-2.0-Vega-27B
      backend_refs:
        - provider: systemone-compatible
          endpoint: http://engine.example.com:8900/v1

global:
  model_catalog:
    deployments:
      local-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        device: auto
```

`provider_model_id` is the model name sent to the remote endpoint. If a backend requires authentication, set `auth: {api_key_env: <VARIABLE>}` on its backend ref. Credentials stay on that backend. Replicas of the same remote model go in its `backend_refs`; different models are separate aliases and stages.

#### Publish an auto entrypoint

An API-scoped entrypoint makes the recipe available as `vllm-sr/auto`. The listener must explicitly allow that native name. Chat and System One aliases are separate: publishing this entrypoint does not change the Chat default recipe.

```yaml
listeners:
  - name: inference
    address: 127.0.0.1
    port: 8801
    systemone:
      models: [vllm-sr/auto]

entrypoints:
  - api: systemone
    model_names: [vllm-sr/auto]
    recipe: native-escalate
```

Add client authentication to the listener before exposing it beyond a trusted development environment. To allow a concrete model as well, add its native provider alias to `systemone.models`.

#### Define the stages

The following fragment accepts `choice`, `score` and `noul` requests. Kai exits
early only when every answer passes its type-specific gate. The threshold below
is a frozen pilot operating point for Decision 2.0 Kai → Vega, not a general
recommended default or an accuracy guarantee. The [reproduction guide](https://github.com/vllm-project/semantic-router/tree/main/tools/calibration/systemone_auto)
pins that experiment’s models and public benchmark. Choose thresholds on
independent calibration data for your own workload, then freeze them before
evaluation.

```yaml
recipes:
  - name: default
  - name: native-escalate
    decisions:
      - name: answer
        rules: {}
        modelRefs: [{model: kai}, {model: vega}]
        algorithm:
          type: escalate
          budget: {deadline: 10s, max_calls: 2}
          escalate:
            quality:
              type: uncalibrated
              accept:
                native:
                  - {question_type: choice, field: top_probability, gte: 0}
                  - {question_type: score, field: top_probability, gte: 0}
                  - {question_type: noul, field: top_probability, gte: 0}
            stages:
              - name: fast
                model: kai
                accept:
                  native:
                    - {question_type: choice, field: top_probability, gte: 0.6059704079536342}
                    - {question_type: score, field: top_probability, gte: 0.6059704079536342}
                    - {question_type: noul, field: top_probability, gte: 0.6059704079536342}
              - {name: strong, model: vega}
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
```

The common `gte: 0` rules require a valid distribution for every answer; they
do not claim an accuracy floor. Vega can return a complete valid answer after
Kai fails its stricter early-exit gate. This sample has no model-backed routing
signals, so its two-call budget covers Kai and, when needed, Vega. If a transport
retry consumes a call, fewer calls remain for later stages.

A stage runs a native model unless it says `kind: judge`, and is named by its
model unless it sets `name`. `modelRefs` is the complete candidate roster. Each stage names one member and can set its own timeout. Stage names cannot use `abstain`, which is reserved for the judge’s no-selection result. A stage cannot add a model that the decision did not declare.

For a native request, the Router starts one deadline and physical-call ledger when the public request handler begins, before it reads the body. The request ceiling is derived from the largest `algorithm.budget.deadline` and `algorithm.budget.max_calls` among the recipe's native decisions. After selecting a decision, the Router tightens both limits to that decision's configured budget. This preserves existing per-decision configuration without adding a recipe-level budget field.

Decision signals and built-in model signals run inside the request deadline and call ledger. Their own timeouts remain upper bounds. Every physical exchange counts, including remote decision-model calls and provider transport retries. Signal telemetry tracks signal exchanges; each algorithm stage's call telemetry includes its retries and judge calls. If signals exhaust the request ceiling, remaining signals stay unknown and execution stops. If the selected decision's tighter budget is already exhausted, its algorithm does not start. Either case returns `systemone_budget_exhausted` with `reason: max_calls` or `reason: deadline`; `model_calls` reports all physical exchanges charged so far. The call ledger cannot see work hidden inside an opaque external provider.

The Router validates every required answer before accepting a response. Missing answers, invalid values, unsupported question types and unproven full-input coverage do not become successful results. If no declared stage satisfies the rule before the budget is exhausted, the request is unresolved: System One escalate always fails rather than serving a rejected answer, so its `on_error` can only be `fail`. Without `quality`, a stage accepts any complete answer that passes its own rules.

The native budget does not replace HTTP server, proxy and client timeouts for
the complete request. In particular, Dashboard's
operator diagnostics use the management API's two-minute response write limit;
configure the full serving path appropriately for longer requests.

#### Try the route

Combine the model, listener, entrypoint and recipe snippets in a configuration
file with `version: v0.5`, connect the Vega Engine endpoint, and start the Router:

```bash
vllm-sr serve --config config.yaml
```

Then send a request to its native endpoint. The question names and criteria reach each attempted model unchanged.
Use your listener's API key if authentication is enabled.

```bash
curl -sS http://localhost:8801/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "state": "I cannot sign into my account.",
    "questions": {
      "task": {
        "type": "choice",
        "require_full_input": true,
        "instructions": "Which team should handle this request?",
        "criteria": {
          "account": "Account access or authentication",
          "billing": "Payments, invoices or refunds"
        }
      }
    }
  }'
```

The response's `routing` field reports the selected model, stage and physical
algorithm model-call count, including retries. A `503` with
`systemone_unresolved` means no complete answer
passed the configured rule; it does not mean that an omitted answer was false.
See the [Router API (English)](https://vllm-sr.ai/docs/next/api/router#route-a-system-one-request) for the
response fields and error contract.

In Dashboard, open **System One → Decision Playground**, then choose
`vllm-sr/auto` under **Automatic routes**. Load an example supported by your
recipe or enter your own questions, then select **Run**. Dashboard uses your
signed-in management permissions, so you do not need to paste a public API key.
**Decision Monitoring** shows stage outcomes, request rates and latency. These
describe live execution; evaluate labeled requests separately to measure
accuracy.

#### Add routing signals

Native signals inspect the complete task document, including its states and questions. The request does not supply a Chat conversation or trusted identity envelope. References to `authz`, `metadata`, `conversation`, `reask`, `user_feedback` and `input_modality` therefore fail configuration validation, including through projections. KB signals are also unavailable in native recipes.

Preference signals must use a native decision task deployment. Explicit contrastive or external preference adapters and MCP domain classifiers are unavailable in native recipes until they share the request lifecycle. Supported model-backed signals retain their own transport timeouts inside the shared request deadline and call ledger.

The recipe's `strategy` still chooses among matching decisions. Native execution uses its explicit stages and budget rather than Chat fallback, candidate requirements, decision reliability, output contracts or adaptation controls. Keep Chat fallback disabled for this native recipe. `modelRefs` declares aliases only; stage order determines selection, so Chat reasoning controls and model weights are rejected. Provider-level retry and timeout settings still apply to remote exchanges.

A recipe can contain several decisions, each with its own stages and budget.
For example, a keyword decision can send requests mentioning an appeal directly
to Vega with one stage, while the default decision tries Kai → Vega. Only the
selected decision executes, and its budget is tightened from the shared request
ceiling after selection.
Keywords inspect the task document, including question text, and callers can
trigger them deliberately. Use them as routing hints, not authorization or a
promise of better accuracy.

#### Choose acceptance evidence

An uncalibrated rule can inspect these observations:

| Field | Meaning |
| --- | --- |
| `top_probability` | Largest probability in the native answer distribution; for `noul`, `max(p, 1-p)` |
| `confidence` | The confidence reported by the native model, when present: System One's published statistic for Choice and Score (see [Decisions](https://vllm-sr.ai/docs/next/model-runtime/reference#decisions)); since runtime contract 3.2.0 its values differ from earlier releases, so recheck thresholds |
| `probability_margin` | For `noul`, `2 × abs(p - 0.5)`; both confident true and confident false answers have a large margin |

A rule bounds its field with `gt`, `gte`, `lt`, or `lte`, in `[0, 1]`. A rule can target a `question_type`, a named `question`, or both, and optionally a named `state`. All required answers need rule coverage. A stage-local `accept` rule adds a condition; it cannot weaken the algorithm's common quality rule.

A high model probability is not a measured accuracy guarantee. `quality.type: calibrated` instead names an immutable resource in `evaluation.calibrations`, declares `loss: bundle_error` and an explicit `max_risk`. The runtime must verify both the artifact's hash and its applicability to the deployed model, task and arrival population. A missing or incompatible artifact must not silently become an uncalibrated pass.

#### Use independently certified acceptance

Calibrated execution is an experimental, narrower path. It currently supports native stages in recipes without signals or projections. Judges and model-backed signal paths need additional inference provenance before they can use certified acceptance. Uncalibrated stages can still use the full routing path and an optional judge.

To prepare a `systemone-calibration/v1` artifact:

1. Freeze the task's question instructions, criteria and inference options, the policy, and each probability bin's acceptance decision. Input text can vary; changing a question creates a different task template. `options.return_meta` controls evidence visibility and does not change the task identity.
2. Collect a fresh certification cohort independently from training and threshold selection. Count source bundles, rather than their questions or augmented copies. The statistical assumption is i.i.d. draws from the declared conditional population; a stratified benchmark average does not automatically satisfy it.
3. Record the full native arrival history and actual model identity from each response. The runtime checks the model, revision, content hash, engine, dtype, numerics and accelerator. A model name alone is insufficient.
4. Record bundle-error counts in each frozen bin and retain the source manifest, labeling rules and gate-definition digest for audit. A bundle is erroneous when any required answer fails the cohort's predeclared correctness rule. The artifact's `certification` identifies this evaluated population and declares the independent sampling method.

The runtime computes a one-sided Clopper–Pearson upper bound from the counts, with a Bonferroni correction across all declared bins. Every predeclared accepting bin must meet `max_risk`, or loading fails. Counts cannot turn a previously rejecting bin into an accepting bin: doing so would change the arrival population of later stages. An unvisited bin can have zero samples only when it remains rejecting.

The artifact binds the recipe's execution semantics and frozen gates, rather than a global configuration hash. Moving credentials, changing listeners or updating explanatory text does not invalidate it. Missing runtime identity, an unknown task, an error history or an uncovered bin leaves the request unresolved. An artifact's declarations still require a dataset audit; the bound describes its evaluated population and makes no guarantee about distribution shifts or an individual request.

The Go artifact helpers are `CalibrationTaskSHA256`, `CalibrationPolicySHA256` and `CalibrationGateSHA256` in `pkg/systemone`. Pilot tuning results are not certification artifacts: even zero errors in 96 independent examples gives an approximately 3.1% upper risk bound at 95% confidence before correcting for multiple bins.

#### Add an optional LLM judge

A `kind: judge` stage calls a declared OpenAI-compatible provider alias and
selects a complete response from earlier native attempts. It preserves that
response's native answer probabilities; an LLM's self-reported confidence does
not replace them. Declare `generation.max_output_tokens` to bound its output.

For an uncalibrated experiment, keep a minimum acceptance floor in the common
quality rule and put stricter early-exit thresholds in the native stages'
`accept` rules. The judge can then choose a valid earlier answer that meets the
common floor even when it did not meet an early-exit threshold. If the common
rule itself rejects that answer, the judge cannot make it pass merely by
selecting it. A judge is additional evidence to evaluate, not an accuracy
guarantee.

```yaml
stages:
  - model: kai
    accept:
      native:
        - {question_type: choice, field: top_probability, gte: 0.9}
  - model: vega
    accept:
      native:
        - {question_type: choice, field: top_probability, gte: 0.9}
  - name: review
    kind: judge
    model: reviewer
    generation: {max_output_tokens: 256}
```

Add `reviewer` to the decision's `modelRefs` and connect it with
`api_format: openai` and increase this algorithm's budget to allow the third
call. The judge consumes the same shared request deadline and call ledger as
the native stages. Its output is a selection from prior complete native
responses, not a replacement answer synthesized by an LLM.

#### Measure the tradeoff

Compare direct Kai, direct Vega and escalate on the same requests. Choose
acceptance thresholds on separate calibration data, then freeze the complete
configuration before evaluation. Report complete responses, unresolved requests,
escalation rates and measured frontend latency together. A threshold does not
enforce a fixed fast-path fraction when the request distribution changes.

Count model-backed signals, retries and judge calls when measuring total
request cost; the algorithm call count covers only execution after the
decision is selected.

Offline replay helps choose experiments; it does not establish live latency or
GPU cost. The [calibration tools](https://github.com/vllm-project/semantic-router/tree/main/tools/calibration/systemone_auto)
include a pinned public JevBench command and separate research replay tools.

### Migrating from confidence and cascade

`escalate` replaces the Looper `confidence` algorithm and the native `cascade`.
`vllm-sr config migrate` rewrites both:

- `confidence.confidence_method` and `threshold` become
  `escalate.accept.confidence.method` and `gte`; `escalation_order: size` or
  `small_to_large` becomes `order: size`, `cost` stays `cost`, and `declared`
  keeps the `modelRefs` order.
- `confidence_method: automix_entailment` becomes `accept.systemone`: the
  decision model's `answer_check` question replaces the external verifier
  server, and its threshold becomes `gte`. Re-fit that threshold.
- `confidence.on_error: fail` becomes `algorithm.on_error: fail`, which now
  fails the request when no answer is accepted; a failed call moves on to the
  next candidate instead of stopping the run.
- A cascade's `quality` and `stages` move under `escalate`, each `acceptance`
  becomes `accept.native` with the predicate bounds on each rule, and native
  stages drop `kind: native`.

Reusable fragment: [`config/fragments/algorithm/looper/escalate.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/looper/escalate.yaml).
