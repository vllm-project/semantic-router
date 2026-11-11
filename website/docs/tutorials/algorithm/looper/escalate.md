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
first, and omitting it keeps the `modelRefs` order. `size` reads the leading
count of a size, so `397B (17B active)` is 397B and `1.6T` is 1600B. A model
without a readable size, or for `cost` without pricing, is tried last.

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
counts against the executor's `deadline` but not its `max_calls`. An answer
that calls tools is judged with each call's tool name and arguments after its
text.

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

A recipe published with `api: systemone` runs escalate with
`escalate.stages`. This guide starts with **Decision 2.0 Kai 0.6B → Vega 27B**:

- Kai answers the original request, and escalate checks that answer before
  deciding whether to call Vega.
- There is no separate mandatory Kai classification call, and the recipe does
  not run again after the first answer.
- It supports `choice`, `score` and `noul` requests, and keeps the original
  question bundle together.

#### Connect the models

Model aliases are shared resources under `providers.models`. A local alias
refers to a model runtime deployment; a remote alias refers to a concrete
Engine or compatible System One endpoint. Choose one transport per alias.

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

- `provider_model_id` is the model name sent to the remote endpoint.
- If a backend requires authentication, set `auth: {api_key_env: <VARIABLE>}`
  on its backend ref. Credentials stay on that backend.
- Replicas of the same remote model go in its `backend_refs`; different models
  are separate aliases and stages.

#### Publish an auto entrypoint

An API-scoped entrypoint makes the recipe available as `vllm-sr/auto`. The
listener must explicitly allow that native name. Chat and System One aliases
are separate: publishing this entrypoint does not change the Chat default
recipe.

```yaml
listeners:
  - name: inference
    address: 127.0.0.1
    port: 8801
    apis:
      systemone:
        models: [vllm-sr/auto]

entrypoints:
  - api: systemone
    model_names: [vllm-sr/auto]
    recipe: native-escalate
```

Add client authentication to the listener before exposing it beyond a trusted
development environment. To allow a concrete model as well, add its native
provider alias to `apis.systemone.models`.

#### Define the stages

The following fragment accepts `choice`, `score` and `noul` requests. Kai
exits early only when every answer passes its type-specific gate.

:::warning The threshold is a pilot operating point
It was frozen for Decision 2.0 Kai → Vega, and is not a recommended default
or an accuracy guarantee. The
[reproduction guide](https://github.com/vllm-project/semantic-router/tree/main/tools/calibration/systemone_auto)
pins that experiment's models and public benchmark. Choose thresholds on
independent calibration data for your workload, then freeze them before
evaluation.
:::

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

- **The common rules check validity, not accuracy.** The `gte: 0` rules
  require a valid distribution for every answer; they claim no accuracy floor.
- **Vega answers after Kai fails its gate.** Vega can return a complete valid
  answer after Kai fails the stricter early-exit gate.
- **The budget.** This sample has no model-backed routing signals, so its
  two-call budget covers Kai and, when needed, Vega. A transport retry that
  uses a call leaves fewer for later stages.

Stage rules:

- A stage runs a native model unless it says `kind: judge`, and is named by its
  model unless it sets `name`.
- `modelRefs` is the complete candidate roster. Each stage names one member,
  can set its own timeout, and cannot add a model the decision didn't declare.
- Stage names cannot use `abstain`, which is reserved for the judge's
  no-selection result.

Budget and deadline:

- **One budget per request.** The Router starts one deadline and
  physical-call ledger when the public request handler begins, before it reads
  the body.
- **The ceiling** is the largest `algorithm.budget.deadline` and
  `algorithm.budget.max_calls` among the recipe's native decisions. Once a
  decision is selected, both tighten to that decision's budget.
- **Signals count.** Decision signals and built-in model signals run inside
  the same deadline and ledger; their own timeouts remain upper bounds.
- **Every physical exchange counts,** including remote decision-model calls
  and provider transport retries. Signal telemetry tracks signal exchanges, and
  each stage's call telemetry includes its retries and judge calls.

When the budget runs out:

| Situation | What happens |
| --- | --- |
| Signals exhaust the request ceiling | The remaining signals stay unknown, and execution stops. |
| The selected decision's tighter budget is already spent | Its algorithm does not start. |

Both return `systemone_budget_exhausted` with `reason: max_calls` or
`reason: deadline`, and `model_calls` reports every physical exchange charged
so far. The ledger can't see work hidden inside an opaque external provider.

Acceptance:

- **Every required answer is validated** before a response is accepted.
  Missing answers, invalid values, unsupported question types and unproven
  full-input coverage never become successful results.
- **No stage accepted means unresolved.** System One escalate always fails
  rather than serving a rejected answer, so its `on_error` can only be `fail`.
- **Without `quality`,** a stage accepts any complete answer that passes its
  own rules.

:::note
The native budget does not replace HTTP server, proxy and client timeouts for
the complete request. Dashboard's operator diagnostics, for example, use the
management API's two-minute response write limit.
:::

#### Try the route

Combine the model, listener, entrypoint and recipe snippets in a configuration
file with `version: v0.5`, connect the Vega Engine endpoint, and start the Router:

```bash
vllm-sr serve --config config.yaml
```

Then send a request to its native endpoint. The question names and criteria
reach each attempted model unchanged. Use your listener's API key if
authentication is enabled.

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
`systemone_unresolved` means no complete answer passed the configured rule; it
does not mean that an omitted answer was false. See the
[Router API (English)](https://vllm-sr.ai/docs/api/router#route-a-system-one-request)
for the response fields and error contract.

To try it in Dashboard:

1. Open **System One → Decision Playground**, and choose `vllm-sr/auto` under
   **Automatic routes**.
2. Load an example your recipe supports, or enter your own questions, then
   select **Run**. Dashboard uses your signed-in management permissions, so no
   public API key is needed.
3. Open **Decision Monitoring** for stage outcomes, request rates and latency.

These views describe live execution; evaluate labeled requests separately to
measure accuracy.

#### Add routing signals

Native signals inspect the complete task document, including its states and
questions. The request carries no Chat conversation or trusted identity, so
references to `authz`, `metadata`, `conversation`, `turn_relation`,
`user_feedback` and `input_modality` fail configuration validation, including
through projections.

Supported model-backed signals keep their own transport timeouts inside the shared request deadline and call ledger.

What a native recipe keeps and drops:

- **Kept:** the recipe's `strategy` still chooses among matching decisions,
  and provider-level retry and timeout settings still apply to remote
  exchanges.
- **Dropped:** native execution uses its explicit stages and budget instead of
  Chat fallback, candidate requirements, decision reliability, output
  contracts or adaptation controls. Keep Chat fallback disabled for this
  recipe.
- **Aliases only:** `modelRefs` declares aliases, and stage order decides
  selection, so Chat reasoning controls and model weights are rejected.

A recipe can contain several decisions, each with its own stages and budget.
For example, a keyword decision can send requests that mention an appeal
straight to Vega with one stage, while the default decision tries Kai → Vega.
Only the selected decision runs, and its budget tightens from the shared
request ceiling.

:::warning Keywords are hints, not authorization
Keywords inspect the task document, question text included, so callers can
trigger them deliberately. Don't use them for authorization or as a promise of
better accuracy.
:::

#### Choose acceptance evidence

An uncalibrated rule can inspect these observations:

| Field | Meaning |
| --- | --- |
| `top_probability` | Largest probability in the native answer distribution; for `noul`, `max(p, 1-p)` |
| `confidence` | The confidence reported by the native model, when present: System One's published statistic for Choice and Score (see [Decisions](https://vllm-sr.ai/docs/model-runtime/reference/http-api#decisions)); since runtime contract 3.2.0 its values differ from earlier releases, so recheck thresholds |
| `probability_margin` | For `noul`, `2 × abs(p - 0.5)`; both confident true and confident false answers have a large margin |

- **Bounds.** A rule bounds its field with `gt`, `gte`, `lt` or `lte`, in
  `[0, 1]`.
- **Targets.** A rule can target a `question_type`, a named `question` or
  both, and optionally a named `state`. Every required answer needs rule
  coverage.
- **Stage rules only add.** A stage-local `accept` rule adds a condition; it
  cannot weaken the algorithm's common quality rule.

A high model probability is not a measured accuracy guarantee.
`quality.type: calibrated` instead names an immutable resource in
`evaluation.calibrations`, and declares `loss: bundle_error` and an explicit
`max_risk`. The runtime must verify the artifact's hash and its fit to the
deployed model, task and arrival population; a missing or incompatible
artifact never silently becomes an uncalibrated pass.

#### Use independently certified acceptance

:::warning Experimental
Calibrated execution is a narrower path. It currently supports native stages
in recipes without signals or projections; judges and model-backed signals
need more inference provenance before they can use it. Uncalibrated stages can
still use the full routing path and an optional judge.
:::

To prepare a `systemone-calibration/v1` artifact:

1. **Freeze the task:** its question instructions, criteria and inference
   options, the policy, and each probability bin's acceptance decision.
   - Input text can vary; changing a question creates a different task
     template.
   - `options.return_meta` controls evidence visibility, and doesn't change
     the task identity.
2. **Collect a fresh certification cohort,** independent from training and
   threshold selection.
   - Count source bundles, not their questions or augmented copies.
   - The statistics assume i.i.d. draws from the declared conditional
     population; a stratified benchmark average does not automatically
     satisfy that.
3. **Record the full native arrival history** and the actual model identity
   from each response. The runtime checks the model, revision, content hash,
   engine, dtype, numerics and accelerator; a model name alone is not enough.
4. **Record bundle-error counts in each frozen bin,** and keep the source
   manifest, labeling rules and gate-definition digest for audit.
   - A bundle is erroneous when any required answer fails the cohort's
     predeclared correctness rule.
   - The artifact's `certification` identifies the evaluated population and
     declares the independent sampling method.

How the runtime uses the counts:

- It computes a one-sided Clopper–Pearson upper bound, with a Bonferroni
  correction across all declared bins.
- Every predeclared accepting bin must meet `max_risk`, or loading fails.
- Counts can't turn a rejecting bin into an accepting one, because that would
  change the arrival population of later stages. An unvisited bin can have
  zero samples only while it stays rejecting.

What the artifact binds:

- **The recipe's execution semantics and frozen gates,** not a global
  configuration hash. Moving credentials, changing listeners or editing
  explanatory text doesn't invalidate it.
- **Unresolved requests:** missing runtime identity, an unknown task, an error
  history or an uncovered bin leave the request unresolved.
- **Its limits:** the declarations still need a dataset audit. The bound
  describes the evaluated population, and promises nothing about distribution
  shifts or an individual request.

The Go artifact helpers are `CalibrationTaskSHA256`, `CalibrationPolicySHA256`
and `CalibrationGateSHA256` in `pkg/systemone`.

:::note
Pilot tuning results are not certification artifacts. Even zero errors in 96
independent examples gives an upper risk bound of about 3.1% at 95%
confidence, before correcting for multiple bins.
:::

#### Add an optional LLM judge

A `kind: judge` stage calls a declared OpenAI-compatible provider alias and
selects a complete response from earlier native attempts. It preserves that
response's native answer probabilities; an LLM's self-reported confidence does
not replace them. Declare `generation.max_output_tokens` to bound its output.

For an uncalibrated experiment:

- Keep a minimum acceptance floor in the common quality rule, and put stricter
  early-exit thresholds in the native stages' `accept` rules.
- The judge can then choose a valid earlier answer that meets the common
  floor, even if it missed an early-exit threshold.
- If the common rule rejects an answer, the judge cannot make it pass by
  selecting it.

A judge is additional evidence to evaluate, not an accuracy guarantee.

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

- **Compare on the same requests:** direct Kai, direct Vega and escalate.
- **Freeze before evaluating.** Choose acceptance thresholds on separate
  calibration data, then freeze the complete configuration.
- **Report together:** complete responses, unresolved requests, escalation
  rates and measured frontend latency.
- **Expect drift.** A threshold does not enforce a fixed fast-path fraction
  when the request distribution changes.

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
