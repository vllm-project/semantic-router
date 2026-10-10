# System One Cascade

## Overview

`cascade` tries declared decision models in order and returns a complete System One response when its acceptance rules pass. Use a small model first, then spend more compute only on requests that need another attempt.

This is an experimental native algorithm for `choice`, `score`, and `noul` requests. It keeps the original question bundle together. It does not turn native answers into Chat messages or invent confidence for a missing answer.

## What Problem Does It Solve?

Always using the largest decision model spends its full cost on easy requests.
A cascade lets early answers stop after an explicit evidence check and sends
harder requests to another declared model. This guide starts with **Decision 2.0
Kai 0.6B → Vega 27B**. Kai answers the original request; the cascade checks that
answer before deciding whether to call Vega. It does not make a separate
mandatory Kai classification call or rerun the recipe after the first answer.

## When to Use

Use it when you have a native decision workload, at least one candidate model
and an acceptance rule you can evaluate. An explicit stage order makes the
escalation path easy to inspect and compare with direct-model baselines.

## Configuration

### Connect the models

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
          base_url: http://localhost:8900/v1

global:
  model_catalog:
    deployments:
      local-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        device: auto
```

`provider_model_id` is the model name sent to the remote endpoint. If a backend requires authentication, set `api_key_env` on its binding. Credentials stay on that binding. Replicas of the same remote model go in its `backend_refs`; different models are separate aliases and stage actions.

## Publish an auto entrypoint

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
    recipe: native-cascade
```

Add client authentication to the listener before exposing it beyond a trusted development environment. To allow a concrete model as well, add its native provider alias to `systemone.models`.

## Define the cascade

The following fragment accepts `choice`, `score` and `noul` requests. Kai exits
early only when every answer passes its type-specific gate. The threshold below
is a frozen pilot operating point for Decision 2.0 Kai → Vega, not a general
recommended default or an accuracy guarantee. The [reproduction guide](https://github.com/vllm-project/semantic-router/tree/main/tools/calibration/systemone_auto)
pins that experiment’s models and public benchmark. Choose thresholds on
independent calibration data for your own workload, then freeze them before
evaluation.

```yaml
recipes:
  - name: native-cascade
    routing:
      decisions:
        - name: answer
          rules: {}
          modelRefs: [{model: kai}, {model: vega}]
          algorithm:
            type: cascade
            budget: {deadline: 10s, max_calls: 2}
            quality:
              type: uncalibrated
              acceptance:
                rules:
                  - {question_type: choice, field: top_probability, predicate: {gte: 0}}
                  - {question_type: score, field: top_probability, predicate: {gte: 0}}
                  - {question_type: noul, field: top_probability, predicate: {gte: 0}}
            stages:
              - name: fast
                kind: native
                model: kai
                accept:
                  rules:
                    - {question_type: choice, field: top_probability, predicate: {gte: 0.6059704079536342}}
                    - {question_type: score, field: top_probability, predicate: {gte: 0.6059704079536342}}
                    - {question_type: noul, field: top_probability, predicate: {gte: 0.6059704079536342}}
              - {name: strong, kind: native, model: vega}
```

The common `gte: 0` rules require a valid distribution for every answer; they
do not claim an accuracy floor. Vega can return a complete valid answer after
Kai fails its stricter early-exit gate. This sample has no model-backed routing
signals, so its two-call budget covers Kai and, when needed, Vega. If a transport
retry consumes a call, fewer calls remain for later stages.

`modelRefs` is the complete candidate roster. Each stage names one member and can set its own timeout. Stage names cannot use `abstain`, which is reserved for the judge’s no-selection result. A stage cannot add a model that the decision did not declare. The algorithm's deadline starts after its decision is selected. Its call limit covers its physical inference exchanges, including transport retries; advancing a stage does not reset either limit. Signal evaluation has its own timeouts and follows request cancellation; signal calls do not consume the selected algorithm's budget. Internal work behind an opaque external provider is not visible to this call ledger.

The Router validates every required answer before accepting a response. Missing answers, invalid values, unsupported question types and unproven full-input coverage do not become successful results. If no declared stage satisfies the rule within the budget, the request is unresolved.

The algorithm deadline is not a whole-request timeout. HTTP server, proxy and
client timeouts still apply to the complete request. In particular, Dashboard's
operator diagnostics use the management API's two-minute response write limit;
configure the full serving path appropriately for longer requests.

## Try the route

Combine the model, listener, entrypoint and recipe snippets in a configuration
file with `version: v0.3`, connect the Vega Engine endpoint, and start the Router:

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

## Add routing signals

Native signals inspect the complete task document, including its states and questions. The request does not supply a Chat conversation or trusted identity envelope. References to `authz`, `metadata`, `conversation`, `reask`, `user_feedback` and `input_modality` therefore fail configuration validation, including through projections. KB signals are also unavailable in native recipes.

Preference signals must use a native decision task deployment. Explicit contrastive or external preference adapters and MCP domain classifiers are unavailable in native recipes until they share the request lifecycle. Supported model-backed signals retain their own transport timeouts and request cancellation. They run before the selected algorithm starts its budget.

The recipe's `strategy` still chooses among matching decisions. Native execution uses its explicit stages and budget rather than Chat fallback, candidate requirements, decision reliability, output contracts or adaptation controls. Keep Chat fallback disabled for this native recipe. `modelRefs` declares aliases only; stage order determines selection, so Chat reasoning controls and model weights are rejected. Provider-level retry and timeout settings still apply to remote exchanges.

### Choose a branch before running its cascade

A recipe can contain several decisions. For example, send requests mentioning
an appeal directly to Vega, and use Kai → Vega for other requests. Replace the
recipe above with this fragment; keep its provider bindings and entrypoint.

```yaml
recipes:
  - name: native-cascade
    routing:
      strategy: priority
      signals:
        keywords:
          - name: appeal
            operator: OR
            keywords: ["appeal", "dispute"]
            case_sensitive: false
      decisions:
        - name: review-first
          priority: 100
          rules:
            operator: OR
            conditions:
              - {type: keyword, name: appeal}
          modelRefs: [{model: vega}]
          algorithm:
            type: cascade
            budget: {deadline: 10s, max_calls: 1}
            quality:
              type: uncalibrated
              acceptance:
                rules:
                  - {question_type: choice, field: top_probability, predicate: {gte: 0}}
                  - {question_type: score, field: top_probability, predicate: {gte: 0}}
                  - {question_type: noul, field: top_probability, predicate: {gte: 0}}
            stages:
              - {name: review, kind: native, model: vega}
        - name: small-first
          priority: 0
          rules: {}
          modelRefs: [{model: kai}, {model: vega}]
          algorithm:
            type: cascade
            budget: {deadline: 10s, max_calls: 2}
            quality:
              type: uncalibrated
              acceptance:
                rules:
                  - {question_type: choice, field: top_probability, predicate: {gte: 0}}
                  - {question_type: score, field: top_probability, predicate: {gte: 0}}
                  - {question_type: noul, field: top_probability, predicate: {gte: 0}}
            stages:
              - name: fast
                kind: native
                model: kai
                accept:
                  rules:
                    - {question_type: choice, field: top_probability, predicate: {gte: 0.9}}
                    - {question_type: score, field: top_probability, predicate: {gte: 0.9}}
                    - {question_type: noul, field: top_probability, predicate: {gte: 0.9}}
              - {name: strong, kind: native, model: vega}
```

Only the selected decision executes. A one-stage cascade uses the same typed
validation and acceptance contract for a single model; the other decision has
its own two-call limit. Both budgets begin after signal evaluation. The keyword
signal needs no model inference. The simpler first example has no routing
signal at all: Kai's one original answer supplies its early-exit evidence.

Keywords inspect the task document, including question text, and callers can
trigger them deliberately. Use them as routing hints, not authorization or a
promise of better accuracy. Evaluate this branch selection separately from the
basic cascade; the `0.9` thresholds remain illustrative.

## Choose acceptance evidence

An uncalibrated rule can inspect these observations:

| Field | Meaning |
| --- | --- |
| `top_probability` | Largest probability in the native answer distribution; for `noul`, `max(p, 1-p)` |
| `confidence` | The confidence reported by the native model, when present |
| `probability_margin` | For `noul`, `2 × abs(p - 0.5)`; both confident true and confident false answers have a large margin |

Predicates use `gt`, `gte`, `lt`, or `lte`, with bounds in `[0, 1]`. A rule can target a `question_type`, a named `question`, or both, and optionally a named `state`. All required answers need rule coverage. A stage-local `accept` rule adds a condition; it cannot weaken the algorithm's common quality rule.

A high model probability is not a measured accuracy guarantee. `quality.type: calibrated` instead names an immutable resource in `evaluation.calibrations`, declares `loss: bundle_error` and an explicit `max_risk`. The runtime must verify both the artifact's hash and its applicability to the deployed model, task and arrival population. A missing or incompatible artifact must not silently become an uncalibrated pass.

### Use independently certified acceptance

Calibrated execution is an experimental, narrower path. It currently supports native stages in recipes without signals or projections. Judges and model-backed signal paths need additional inference provenance before they can use certified acceptance. Uncalibrated cascades can still use the full routing path and an optional judge.

To prepare a `systemone-calibration/v1` artifact:

1. Freeze the task's question instructions, criteria and inference options, the policy, and each probability bin's acceptance decision. Input text can vary; changing a question creates a different task template. `options.return_meta` controls evidence visibility and does not change the task identity.
2. Collect a fresh certification cohort independently from training and threshold selection. Count source bundles, rather than their questions or augmented copies. The statistical assumption is i.i.d. draws from the declared conditional population; a stratified benchmark average does not automatically satisfy it.
3. Record the full native arrival history and actual model identity from each response. The runtime checks the model, revision, content hash, engine, profile, numerics and accelerator. A model name alone is insufficient.
4. Record bundle-error counts in each frozen bin and retain the source manifest, labeling rules and gate-definition digest for audit. A bundle is erroneous when any required answer fails the cohort's predeclared correctness rule. The artifact's `certification` identifies this evaluated population and declares the independent sampling method.

The runtime computes a one-sided Clopper–Pearson upper bound from the counts, with a Bonferroni correction across all declared bins. Every predeclared accepting bin must meet `max_risk`, or loading fails. Counts cannot turn a previously rejecting bin into an accepting bin: doing so would change the arrival population of later stages. An unvisited bin can have zero samples only when it remains rejecting.

The artifact binds the recipe's execution semantics and frozen gates, rather than a global configuration hash. Moving credentials, changing listeners or updating explanatory text does not invalidate it. Missing runtime identity, an unknown task, an error history or an uncovered bin leaves the request unresolved. An artifact's declarations still require a dataset audit; the bound describes its evaluated population and makes no guarantee about distribution shifts or an individual request.

The Go artifact helpers are `CalibrationTaskSHA256`, `CalibrationPolicySHA256` and `CalibrationGateSHA256` in `pkg/systemone`. Pilot tuning results are not certification artifacts: even zero errors in 96 independent examples gives an approximately 3.1% upper risk bound at 95% confidence before correcting for multiple bins.

## Add an optional LLM judge

This opt-in extension is separate from the Kai → Vega example above. Evaluate
its quality and latency independently before adding it to your serving path.

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
  - name: fast
    kind: native
    model: kai
    accept:
      rules:
        - {question_type: choice, field: top_probability, predicate: {gte: 0.9}}
  - name: strong
    kind: native
    model: vega
    accept:
      rules:
        - {question_type: choice, field: top_probability, predicate: {gte: 0.9}}
  - name: review
    kind: judge
    model: reviewer
    generation: {max_output_tokens: 256}
```

Add `reviewer` to the decision's `modelRefs` and connect it with
`api_format: openai` and increase this algorithm's budget to allow the third
call. The judge consumes that same algorithm deadline and call budget. Its
output is a selection from prior complete native responses, not a replacement
answer synthesized by an LLM.

## Measure the tradeoff

Compare direct Kai, direct Vega and the cascade on the same requests. Choose
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

Reusable fragment: [`config/fragments/algorithm/native/cascade.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/native/cascade.yaml).
