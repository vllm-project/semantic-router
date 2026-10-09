# System One Cascade

## Overview

`cascade` tries declared decision models in order and returns a complete System One response when its acceptance rules pass. Use a small model first, then spend more compute only on requests that need another attempt.

This is an experimental native algorithm for `choice`, `score`, and `noul` requests. It keeps the original question bundle together. It does not turn native answers into Chat messages or invent confidence for a missing answer.

## What Problem Does It Solve?

Always using the largest decision model spends its full cost on easy requests.
A cascade lets early answers stop after an explicit evidence check and sends
harder requests to another declared model.

## When to Use

Use it when you have a native decision workload, at least one candidate model
and an acceptance rule you can evaluate. Start with a handwritten order when
you want a transparent baseline before fitting a policy.

## Configuration

### Connect the models

Model aliases are shared resources under `providers.models`. A local alias refers to a model runtime deployment; a remote alias refers to a concrete Engine or compatible System One endpoint. Choose one transport per alias.

```yaml
providers:
  models:
    - name: kai
      api_format: systemone
      deployment: local-kai
    - name: nox
      api_format: systemone
      provider_model_id: vllm-sr/Decision-2.0-Nox-4B
      backend_refs:
        - provider: systemone-compatible
          base_url: http://localhost:8900/v1
          api_key_env: DECISION_BACKEND_TOKEN

global:
  model_catalog:
    deployments:
      local-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        device: cpu
```

`provider_model_id` is the model name sent to the remote endpoint. Credentials stay on the provider binding; they never belong in a learned policy file. Replicas of the same remote model go in its `backend_refs`; different models are separate aliases and stage actions.

## Publish an auto entrypoint

An API-scoped entrypoint makes the recipe available as `vllm-sr/auto`. The listener must explicitly allow that native name. Chat and System One aliases are separate: publishing this entrypoint does not change the Chat default recipe.

```yaml
listeners:
  - name: inference
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

The following fragment accepts `choice` requests. Its probability threshold is illustrative; evaluate it on your own workload before use.

```yaml
recipes:
  - name: native-cascade
    routing:
      budget: {deadline: 3s, max_calls: 3}
      decisions:
        - name: classify
          rules: {}
          modelRefs: [{model: kai}, {model: nox}]
          algorithm:
            type: cascade
            quality:
              type: uncalibrated
              acceptance:
                rules:
                  - question_type: choice
                    field: top_probability
                    predicate: {gte: 0.9}
            stages:
              - {name: fast, kind: native, model: kai}
              - {name: strong, kind: native, model: nox, timeout: 1s}
```

`modelRefs` is the complete candidate roster. Each stage names one member and can set its own timeout. Stage names cannot use `abstain`, which is reserved for the judge’s no-selection result. A stage cannot add a model that the decision did not declare. The shared deadline and call limit cover this Router's physical inference exchanges, including model-backed signals and transport retries; advancing a stage does not reset either limit. Internal work behind an opaque external provider is not visible to this call ledger.

The Router validates every required answer before accepting a response. Missing answers, invalid values, unsupported question types and unproven full-input coverage do not become successful results. If no declared stage satisfies the rule within the budget, the request is unresolved.

## Try the route

After starting the Router with this configuration, send a request to its native
endpoint. The question names and criteria reach each attempted model unchanged.
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
        "criteria": {
          "account": "Account access or authentication",
          "billing": "Payments, invoices or refunds"
        }
      }
    }
  }'
```

The response's `routing` field reports the selected model, stage and physical
model-call count. A `503` with `systemone_unresolved` means no complete answer
passed the configured rule; it does not mean that an omitted answer was false.
See the [Router API](../../../api/router.md#route-a-system-one-request) for the
response fields and error contract.

In Dashboard, open **System One → Decision Playground**, then choose
`vllm-sr/auto` under **Automatic routes**. Load an example supported by your
recipe or enter your own questions, then select **Run**. Dashboard uses your
signed-in management permissions, so you do not need to paste a public API key.
**Decision Monitoring** shows stage outcomes, request rates and latency. These
describe live execution; evaluate labeled requests separately to measure
accuracy.

## Add routing signals

Native signals inspect the complete task document, including its states and questions. The request does not supply a Chat conversation or trusted identity envelope. References to `authz`, `metadata`, `conversation`, `reask`, `user_feedback` and `input_modality` therefore fail configuration validation, including through projections. KB signals also remain unavailable until their model calls participate in the native request budget.

Preference signals must use a native decision task deployment. Explicit contrastive or external preference adapters and MCP domain classifiers are unavailable in native recipes until they share the request lifecycle. Remote classifier and embedding transports account for each physical retry in the same call budget.

The recipe's `strategy` still chooses among matching decisions. Native execution uses its explicit stages and budget rather than Chat fallback, candidate requirements, decision reliability, output contracts or adaptation controls. Keep inherited Chat fallback disabled for this recipe. `modelRefs` declares aliases only; stage order or the learned policy determines selection, so Chat reasoning controls and model weights are rejected. Provider-level retry and timeout settings still apply to remote exchanges.

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

Calibrated execution is an experimental, narrower path. It currently supports native stages in recipes without signals or projections. Judges and model-backed signal paths need additional inference provenance before they can use certified acceptance. Uncalibrated cascades and policies can still use the full routing path and an optional judge.

To prepare a `systemone-calibration/v1` artifact:

1. Freeze the task's question instructions, criteria and inference options, the policy, and each probability bin's acceptance decision. Input text can vary; changing a question creates a different task template. `options.return_meta` controls evidence visibility and does not change the task identity.
2. Collect a fresh certification cohort independently from training and threshold selection. Count source bundles, rather than their questions or augmented copies. The statistical assumption is i.i.d. draws from the declared conditional population; a stratified benchmark average does not automatically satisfy it.
3. Record the full native arrival history and actual model identity from each response. The runtime checks the model, revision, content hash, engine, profile, numerics and accelerator. A model name alone is insufficient.
4. Record bundle-error counts in each frozen bin and retain the source manifest, labeling rules and gate-definition digest for audit. A bundle is erroneous when any required answer fails the cohort's predeclared correctness rule. The artifact's `certification` identifies this evaluated population and declares the independent sampling method.

The runtime computes a one-sided Clopper–Pearson upper bound from the counts, with a Bonferroni correction across all declared bins. Every predeclared accepting bin must meet `max_risk`, or loading fails. Counts cannot turn a previously rejecting bin into an accepting bin: doing so would change the arrival population of later stages. An unvisited bin can have zero samples only when it remains rejecting.

The artifact binds the recipe's execution semantics and frozen gates, rather than a global configuration hash. Moving credentials, changing listeners or updating explanatory text does not invalidate it. Missing runtime identity, an unknown task, an error history or an uncovered bin leaves the request unresolved. An artifact's declarations still require a dataset audit; the bound describes its evaluated population and makes no guarantee about distribution shifts or an individual request.

The Go artifact helpers are `CalibrationTaskSHA256`, `CalibrationPolicySHA256` and `CalibrationGateSHA256` in `pkg/systemone`. Pilot tuning results are not certification artifacts: even zero errors in 96 independent examples gives an approximately 3.1% upper risk bound at 95% confidence before correcting for multiple bins.

## Add an optional LLM judge

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
    model: nox
    accept:
      rules:
        - {question_type: choice, field: top_probability, predicate: {gte: 0.9}}
  - name: review
    kind: judge
    model: reviewer
    generation: {max_output_tokens: 256}
```

Add `reviewer` to the decision's `modelRefs` and connect it with
`api_format: openai`. The judge is opt-in and consumes the same request deadline
and call budget. Its output is a selection from prior complete native
responses, not a replacement answer synthesized by an LLM.

## Compare with a learned policy

Keep the same aliases, stages and request budget when comparing the handwritten order with a [learned policy](./policy.md). Record complete responses, failures, escalation rates and measured end-to-end latency. Offline replay helps select experiments but does not establish live serving latency.

Reusable fragment: [`config/fragments/algorithm/native/cascade.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/native/cascade.yaml).
