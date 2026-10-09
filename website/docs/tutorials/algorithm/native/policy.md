# System One Learned Policy

## Overview

`policy` uses fitted, data-only parameters to choose the next declared System One model after observing a native answer. It shares the same model aliases, stage actions, quality rule and request budget as a handwritten cascade.

The first enabled native stage is the fast path. After each observation, the policy estimates which remaining native action offers useful error reduction relative to its measured cost. Its prediction selects work; it does not certify the correctness of the result.

## What Problem Does It Solve?

A fixed escalation order cannot use differences in which models correct which
errors. A learned policy estimates those differences from complete candidate
responses while keeping the operator's model and budget boundaries.

## When to Use

Use it after collecting a reproducible candidate matrix and establishing a
handwritten cascade baseline. Keep a held-out evaluation set to verify whether
adaptation improves your actual quality and cost tradeoff.

## Configuration

### Keep one execution contract

Start with the [cascade configuration](./cascade.md). Keep its `quality`, `stages`, model aliases and budget unchanged. Change `algorithm.type` and add the following `policy` block to that same algorithm:

```yaml
algorithm:
  type: policy
  # Keep the cascade's quality and stages here.
  policy:
    source: ./policies/predicted-gain.json
    sha256: "0000000000000000000000000000000000000000000000000000000000000000"
    cost_weight: 0.001
```

The all-zero digest is a placeholder. Replace the path and digest with the exact artifact produced by your experiment; the runtime rejects missing files, changed bytes and unsupported artifact versions.

The policy can only select enabled actions already declared in `stages`. It cannot add an endpoint, provide credentials, change the model roster, enlarge the call budget or weaken the common quality rule. Each stage runs at most once per request.

`cost_weight` penalizes an action's measured mean execution cost in milliseconds when comparing predicted quality gains. The artifact's `training.cost_metric` distinguishes runtime compute time from client elapsed time. Zero favors predicted gain alone. Choose the operating point using calibration data, then report its quality and realized cost on held-out requests. This parameter is not a currency price or a latency service-level guarantee.

An optional judge belongs at the end of the stages list. It is an authored fallback, rather than a learned action: when no native answer passes and the policy has no further native action, the executor can invoke this judge once within the remaining budget.

## Fit and evaluate separately

Use independent groups for fitting, calibration and held-out evaluation. Group requests from the same document, source or template together so near-duplicates do not cross the split.

Collect every candidate's response on the same admitted inputs. Preserve the native answer distributions, model revisions, failures and timing records. Training only on requests that an existing cascade already escalates makes counterfactual comparisons incomplete.

Collect identities through the endpoint you will deploy. A public Engine may expose an alias in its response's `model` field, so an identity collected directly from a private worker may not match the public path. Policy action bindings record the observed model ID and runtime metadata; deployment must not invent or rename that identity to make a different path pass validation.

Evaluate the learned policy and handwritten cascade against single-model baselines under the same budget and acceptance semantics. Report whole-bundle error alongside per-question metrics, unresolved requests, escalation frequency and total physical calls. Count failed requests rather than dropping them from quality or latency results.

The repository's [`tools/calibration/systemone_auto`](https://github.com/vllm-project/semantic-router/tree/main/tools/calibration/systemone_auto) contains the pilot collector and offline comparison workflow. Replay estimates help choose the next live experiment; confirm any deployment claim with actual frontend requests and transport timing.

## Quality calibration is separate

A policy's predicted improvement and a model's raw probability are different from an evaluated probability of correctness. Keep policy fitting and quality calibration as separate artifacts. If using `quality.type: calibrated`, the evidence must apply to the specific candidate and the requests that reach that stage, including the effect of earlier routing decisions.

Reusable fragment: [`config/fragments/algorithm/native/policy.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/native/policy.yaml).
