---
title: Safety Models
description: Configure safety inference and preserve the distinction between model evidence and policy actions.
---

# Safety models

Safety inference produces evidence for signals and plugins. Decisions determine
what to do with it. Deployments select execution; module and route settings
retain thresholds, matching, and failure policy.

Identity, authorization, and rate limits are shared services rather than model
runtime tasks. Configure those through [Security hardening](../security-hardening.md)
and [Authorization signals](../../tutorials/signal/heuristic/authz.md).

## Prompt guard

The maintained local default uses the `mmbert32k` variant. Its Router task
still has a 512-token maximum. Merge this module fragment when the selected
checkpoint and mapping are available:

```yaml
global:
  model_catalog:
    modules:
      prompt_guard:
        enabled: true
        variant: mmbert32k
        threshold: 0.7
        on_error: block
```

Use a recipe's `prompt_guard` binding to select a different supported local
model or the [named external guard](external.md#bind-a-named-guardrail-service).
A binding does not create the consuming jailbreak rule or blocking plugin.
Configure that policy in the recipe.

Sequence guards preserve the model's complete label distribution and the
configured positive labels. Chat guards preserve their categorical decision.
A returned “unsafe” verdict does not imply confidence `1.0`; an unreported
score remains unavailable. Negative labels are matched explicitly, so the
word `unsafe` is not accidentally treated as containing a safe verdict.

## Classifier failure is not a model score

A failed or invalid guardrail inference records a signal error and evaluates
to `Unknown`. A consuming decision's root `rules.on_unknown` resolves it as
`no_match`, `match`, or `fail_request`. A global `fail_request` resolution
rejects the request even if another decision matches.

When that root policy is absent, the existing `prompt_guard.on_error` applies:
`allow` maps the failure to no match; `block` maps it to a policy match. The
failure itself has **no probability**. Diagnostics retain
`signal_error_matches`; unavailable confidence is JSON `null` with its
availability flag false. Consumers must not convert it to zero or one.

A request is blocked only if its consuming decision acts on that result.
For response scanning, the `response_jailbreak` plugin's action remains
`block`, `header`, or `none`. Generic classifier condition-level `on_error`
is a separate compatibility policy; see
[Classifier signals](../../tutorials/signal/learned/classifier.md).

## PII

`pii_classifier` uses `token_spans.v1`, with labels mapped to configured PII
types. Local token adapters retain real scores, BIO entity boundaries, UTF-8
byte offsets, and input/truncation metadata. The PII consumer excludes outside
labels before applying its configured thresholds. It does not fabricate a
score for an unscored entity.

Remote PII must implement the [span protocol](external.md#pii-spans-and-grounding-inputs).
A malformed or partial result is not equivalent to an empty complete scan.
Use a complete compatible token checkpoint and mapping, such as the explicit
[PII binding example](models-and-bindings.md#declare-one-local-deployment).
Keep PII actions and response redaction in their consuming recipe/plugin.

## Grounding and NLI

The local hallucination detector takes **context, question, and answer**.
It preserves the answer portion of its supported input window and returns
answer-relative hallucination spans. The separate NLI explainer takes a
**premise and hypothesis**, preserves the hypothesis, and returns the
entailment/neutral/contradiction distribution. These are not ordinary
single-text token and sequence requests.

```yaml
global:
  model_catalog:
    modules:
      hallucination_mitigation:
        enabled: true
        detector:
          backend: candle
          model_ref: hallucination_detector
          threshold: 0.82
        explainer:
          model_ref: hallucination_explainer
          threshold: 0.9
```

The system aliases resolve to their catalog artifacts unless a recipe binding
overrides them. Both dedicated local tasks currently use Candle; ORT and
HTTP NLI are not implemented. Native classification input is bounded by the
actual task limit and at most 512 tokens, including its paired/structured
preprocessing.

The external hallucination detector uses its existing `backend: endpoint`
settings or an HTTP deployment binding with `adapter: http_chat`. It does not
supply a local NLI explainer. Chat-derived spans retain unavailable confidence;
an explanation string is not a scored entailment result. Route features that
require NLI must have an actual supported explainer or follow their configured
failure policy.

## Fact-check, feedback, and modality

Fact-check and feedback are sequence tasks with their own declared mappings.
Their empty-input policy defaults retain the established label and expose
`policy_default: empty_text`, `confidence: null`, and
`confidence_available: false`. Disabled or unavailable inference also does
not supply a model probability.

The output-modality task classifies `AR`, `DIFFUSION`, or `BOTH`. Its optional
keyword/hybrid decision rules remain consumer policy. Image/audio embedding
capability is a separate feature of an embedding checkpoint; an output-modality
classifier does not imply that capability.

## Choose and validate artifacts

Keep model files, label order, thresholds, and intended data domain together
when changing a safety model. A model or directory name is not evidence that
a checkpoint is merged or that its labels match a task. Custom complete local
artifacts do not require registry registration, but preparation still checks
actual architecture, tensors, tokenizer, and labels.

The [mmBERT safety training guide](../../training/mmbert-safety-classifier.md)
covers training and export. Newly trained models and unmerged adapters are not
automatically installed or certified by this runtime. Use a generic classifier
for a compatible separately trained label head; do not assume a new built-in
`safety.<name>` binding exists. Verify representative positive, negative,
Unicode, boundary-length, and failure cases before changing production policy.
