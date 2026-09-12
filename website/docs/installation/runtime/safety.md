---
title: Safety models
description: Configure prompt guard, PII, and hallucination checks and choose how to handle failures.
---

Safety models detect risks; routing decisions and plugins determine the action.
Enabling a model alone does not block or redact a request.

| Check | What it detects | Configure the action |
| --- | --- | --- |
| Prompt guard | Prompt injection and jailbreaks | [Jailbreak signals](../../tutorials/signal/learned/jailbreak.md) |
| PII | Personal information in text | [PII signals](../../tutorials/signal/learned/pii.md) |
| Hallucination | Answer claims unsupported by supplied context | [Hallucination plugin](../../tutorials/plugin/hallucination.md) |
| Fact-check | Whether a request needs factual verification | [Fact-check signals](../../tutorials/signal/learned/fact-check.md) |

## Prompt guard

To enable the maintained local guard, merge this into your configuration:

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

Then add the jailbreak signal and decision that should handle a match. To use
a separately hosted model, follow [External services](external.md).
The recipe's `prompt_guard` binding selects the model; the threshold and
routing policy remain in their existing settings.

## PII

Use a complete token-classification checkpoint with the matching PII label map.
Select it through the recipe's `pii_classifier` binding with
`contract: token_spans.v1`. Set its deployment and adapter as in
[In-process models](in-process.md), and provide the checkpoint's
`mapping_path`. The [PII guide](../../tutorials/signal/learned/pii.md) covers
entity thresholds and redaction.

External PII services must return scored entities and valid Unicode text
positions. An invalid response is an error, not an empty successful scan.

## Hallucination detection

The local detector checks an answer against its context and question. An
optional NLI explainer checks whether a premise supports a hypothesis.
Enable the maintained models with:

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

These local models use Candle. A remote chat service can replace the detector;
NLI still requires a supported local explainer. Configure how context is
supplied and how detected spans are handled in the
[hallucination guide](../../tutorials/plugin/hallucination.md).

## Handle failures and missing scores

A model error produces an unknown result. The decision's `rules.on_unknown`
chooses `no_match`, `match`, or `fail_request`. Without that setting, prompt
guard uses `on_error`: `allow` means no match, while `block` means a policy match.
The consuming decision still determines the resulting action.

A chat verdict or policy fallback may have no confidence score. Diagnostics
show `confidence: null` with `confidence_available: false`; this is different
from a model score of zero. Test both model matches and service failures when
setting the policy.

For custom safety checkpoints, see the
[training and export guide](../../training/mmbert-safety-classifier.md).

For access control and rate limits, see [Security hardening](../security-hardening.md).
