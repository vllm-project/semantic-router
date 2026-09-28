# User Feedback Signal

## Overview

`user-feedback` detects correction, dissatisfaction, or escalation feedback
from the conversation. Define its labels under
`routing.signals.user_feedbacks`.

This family is learned: it relies on the feedback detector configured under `global.model_catalog.modules.feedback_detector`. Leave `feedback_mapping_path` empty so the detector reads `id2label` from the model's `config.json`; set it only to a mapping file whose index order matches the model head.

## Key Advantages

- Lets the router react when users say the answer was wrong or unclear.
- Keeps escalation behavior visible inside routing decisions.
- Helps follow-up turns switch to stronger models or safer plugins.
- Reuses the same feedback detector across multiple routes.

## What Problem Does It Solve?

Follow-up turns often need different routing than the first answer. If the router ignores user feedback, it can keep repeating the same weak path after the user signals failure.

`user-feedback` solves that by exposing dissatisfaction and correction signals directly in the routing graph.

## When to Use

Use `user-feedback` when:

- follow-up corrections should escalate to a stronger model
- negative feedback should trigger more detailed or safer handling
- the router should react differently to “wrong answer” vs “need clarification”
- conversation state matters more than the original domain alone

## Configuration

```yaml
routing:
  signals:
    user_feedbacks:
      - name: wrong_answer
        description: User indicates the current answer is incorrect.
      - name: need_clarification
        description: User asks for a clearer or more detailed follow-up.
```

Define the feedback labels your decisions will consume, then let the learned detector decide which one matches each turn.

## Dependencies and Limitations

The feedback detector processes conversational text and can confuse quoted or
hypothetical complaints with real feedback. Evaluate it on follow-up traffic
and keep a normal fallback path.

The Router evaluates this signal only for a non-empty textual user turn after
an assistant answer. First turns, tool-result continuations, assistant prefills,
and user turns without text do not trigger feedback inference. The standalone
classification API expects the caller to supply a genuine follow-up; it rejects
empty input. Four-way classification cannot determine whether an arbitrary new
question is feedback, and even high confidence does not establish applicability.
A new topic later in a conversation can still trigger a false match; validate
that case before using feedback to change the selected model.

For existing four-class models, a prediction below the configured `threshold` is reported as `satisfied` with the model's own probability for that class beside it. That number is often far below the threshold, because the model put its mass on a class the threshold rejected. Read the pair as uncertain, not as evidence the user was satisfied. See a complete example:
[`config/fragments/signal/user-feedback/escalation.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/user-feedback/escalation.yaml).

Models that declare `NO_FEEDBACK` in their label mapping can distinguish ordinary
follow-up tasks from feedback. That result does not match any of the four
feedback rules. For these models, a prediction below the detector's configured
`threshold` is uncertain: it follows the decision's `on_unknown` policy instead
of activating `satisfied`. The standalone classification API preserves the
predicted label and probability and returns `abstained: true` for this case.
Existing four-class models retain their earlier threshold behavior.

When a reachable routing decision depends on this signal, its configured model
must initialize successfully or Router startup fails. A model used only by the
standalone diagnostics API remains best-effort and does not block unrelated
routes when it is unavailable.
