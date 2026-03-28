# User Feedback Signal

## Overview

`user-feedback` detects correction, dissatisfaction, or escalation feedback from the conversation. It maps to `config/signal/user-feedback/` and is declared under `routing.signals.user_feedbacks`.

This family is learned: it relies on the feedback detector configured under `global.model_catalog.modules.feedback_detector`.

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

Source fragment family: `config/signal/user-feedback/`

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
