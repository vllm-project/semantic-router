# Metadata Signal

## Overview

`metadata` matches bounded string values supplied by the caller in request
metadata. It is intended for deterministic application hints such as consent,
cohort, or workload class.

Metadata is untrusted input. Authorization and authenticated identity continue
to use the `authz` signal and trusted headers.

## Key Advantages

- routes on explicit application context without inferring it from prompt text
- keeps untrusted hints separate from authenticated identity
- supports reusable named rules with exact, set-membership, or existence tests

## What Problem Does It Solve?

Some routing facts do not belong in the prompt. A caller may know that a request
is part of a canary cohort or that remote-processing consent was denied.
Metadata signals expose those facts to decisions without passing them to model
selectors.

## When to Use

Use metadata signals for non-authoritative application hints. Do not use them to
grant permissions, bypass guardrails, or establish user identity.

## Configuration

```yaml
document:
  signals:
    metadata:
      - name: consent-denied
        key: consent
        predicate:
          equals: denied
      - name: canary-cohort
        key: cohort
        predicate:
          in: [beta, canary]
      - name: has-workload-class
        key: workload_class
        predicate:
          exists: true
```

Exactly one predicate comparator is required. Request metadata values are
strings and are evaluated before decision matching. Rule names and keys must be
trimmed. Requests accept at most 32 entries, 128-byte keys, and 1024-byte
values.

Chat Completions, Anthropic Messages, `/api/v1/classify/intent`, and
`/api/v1/eval` all accept the same top-level string map:

```json
{
  "metadata": {
    "consent": "denied",
    "cohort": "canary"
  }
}
```

## Dependencies and Limitations

Metadata is caller-controlled and is not forwarded to model selectors. It must
not grant privileges or bypass safety policy; use `authz` for trusted identity.
See a complete example:
[`config/fragments/signal/metadata/routing-hints.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/metadata/routing-hints.yaml).
