# Agentic Facts Signal

## Overview

`agentic_facts` matches validated, trusted facts about an external agent that
delegated the current request: which role it declared (`delegated_role`) and
what phase of its task it is in (`task_phase`). Unlike the `metadata` signal,
these facts are never taken from an arbitrary caller-supplied map. They come
from a single configured carrier header, are accepted only when the request
also carries the configured trust marker, and are validated against a
versioned envelope schema before a rule can ever see them.

## Key Advantages

- keeps a trust boundary between operator-declared agent facts and ordinary
  untrusted request metadata
- routes on delegated role or task phase without parsing prompt text
- fails closed on the whole envelope rather than accepting a partially valid
  one: an untrusted or malformed envelope contributes no rule matches

## What Problem Does It Solve?

A multi-agent workflow may delegate a subtask to another agent acting in a
specific role, such as a reviewer or an executor. That context is useful for
routing, but it must not be trusted just because a client claims it. This
signal only evaluates facts that already passed the request-boundary trust
check and schema validation, so a decision that reads `agentic_facts` can rely
on the same guarantee `authz` gives for identity.

## When to Use

Use `agentic_facts` to route based on a delegated agent's declared role or task
phase once the request boundary is configured to accept and validate the
envelope. Do not use it as an authorization mechanism on its own; it narrows or
informs selection, it does not replace `authz`.

## Configuration

```yaml
routing:
  signals:
    agentic_facts:
      - name: reviewer_role
        description: Caller declared itself as a reviewer subagent.
        field: delegated_role
        predicate:
          equals: reviewer
      - name: execution_phase
        description: Caller is in an execution or verification task phase.
        field: task_phase
        predicate:
          in:
            - execute
            - verify
```

`field` is one of `delegated_role` or `task_phase`. Exactly one predicate
comparator, `equals` or `in`, is required per rule.

## Dependencies and Limitations

`agentic_facts` only evaluates facts accepted at the request boundary; it never
parses an envelope that arrives without the configured trust marker. See a
complete example:
[`config/fragments/signal/agentic-facts/delegated-role.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/agentic-facts/delegated-role.yaml).
