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

Matching is exact and case-sensitive, the same as the `metadata` signal:
`equals: reviewer` does not match a declared role of `Reviewer`.

## Replay

When [Router Replay](../../learning/memory-and-replay) is enabled, each record
shows what happened to the envelope in `route_diagnostics`:

| Field | Values |
| --- | --- |
| `agentic_facts_status` | `accepted` or `rejected` |
| `agentic_facts_reasons` | one entry per rejection, as `field:reason`, or a bare reason when the whole envelope failed |

A rejected envelope:

```json
"route_diagnostics": {
  "agentic_facts_status": "rejected",
  "agentic_facts_reasons": ["expires_at:expired", "lineage.depth:too_deep"]
}
```

An envelope sent without the trust marker:

```json
"agentic_facts_reasons": ["untrusted"]
```

Both fields are absent when the contract is disabled or the request carried no
envelope, so existing records do not change.

Replay never stores a value the caller sent. Reasons contain only schema field
names and reason codes. To see which rule matched, read `signals.agentic_facts`,
which lists the operator's rule names. Callers without the `replay.detail`
permission see the status, but `agentic_facts_reasons` is returned empty.

A request refused because required capabilities leave no eligible model still
writes a record, with lifecycle `failed`, terminal reason `selection_rejected`,
and `agentic_facts_status: accepted`.

## Required Capabilities

An accepted envelope may list `required_capabilities`. They narrow the models a
decision already lists in `modelRefs`; they never add one. They apply only in a
recipe that opts into strict candidate requirements:

```yaml
recipes:
  - name: agent-workloads
    routing:
      candidate_requirements:
        capabilities: declared
```

In that mode a model stays a candidate only when its model card declares every
capability the request needs, both those derived from the request itself and
those the caller requires. A model that declares no capabilities is excluded,
so every model the recipe can route to must declare them. Without this opt-in,
`required_capabilities` is validated but changes nothing.

A route action's destination is checked the same way. If it lacks a required
capability, a capable model from the decision's `modelRefs` serves the request.
When no model qualifies, the request fails with `503` and the code
`no_eligible_model`. The response names no model and no capability.

## Dependencies and Limitations

`required_capabilities` accepts only canonical protocol capability names, such
as `text`, `tools`, `reasoning`, `structured_json`, or `image_input`. Names are
trimmed and matched without regard to case. An unknown name, including a model
card alias such as `vision` or `tool_use`, rejects the whole envelope with
`required_capabilities:malformed`.

`agentic_facts` only evaluates facts accepted at the request boundary; it never
parses an envelope that arrives without the configured trust marker. See a
complete example:
[`config/fragments/signal/agentic-facts/delegated-role.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/agentic-facts/delegated-role.yaml).
