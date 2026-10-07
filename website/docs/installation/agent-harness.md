---
title: Connect an agent harness
description: Connect an agent harness to a stable model API with explicit routing policy, model limits, and session continuity.
---

# Connect an agent harness

Point your harness at a Router entrypoint such as `vllm-sr/auto`. The harness
owns the agent loop, tools, and task state; the Router selects models or runs a
bounded multi-model workflow under your policy. Inference backends execute the
model calls.

[Install the Router](/docs/installation) first, or
[ask an agent to install it](agent).

## Choose the inference connection

Configure the harness's model provider. Setting names vary by harness version.

| Setting | What to use |
| --- | --- |
| Inference address | The public inference listener or gateway. The local Quickstart uses `http://localhost:8899`; the management API and Dashboard have different addresses. |
| Client protocol | OpenAI Chat Completions, OpenAI Responses, or Anthropic Messages, matching the requests your harness sends. |
| Public model ID | `vllm-sr/auto` for the default policy, or a published entrypoint from `GET /v1/models`. |
| Authentication | The credentials required by your inference listener or gateway. Backend provider credentials are configured separately by the operator. |
| Model limits and capabilities | Context, output, tools, vision, and reasoning settings that the selected recipe can support. |

Clients may append `/v1` themselves. Check that the final path matches:

| Client API | Final request path | Requirement |
| --- | --- | --- |
| Chat Completions | `POST /v1/chat/completions` | Send the Chat Completions request shape. |
| Responses | `POST /v1/responses` | The Router's `global.services.response_api` and its store must be available. |
| Messages | `POST /v1/messages` | Send the Messages request shape and an appropriate `anthropic-version` header. |

Client and backend protocols can differ, within the [compatibility
matrix](protocol-compatibility). Free-form custom tools cannot reach a Messages
backend; Codex CLI has additional limits listed there. Keep credentials in the
harness's supported secret store or environment bindings.

## Verify the public model

Configure at least one backend, then list the local stack's public models:

```bash
curl -sS http://localhost:8899/v1/models
```

Use a [virtual entrypoint](../tutorials/global/entrypoints) to apply routing
policy. Concrete provider model names bypass recipe routing.

Send a minimal request and include response headers:

```bash
curl -sS -i http://localhost:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

For other deployments, use their inference address and authentication. Check
the answer plus `x-vsr-selected-recipe` and `x-vsr-selected-model`; HTTP success
alone does not verify the output or route. See [routing
receipts](../troubleshooting/vsr-headers) and [CLI previews and probes](../api/cli).

## Set budgets before running a task

`GET /v1/models` discovers virtual names and routing metadata; it does not
advertise their context windows or output limits. Configure these in the
harness. To keep every candidate usable, take the smallest context window and
output limit across the recipe's selectable models, including the configured
default, and enable only their shared capabilities.

For heterogeneous pools, recipe-wide `candidate_requirements` can check output
budgets and declared tool, reasoning, and structured-output support before
scoring. This does not configure the harness's own context management. Follow
[Virtual Models: limits for agent clients](../tutorials/global/entrypoints-and-recipes#limits-for-agent-clients)
for the example and exact eligibility behavior.

## Preserve tool and conversation continuity

Run a real tool loop: model requests a tool → harness executes it → next call
carries the matching result. For streaming, check complete tool arguments, the
terminal event, and final answer. Protocol compatibility alone does not verify
the model and harness together.

When Router Learning protection is configured, it needs explicit identities.
The default `scope: conversation` requires both headers on related calls:

```http
x-session-id: harness-demo-session
x-conversation-id: harness-demo-conversation
```

These are example values. Use distinct, stable identifiers for your actual
sessions and conversations, supplied by the harness or a trusted gateway.
`scope: session` requires only the configured session header. A missing required
identity leaves the request routable without retaining a model through
protection. Responses history, including `previous_response_id`, does not
replace these headers. See [Session identification](../api/session-identification)
for identity priority and privacy boundaries.

Check the selected model on tool continuations and user follow-ups. The harness
retains tool permissions, task state, and the outer loop; recipes do not configure
delegated agents.

## Program the policy behind the entrypoint

[Connect models and publish a recipe](../tutorials/global/models-entrypoints-serving)
behind the same entrypoint. Signals, decisions, and algorithms define which
models are eligible and how they run.

The [Agent Routing recipe](https://github.com/vllm-project/semantic-router/tree/main/config/recipes/agent)
is a starting point for local, specialist, and frontier model lanes. Review its
model bindings, classifier requirements, and replay retention before using it;
its checked-in configuration records request and response data. Use the
[recipe evaluation workflow](../benchmarking/agent-evaluation-loop) to test
representative tasks before adopting a policy change.
