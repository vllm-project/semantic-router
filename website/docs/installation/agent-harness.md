---
title: Connect an agent harness
description: Connect an agent harness to a stable model API with explicit routing policy, model limits, and session continuity.
---

# Connect an agent harness

An open, programmable **decision layer** for models and compute.

Connect the model calls from your agent harness to a public entrypoint, then
evolve the models and routing policy behind that name.

An agent harness manages the agent loop, tool execution, and task state. The
Router evaluates policy for each model call and can select one model or run a
configured, bounded multi-model workflow. Inference runtimes and serving
platforms execute those calls and manage their compute.

Start with the [Quickstart](/docs/installation) if you do not have a running
Router. [Install with an agent](agent) explains how an agent can install and
operate it; this guide connects the harness that will use its inference API.

## Choose the inference connection

Configure your harness's model provider using the settings below. Use the
setting names supported by your installed harness version.

| Setting | What to use |
| --- | --- |
| Inference address | The public inference listener or gateway. The local Quickstart uses `http://localhost:8899`; the management API and Dashboard have different addresses. |
| Client protocol | OpenAI Chat Completions, OpenAI Responses, or Anthropic Messages, matching the requests your harness sends. |
| Public model ID | `vllm-sr/auto` for the default policy, or a published entrypoint from `GET /v1/models`. |
| Authentication | The credentials required by your inference listener or gateway. Backend provider credentials are configured separately by the operator. |
| Model limits and capabilities | Context, output, tools, vision, and reasoning settings that the selected recipe can support. |

Some clients expect a base URL ending in `/v1`; others append `/v1` themselves.
Check the final request path rather than copying a URL between harnesses:

| Client API | Final request path | Requirement |
| --- | --- | --- |
| Chat Completions | `POST /v1/chat/completions` | Send the Chat Completions request shape. |
| Responses | `POST /v1/responses` | The Router's `global.services.response_api` and its store must be available. |
| Messages | `POST /v1/messages` | Send the Messages request shape and an appropriate `anthropic-version` header. |

The client and backend may use different protocols. Compatibility still depends
on the features in the request: for example, free-form custom tools cannot be
translated to a Messages backend. Review the [Protocol Compatibility
Matrix](protocol-compatibility), including its Codex CLI limitations, before
enabling harness features. Keep credentials in the harness's supported secret
store or environment bindings.

## Verify the public model

After configuring at least one backend in the local stack, list its public
models:

```bash
curl -sS http://localhost:8899/v1/models
```

Use a virtual entrypoint when the Router should apply routing policy. A concrete
provider model name selects that model directly and bypasses recipe routing.
See [Entrypoints](../tutorials/global/entrypoints) for the resolution rules.

Send a minimal request and include response headers:

```bash
curl -sS -i http://localhost:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

These commands use the local Quickstart listener. For another deployment, use
its inference address and required authentication. Check the final assistant
output and the `x-vsr-selected-recipe` and `x-vsr-selected-model` routing
receipts; a successful HTTP status alone does not establish a useful answer or
the intended route. [VSR routing headers](../troubleshooting/vsr-headers)
explains the receipts and [the CLI reference](../api/cli) covers route previews
and end-to-end probes.

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

Verify a real tool loop in the harness: a model requests a tool, the harness
executes it, and the next model call carries the matching tool result. With
streaming enabled, also verify that the harness receives the complete tool
arguments, terminal event, and final answer. Protocol translation preserves the
supported wire contract; model capability and harness behavior still need
end-to-end verification.

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

Inspect the selected model on tool continuations and user follow-ups when
verifying protection. Keep tool execution permissions and task state in the
harness; a routing recipe does not configure delegated agents or own the outer
agent loop.

## Program the policy behind the entrypoint

Use [Models, Entrypoints, and Serving](../tutorials/global/models-entrypoints-serving)
to connect models and publish a recipe. Signals describe the request,
decisions enforce routing policy, and algorithms select or coordinate eligible
models. A stable entrypoint lets the harness keep its model ID while that policy
evolves.

The [Agent Routing recipe](https://github.com/vllm-project/semantic-router/tree/main/config/recipes/agent)
is a starting point for local, specialist, and frontier model lanes. Review its
model bindings, classifier requirements, and replay retention before using it;
its checked-in configuration records request and response data. Use the
[recipe evaluation workflow](../benchmarking/agent-evaluation-loop) to test
representative tasks before adopting a policy change.
