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

### Model names the harness picks itself

A harness can request models other than the one you configured. Claude Code,
for example, asks for its own Haiku model ID when it starts an Explore
subagent. The Router answers a model it does not publish with "the requested
model is not available", and that subagent fails. Point every model setting
the harness has at an entrypoint (for Claude Code, `ANTHROPIC_MODEL` and the
`ANTHROPIC_DEFAULT_*_MODEL` variables), or list the IDs it sends in the
entrypoint's `model_names`:

```yaml
entrypoints:
  - model_names: [vllm-sr/auto, claude-haiku-5-5]
    recipe: default
```

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

Claude Code sends `x-claude-code-session-id` on every Messages request and
does not send `x-session-id`. To hold its runs, set
`identity.headers.session: x-claude-code-session-id`. Its subagents send the
same session ID as the agent that started them, so under that setting they
are held on the model their parent was given.

### Agents that send tool output as messages

Some harnesses, such as mini-SWE-agent, do not use the protocol's tool calls: the model writes a command in its reply and the
harness sends the command's output back as the next user message. The Router
cannot tell that output from a new user turn, so it never sees a tool loop, and
signals read the latest output as if the user had written it.

Under `scope: conversation`, a decision that changes on such an output releases
the current model, and `min_turns_before_switch` does not hold it. Use
`scope: session` with a new session ID for every agent run instead: a changed
decision alone then keeps the run on its model, while idle timeout, a bypass
decision, candidate exclusion, or a qualified [rescue](../tutorials/learning/protection)
can still release it. To restart a task on a
different model, start a new run with a new session ID, or name the concrete
model, which bypasses recipe routing.

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

## Keep agent calls on a warm prompt cache

An agent resends its conversation on every call, so most of each prompt is a
prefix the backend has already processed. Whether that prefix is still cached
decides much of a call's latency and cost, and three things lose it.

- **A model change.** A prompt cache belongs to one model, so a switch in the
  middle of a run computes the whole context again on the new model. Keep each
  run on one model as described above. The
  [hybrid selector](../tutorials/algorithm/selection/hybrid) also favors the
  model that served the session's previous turn.
- **A replica change.** Each replica keeps its own prefix cache, and a model's
  `lb_policy` (`round_robin` or `least_request`, see
  [model configuration](model-configuration#tune-timeouts-retries-and-endpoint-health))
  does not keep a session on one replica. When several replicas serve agent
  sessions, route the model to a pool whose scheduler places requests by prefix
  cache and load, such as [llm-d](k8s/llm-d), instead of listing the replicas
  as `backend_refs`.
- **Eviction.** When the sessions in flight need more KV cache than the GPU
  holds, a returning session finds its prefix evicted. A CPU offload tier on the
  backend, such as LMCache for vLLM, returns those prefixes instead of computing
  them again.

The harness can also start the prompt over itself. When it compacts its history,
replacing earlier turns with a summary to stay within the context window, the
backend computes most of the new prompt again on the same model and replica.
The Router neither causes nor prevents this, so a drop in prefix reuse right
after a compaction is not a routing or placement fault.

Leave cross-model KV transfer (`global.integrations.kv_transfer`) off for agent
entrypoints. The receiving model must serve with prefix caching disabled, and
the [connector](https://github.com/vllm-project/semantic-router/tree/main/src/kv_connector)
rejects scaled RoPE, so neither model can serve a context extended past its
native length.
