---
title: Connect an agent harness
description: Connect an agent harness to a stable model API with explicit routing policy, model limits, and session continuity.
---

# Connect an agent harness

Point your harness at a Router entrypoint such as `vllm-sr/auto`. The harness
owns the agent loop, tools, and task state; the Router selects models or runs a
bounded multi-model workflow under your policy. Inference backends execute the
model calls.

[Install the Router](installation.md) first, or
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
receipts](../api/headers) and [CLI previews and probes](../api/cli).

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

A listener's `timeout` (300 seconds by default) bounds a whole call, the
streamed response included, and arriving output does not extend it. A call
that reaches the limit is cut off: the stream is reset and the rest of the
answer never arrives.

:::warning Long reasoning calls hit the listener timeout
For agent models, set the model's `total_timeout` to `0s` and keep
`idle_timeout` to bound a stalled stream; see
[model configuration](model-configuration#tune-timeouts-retries-and-endpoint-health).
In live Terminal-Bench 2.1 runs of Claude Code 2.1.304 at `xhigh` effort on
Qwen3.8-27B-FP8 (vLLM 0.32, one MI300X), 2.5% to 4% of calls took more than
300 seconds, and the longest about 17 minutes.
:::

## Preserve tool and conversation continuity

Run a real tool loop: model requests a tool → harness executes it → next call
carries the matching result. For streaming, check complete tool arguments, the
terminal event, and final answer. Protocol compatibility alone does not verify
the model and harness together.

If every request fails with `tool_text_limit`, the harness sends a tool name or
description over the Router's limit. Claude Code's Artifact tool is over the
16 KiB description default. Raise
`global.router.protocol_limits.tool_description_bytes`; see
[Validation and failure behavior](./protocol-compatibility#validation-and-failure-behavior).

When Router Learning protection is configured, it needs explicit identities.
The default `scope: conversation` requires both headers on related calls:

```http
x-session-id: harness-demo-session
x-conversation-id: harness-demo-conversation
```

These are example values:

- **Use real identities.** Send distinct, stable identifiers for your sessions
  and conversations, from the harness or a trusted gateway.
- **`scope: session`** requires only the configured session header.
- **A missing identity** leaves the request routable, but no model is kept
  through protection.
- **Responses history**, including `previous_response_id`, does not replace
  these headers.

See [Session identification](../api/session-identification) for identity
priority and privacy boundaries.

When the task changes phase, the harness can ask for a fresh routing decision
in the same session: it sends `x-vsr-protection-release: 1` on the first
request of the new phase, at a user turn rather than inside a tool loop. A
refused release returns its reason in `x-vsr-warnings`. See
[Explicit release](../tutorials/learning/protection#explicit-release).

:::warning Send the release header on one request only
Don't set it through Claude Code's `ANTHROPIC_CUSTOM_HEADERS`, which sends it
on every request.
:::

Check the selected model on tool continuations and user follow-ups. The harness
retains tool permissions, task state, and the outer loop; recipes do not configure
delegated agents.

Claude Code sends `x-claude-code-session-id` instead of `x-session-id`, and
Codex CLI sends `session-id` and `thread-id`. See
[Agents that delegate to subagents](../tutorials/learning/protection#agents-that-delegate-to-subagents)
for the identity headers that hold their runs and their subagents.

### Agents that send tool output as messages

Some harnesses, such as mini-SWE-agent, don't use the protocol's tool calls.
The model writes a command in its reply, and the harness sends the command's
output back as the next user message. The Router can't tell that output from
a new user turn, so it never sees a tool loop, and signals read the output as
if the user had written it.

Under `scope: conversation`, a decision that changes on such an output releases
the current model, and `min_turns_before_switch` does not hold it. Use
`scope: session` with a new session ID for every agent run instead:

- **A changed decision alone** then keeps the run on its model.
- **Idle timeout, a bypass decision, candidate exclusion or a qualified
  [rescue](../tutorials/learning/protection)** can still release it.
- **To restart a task on another model,** start a new run with a new session
  ID, or name the concrete model, which bypasses recipe routing.

## Program the policy behind the entrypoint

[Connect models and publish a recipe](../tutorials/global/models-entrypoints-serving)
behind the same entrypoint. Signals, decisions, and algorithms define which
models are eligible and how they run.

The [Agent Routing recipe](https://github.com/vllm-project/semantic-router/tree/main/config/recipes/agent)
is a starting point for local, specialist, and frontier model lanes. Review its
model bindings, decision-model questions, and replay retention before using
it; its checked-in configuration records request and response data. Use the
[recipe evaluation workflow](../benchmarking/agent-evaluation-loop) to test
representative tasks before adopting a policy change.

Decide reasoning in the decision as well. For a model with a reasoning family,
such as a catalog model, the decision's `modelRefs` entry chooses whether the
backend thinks and at what effort; the harness's own request does not:

| Decision's `modelRefs` entry | What the Router sends |
| --- | --- |
| `use_reasoning` left out | Reasoning off |
| `use_reasoning: true` | The decision's `reasoning_effort`, or the family default when none is set |

This holds whatever effort the harness asks for. Claude Code, for example,
asks for adaptive thinking with its effort setting on every call. Set
`use_reasoning: true` and the effort you want on the decisions behind an agent
entrypoint. See [Model Reasoning](./model-reasoning#select-reasoning-in-a-decision).

## Keep agent calls on a warm prompt cache

An agent resends its conversation on every call, so most of each prompt is a
prefix the backend has already processed. Whether that prefix is still cached
decides much of a call's latency and cost, and three things lose it.

- **A model change.** A prompt cache belongs to one model, so a switch in the
  middle of a run computes the whole context again on the new model. Keep each
  run on one model as described above. The
  [multi_factor selector](../tutorials/algorithm/selection/multi-factor#cache-affinity)
  also favors the model that served the session's previous turn.
- **A replica change.** Each replica keeps its own prefix cache, and a model's
  `lb_policy` (`round_robin` or `least_request`, see
  [model configuration](model-configuration#tune-timeouts-retries-and-endpoint-health))
  does not keep a session on one replica. When several replicas serve agent
  sessions, route the model to a pool whose scheduler places requests by prefix
  cache and load, for example [llm-d](k8s/llm-d) (an experimental integration),
  instead of listing the replicas as `backend_refs`.
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
