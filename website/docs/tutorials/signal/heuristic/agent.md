# Agent Signal

## Overview

`agent` matches the agent a request belongs to: the main agent or a subagent,
the harness that sent it, the type the harness gives the subagent, and the
request's class, such as a compaction. It reads the identity headers Claude
Code, Codex and OpenCode send, with no model inference. Define these rules
under `recipes[].signals.agent`.

## Key Advantages

- Gives each kind of subagent its own model and reasoning effort, for example
  exploration on a small model at low effort and reviews on a strong model at
  high effort, while the main agent keeps the recipe's default.
- Sends a harness's housekeeping, such as compactions and session titles, to
  a small model.
- Reads the same agent identity as [protection](../../learning/protection#agents-that-delegate-to-subagents),
  so a subagent stays on the model its first request got for its whole life.
- Combines with any other signal, so a decision can block a subagent that
  sends personal data or route a reviewer that sees dangerous content.

## What Problem Does It Solve?

A coding harness sends its main agent and every subagent through the same
model name. Without a signal for the agent, a recipe can only route each
request by its text, so an exploration subagent that greps files gets the
model and effort of the main agent, and a reviewer gets whatever its prompt
happens to match.

## When to Use

Use `agent` when:

- subagents of different kinds need different models or efforts;
- a policy applies only to subagents, or only to the main agent;
- Claude Code, Codex and OpenCode traffic share a recipe but need different
  routes.

## Configuration

```yaml
recipes:
  - name: default
    signals:
      agent:
        - name: explore
          description: A Claude Code Explore subagent or a Codex spawn_agent child.
          role: subagent
          types: [Explore, collab_spawn]
        - name: reviewer
          description: A Codex review child.
          harness: codex
          types: [review]
        - name: any_subagent
          role: subagent
        - name: housekeeping
          description: Claude Code and OpenCode compactions and side requests such as session titles.
          request_classes: [compaction, auxiliary]
```

| Field | Meaning |
| --- | --- |
| `role` | `main` or `subagent`. |
| `harness` | `claude_code`, `codex`, `opencode`, `deepseek_harness`, `cursor` or `openclaw`. |
| `types` | Subagent types, any of which matches. The comparison ignores case. A rule with `types` matches subagents only, so `role: main` is refused with it. |
| `request_classes` | Request classes, any of which matches: `main`, `subagent`, `workflow`, `compaction` or `auxiliary`. Claude Code sends the class, and OpenCode, Codex and OpenClaw calls get one as the table below shows; DeepSeek Harness and Cursor calls carry none, so `harness: deepseek_harness` and `harness: cursor` are refused with it. |

A rule matches when every field it sets matches, and must set at least one.
Rule names are unique within a recipe.

### What each harness sends

The role is the agent identity protection reads: a request is a subagent's
when protection reads an agent id for it, and the main agent's when it names
a session and no agent. With `identity.headers.agent` configured, that header
decides the role.

| Harness | Main agent | Subagent | Type | Request class |
| --- | --- | --- | --- | --- |
| Claude Code | `x-claude-code-session-id` and no agent id | `x-claude-code-agent-id` | `x-claude-code-agent-type`: a built-in agent's name, such as `Explore`, `Plan` or `general-purpose`, or `custom` for every custom agent, `teammate` for an agent team member and `fork` for a fork | `x-claude-code-request-class`: `main`, `subagent`, `workflow`, `compaction` or `auxiliary`, on every request |
| Codex CLI | `thread-id` equal to `session-id` | `thread-id` not equal to `session-id` | `x-openai-subagent`: `collab_spawn` for a `spawn_agent` child, whatever its role, and `review` for a review child | none sent: its title thread (turn metadata `thread_title`) is `auxiliary`, a call without tools whose last user message opens with its checkpoint prompt, `You are performing a CONTEXT CHECKPOINT COMPACTION.`, is `compaction`, and every other call is `main` or `subagent` |
| OpenCode | `x-session-id` and no `x-parent-session-id` | `x-parent-session-id`: the session that spawned the subagent, whose own session is in `x-session-id` | none | none sent: a call without tools that opens with OpenCode's built-in title instructions is `auxiliary`, one that opens with its summary instructions is `compaction`, and every other call is `main` or `subagent` |

Claude Code sends the type and the class only when
`CLAUDE_CODE_GATEWAY_HINT_HEADERS=1` is set in its environment, as
[Anthropic's gateway guide](https://code.claude.com/docs/en/llm-gateway-protocol#gateway-hint-headers)
documents; `vllm-sr connect claude-code` sets it in the `env` block of the
settings it writes. Without it, a rule with `role` or `harness` still matches,
and a rule with `types` or `request_classes` does not. The type comes only with
a subagent's own turns: a subagent's compaction or side request keeps its
agent id, so its role, and carries its class but no type. These headers were
checked with Claude Code 2.1.325 and Codex CLI 0.166.0.

OpenCode is known by its User-Agent, `opencode/<version>`. A subagent runs in a
child session of its own, so the Router files its calls under the session that
spawned it, with the child as the agent, unless the request pins a session in
`x-vsr-session-id`; its subagents cannot start subagents of their own by
default, so a run has one level. OpenCode marks no request class, so its two
side requests are known by the instructions they open with: `You are a title
generator.` for a session title and `You are an anchored context summarization
assistant` (OpenCode 1.19: `You are a context summarization agent.`) for a
summary of the session. A project that replaces those prompts
sends calls that read as `main`. This was checked with OpenCode 1.18.10.

DeepSeek Harness is known by its User-Agent, `deepseek-harness/<version>`. It
names no session, subagent or request class, so `request_classes` cannot go
with `harness: deepseek_harness`, and its calls take the `main` role only when
the request pins a session in `x-vsr-session-id`.

OpenClaw marks no request class either: a call without tools whose system
prompt opens with its summary instructions, `You are a context summarization
assistant.`, is `compaction`, and every other call is `main` or `subagent`.

Cursor, with a custom OpenAI base URL, sends its calls from Cursor's servers
with `User-Agent: Cursor/<version>`; an agent that only contains `Cursor/`,
such as the browser built into the Cursor desktop app, is not Cursor. OpenClaw
sends the generic agent of the OpenAI SDK it calls through, so it is known by
the comment its system prompt opens with, `<!-- openclaw:`. Its compaction
carries no such comment and is known as a call without tools whose system
prompt opens with its summary instructions, sent with `User-Agent: OpenAI/JS`
and a `prompt_cache_key`. Neither sends a session header or names a
subagent, Cursor names no request class, so `request_classes`
cannot go with `harness: cursor`, and their calls take the `main` role only
when the request pins a session in `x-vsr-session-id`. This was checked
with OpenClaw 2026.11.5 and with Cursor's recorded calls.

## Decision Usage

Each decision names its model and effort in `modelRefs`, as any decision does;
see [Model reasoning](../../../installation/model-reasoning#select-reasoning-in-a-decision).
The main agent matches no agent rule here and keeps the default decision,
except for its compactions and side requests, which the `housekeeping`
decision sends to the small model, ahead of the subagent routes so that a
subagent's compaction goes there too.

```yaml
recipes:
  - name: default
    decisions:
      - name: exploration
        priority: 100
        rules:
          type: agent
          name: explore
        modelRefs:
          - model: small-model
            use_reasoning: true
            reasoning_effort: low
      - name: review
        priority: 100
        rules:
          type: agent
          name: reviewer
        modelRefs:
          - model: strong-model
            use_reasoning: true
            reasoning_effort: high
      - name: housekeeping
        priority: 150
        rules:
          type: agent
          name: housekeeping
        modelRefs:
          - model: small-model
            use_reasoning: true
            reasoning_effort: low
      - name: default
        priority: 1
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: default-model
```

Protection holds each subagent on the model its first request got, under its
own key, so the subagent keeps it for its whole life. The agent headers come
with every request of the subagent, so its requests keep matching the same
decision; a decision of higher priority that matches one of them, such as a
personal-data route, moves that request. A compaction or side request is held
under a key of its class beside its agent's, so the `housekeeping` decision
never moves the model its agent is held on.

### Block a request

A decision whose `fast_response` sets `block: true` refuses the request with
its message; see [Fast Response](../../plugin/fast-response#block-a-request).
Here a subagent that sends personal data, and any request with dangerous
content, are refused:

```yaml
recipes:
  - name: default
    signals:
      pii:
        - name: personal_data
      safety:
        - name: dangerous
          labels: [Violence, Guns and Illegal Weapons, S1, S9]
          predicate: {gte: 0.7}
    decisions:
      - name: block_subagent_personal_data
        priority: 300
        rules:
          operator: AND
          conditions:
            - {type: agent, name: any_subagent}
            - {type: pii, name: personal_data}
        plugins:
          - type: fast_response
            configuration:
              message: Subagents may not send personal data.
              block: true
      - name: block_dangerous
        priority: 300
        rules:
          type: safety
          name: dangerous
        plugins:
          - type: fast_response
            configuration:
              message: This request was blocked by the safety policy.
              block: true
```

### Custom agents and roles

Claude Code reports every custom agent as `custom`, and Codex reports every
`spawn_agent` role as `collab_spawn`, so neither names one custom agent or
role. To give one its own model and effort, give it its own model name: publish
a name in an entrypoint's `model_names` and name it in the agent's definition,
the `model` field of a Claude Code agent's frontmatter. That entrypoint's
recipe chooses the model and effort.

```yaml
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
  - model_names: [vllm-sr/review]
    recipe: review
```

```markdown
---
name: code-reviewer
description: Reviews a change for mistakes.
model: vllm-sr/review
---
```

Claude Code sends that name as the subagent's model, so its requests resolve
to that entrypoint. In Codex, `spawn_agent` can start a child on another
published model name the same way, once Codex's model catalog lists it; see
[Protocol compatibility](../../../installation/protocol-compatibility#namespace-tools-multi-agent-and-mcp).

## Seeing Why

The response names the decision in `x-vsr-selected-decision`. The replay record
lists the rule under `signals.agent`, and `x-vsr-trace` lists `agent:<name>`
under `signals` for a caller allowed to see it.

## Dependencies and Limitations

The headers are sent by the client, so a caller can claim any role or type,
and any client can name itself OpenCode in its User-Agent. Use the signal to
choose models, not to grant access. The classify and routing
preview APIs receive no harness headers, so agent rules never match there.
A native System One recipe cannot read the signal.
See a complete example:
[`config/fragments/signal/agent/subagent-roles.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/agent/subagent-roles.yaml).
