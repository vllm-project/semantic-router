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

## Claude Code and Codex CLI

Claude Code on a Claude subscription runs Anthropic's models, and the
[vSR plugin](https://github.com/vllm-project/semantic-router/tree/main/src/agent-harness/claude-code)
adds guard, shield, redact and check to it without a gateway. Route Claude Code
or Codex CLI through the Router when your own models or provider keys should
answer: the Router then chooses the model for every call, the main
conversation included, while vSR keeps judging tool output and shell commands
inside Claude Code. Codex reaches the Router the same way; vSR's checks are not
available for Codex yet.

The two harnesses need different Router-wide settings: the losses the Router
accepts, the headers that identify a session and a subagent, and defaults for
what the harness leaves out. Start with one Router for each; both can use the
same backends and decision model. This was checked with Claude Code 2.1.322
with vSR 0.1.10, Claude Code 2.1.317 without it, and Codex CLI 0.166.0, against
`anthropic` provider models served by vLLM 0.32.0.

### Connect with `vllm-sr connect`

```bash
export VLLM_SR_API_KEY=<the listener's API key>   # leave unset when it has none
vllm-sr connect claude-code --config claude-code.yaml
vllm-sr connect codex --config codex.yaml
codex --profile vllm-sr
```

Here each harness has its own Router, served from `claude-code.yaml` and
`codex.yaml`. The command reaches the gateway at the first listener of the
configuration it is given (`config.yaml` by default); `--base-url` names
another origin.

The command shows the changes and asks before writing them:

- Claude Code: the `env` block of `~/.claude/settings.json` (or
  `$CLAUDE_CONFIG_DIR/settings.json`; `--scope project` writes
  `.claude/settings.local.json`). The other settings stay as they are. The
  file holds the key when one is set, so only you can read it afterwards. In a
  git repository, `--scope project` adds the file and the command's undo record
  to `.git/info/exclude` when git would otherwise track them, and refuses to
  write a key into either one once git already tracks it. `ANTHROPIC_API_KEY`
  is set empty: otherwise Claude Code also sends a key your shell exports, as
  `x-api-key` beside the gateway's token.
- Codex: the `~/.codex/vllm-sr.config.toml` profile, which holds a `vllm-sr`
  provider, selects it and turns off hosted web search, which the Router does
  not serve. `~/.codex/config.toml` is not touched.

Neither harness knows `vllm-sr/auto`, so the command gives them its context
window: the smallest `context_window_size` of the models the entrypoint can
reach in the configuration, which are the models its recipe's decisions name
and `providers.defaults.model` (a catalog-backed model without one takes its
card's limit). Image and System One models are left out. It says where the
number came from and writes it as `CLAUDE_CODE_MAX_CONTEXT_TOKENS` for Claude
Code and `model_context_window` for Codex. When a model states no window, the
command names it and sets none: Claude Code then assumes a window of its own
(1,000,000 tokens in 2.1.323) and says so at start, and Codex falls back to
default model metadata. `--context-window` sets the window either way.

It then reports three checks, each on its own line:

- `Router configuration`: what the configuration given with `--config` lacks
  for this harness, from the settings below, without changing it.
- `Basic reply`: a plain request in the harness's protocol gets an answer, and
  the line names the model that gave it.
- `Streamed tool loop`: a streamed tool call, then the answer once its result
  goes back, as the harness sends them. A stream that ends in an error fails
  it, and the check runs only after the basic reply passed.

On a failure the written changes stay, so fix the gateway and run the command
again, or take them back with `--undo`. With the vSR plugin installed it warns
while the plugin's router app is on, and when the listener also serves System
One it prints the setting that points vSR at it. `--dry-run` prints the plan
and writes nothing.

After writing, the command names any settings that win over what it wrote,
and how to stop them. For Claude Code that is a `.claude/settings.json` or
`.claude/settings.local.json` in the current directory that sets one of its
`env` keys: Claude Code reads project settings only in the directory it starts
in, and they win over the user's. Variables your shell exports do not win over
the settings' `env`. For Codex it is a `.codex/config.toml` from the
repository's top down to the current directory that sets `model`: once Codex
trusts the project, it asks for that model instead of the profile's.
`OPENAI_BASE_URL` does not change the `vllm-sr` provider. These were measured
with Claude Code 2.1.323 and Codex CLI 0.166.0.

`--status-line` also sets Claude Code's status line to show which models
answered this session, the last first, such as `vllm-sr ⇢ qwen36 (4) ·
qwen38-27b (2)`. Claude Code asks for `vllm-sr/auto`, so its own model display
names only the entrypoint. The line reads the session's transcript and its
subagents' transcripts, where each reply keeps the model that wrote it, so it
needs no Router access. The script
sits beside the settings as `vllm-sr-status-line.py`, and `--undo` puts back
the status line you had before.
`--undo` puts back what each key, or the Codex profile, held before the first
run, leaves alone what changed since, and keeps everything else in the files.

### Try it without writing anything

```bash
vllm-sr connect claude-code --config claude-code.yaml --launch
vllm-sr connect codex --config codex.yaml --launch -- exec "fix the failing test"
```

`--launch` runs the same checks and the same probe, then starts the harness
through the gateway for that one session and exits with its status. Nothing is
written: Claude Code gets the settings with `--settings` from a temporary file
only you can read, which holds the key and is removed when Claude Code exits,
and Codex gets them as `-c` overrides and reads the key from `VLLM_SR_API_KEY`,
so it never appears on the command line. Both win over the user's and the
project's settings, so the overrides above do not apply. Arguments after `--`
go to the harness, and a gateway that fails the probe means the harness is not
started.

### Router settings for Claude Code

```yaml
global:
  router:
    allow_lossy_translation: [tools.defer_loading]
    learning:
      enabled: true
      protection:
        enabled: true
        scope: session
        identity:
          headers:
            session: x-claude-code-session-id
            agent: x-claude-code-agent-id
```

Claude Code's tool search marks tools for loading on demand. Without
`tools.defer_loading` the Router refuses the first calls of every session, and
Claude Code goes on without tool search. Toward an `openai` backend, also list
`tool_result.is_error`, `messages[].role` and `content.file`, as
[protocol compatibility](protocol-compatibility#validation-and-failure-behavior)
describes. An `anthropic` backend receives those as sent.

### Router settings for Codex CLI

```yaml
recipes:
  - name: default
    decisions:
      - name: agents
        # rules and modelRefs as in your recipe
        plugins:
          - type: request_params
            configuration:
              default_max_tokens: 32000
global:
  router:
    allow_lossy_translation: [reasoning.signature]
    learning:
      enabled: true
      protection:
        enabled: true
        scope: session
        identity:
          headers:
            session: session-id
            agent: thread-id
  services:
    response_api:
      enabled: true
      store_backend: memory
```

Codex sends no output limit, so a decision that sends it to an `anthropic`
backend sets one. Thinking from an `anthropic` backend carries a signature that
a Responses client cannot receive: without `reasoning.signature` the stream
ends in a `lossy_translation` error where the thinking block closes, and Codex
reconnects until it gives up. The Responses endpoint needs the Router's
Responses service and store. See [Codex CLI](protocol-compatibility#codex-cli)
for the other fields Codex sends.

### With the vSR plugin

One listener can serve both systems: the routed model calls for Claude Code
and, at `/v1/decisions`, the System One questions vSR asks. Publish the decision
model on the listener and point vSR at it, so the plugin and the gateway share
one address, one API key and one decision model. A listener that names its
`apis` serves only those, so list the model APIs the harnesses use next to
`systemone`:

```yaml
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
    apis:
      messages: {}               # Claude Code
      responses: {}              # Codex CLI
      chat_completions: {}
      systemone:
        models: [vela3-lite]     # the public names vSR may ask for
global:
  model_catalog:
    deployments:
      primary:
        provider: model_runtime
        public_name: vela3-lite
        artifact: vllm-sr/vela3-lite
```

vSR's apps are measured on vela3-lite. Models of the runtime's `vela3` family,
such as vela3-fast, answer no `score` question, which vSR's router app asks,
and score risky commands far below the line guard holds at, so publish
vela3-lite for vSR.

```bash
echo '{"runtime": "remote", "runtimeModel": "vela3-lite",
  "runtimeUrl": "http://localhost:8899/v1/decisions"}' \
  | claude plugin configure vsr@vllm-sr --values-stdin
```

System One checks the listener's API keys too. On a listener with keys, add
`"runtimeToken"` with the key, which `vllm-sr connect` fills in from
`$VLLM_SR_API_KEY` in the command it prints.

A Router rule about what the agent reads, such as a prompt injection in a file
or a web page, should read the tool results and leave the user turn out
(`input: {tool_results: true, request: false}`). Claude Code's user turn carries
its own instructions and reminders, and a rule that reads it also judges those.

Turn the plugin's router app off (`/vsr router off`) so the plugin and the
Router do not both choose a subagent's model. Its escalate app still asks Claude
to re-check an answer you push back on, but no longer moves the next subagent
up a model, since that step belongs to the router app. The other apps work as
before. When the plugin's decision runtime does not answer, its tally says
`model offline` and its checks step aside while Claude goes on, except guard's
fixed rules, which still hold, and an app set to wait until done, which withholds
the output or holds the command instead. A `SKIPPED` row says what one check
left unchecked. Both are explained in the
[plugin README](https://github.com/vllm-project/semantic-router/tree/main/src/agent-harness/claude-code#what-you-see).

### Settings by hand

The command writes these. Claude Code reads them from its settings:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8899",
    "ANTHROPIC_AUTH_TOKEN": "<the listener's API key, or any value without one>",
    "ANTHROPIC_API_KEY": "",
    "ANTHROPIC_MODEL": "vllm-sr/auto",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "vllm-sr/auto",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "vllm-sr/auto",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "vllm-sr/auto",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32000"
  }
}
```

Codex reads the profile from `~/.codex/vllm-sr.config.toml`, which
`codex --profile vllm-sr` layers over `~/.codex/config.toml`, and the provider
can live in the same file:

```toml
model = "vllm-sr/auto"
model_provider = "vllm-sr"
web_search = "disabled"

[model_providers.vllm-sr]
name = "vLLM Semantic Router"
base_url = "http://localhost:8899/v1"
wire_api = "responses"
env_key = "VLLM_SR_API_KEY"
```

Leave `env_key` out for a listener without keys: Codex refuses to start while
the variable it names is unset, and `vllm-sr connect` writes it only when
`VLLM_SR_API_KEY` is set.

Codex does not know `vllm-sr/auto`, so it warns that it uses fallback model
metadata. `vllm-sr connect` sets `model_context_window` in the profile from the
configuration, or from `--context-window`, so its budget follows your
backends, though the warning stays.

The Codex TUI sends a `namespace` tool on every turn. A Router that refuses
namespace tools answers its first turn with `unsupported_tool`; on such a
Router, run `codex exec` with `[features] multi_agent = false` instead.

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

When Router Learning protection is configured, it needs an explicit session
identity. Send both headers on related calls:

```http
x-session-id: harness-demo-session
x-conversation-id: harness-demo-conversation
```

These are example values:

- **Use real identities.** Send distinct, stable identifiers for your sessions
  and conversations, from the harness or a trusted gateway.
- **With only the session header**, `scope: session` holds the whole session,
  and the default `scope: conversation` holds each conversation, named by how
  it starts.
- **Without the session header** the request stays routable, but no model is
  kept through protection.
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
