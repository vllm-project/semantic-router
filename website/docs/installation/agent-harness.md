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
inside Claude Code. Codex reaches the Router the same way, with its own vSR
plugin.

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
another origin. When that listener has `api_keys` and the variable is unset,
the command stops before writing anything, so a key written before is never
replaced.

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
  not serve. `~/.codex/config.toml` is not touched, unless `--app` asks for it
  ([the Codex app](#the-codex-app)).

It writes what it showed or nothing: a file that changes while the command
waits, or while it writes the files before it, is refused, and when one file
is refused or cannot be written, the files written before it get back their
text and mode, except one edited again since, which is named and left as it
is. Each file is replaced in one step and keeps its mode, except that Claude
Code's settings become readable by you only.

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
  for this harness and the models its entrypoint reaches, from the settings
  below, without changing it. For every harness that includes
  [the timeout and fallback an agent call needs](#keep-long-and-failed-calls-in-the-turn).
- `Basic reply`: a plain request in the harness's protocol gets an answer, and
  the line names the model that gave it.
- `Streamed tool loop`: a streamed tool call, then the answer once its result
  goes back, as the harness sends them. A stream that ends in an error fails
  it, and the check runs only after the basic reply passed.

On a failure the line says what failed in words: a key the gateway refused, no
connection at that address, or no answer in time. Only the listener's own
refusal (`invalid_api_key`) is read as a refused key. Any other 401, such as
the Router's when it holds no credential for the selected backend or the
backend's own, is shown as its body says. The written changes stay, so
fix what failed and run the command again, or take them back with `--undo`.
With the vSR plugin installed it warns
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

### Check and status

```bash
vllm-sr connect codex --config codex.yaml --check
vllm-sr connect claude-code --config claude-code.yaml --dry-run --check
vllm-sr connect status
```

`--check` then runs the harness itself once through the gateway: `claude -p`
with the settings the command wrote, `codex exec --profile vllm-sr`, or
`pi --mode json --provider vllm-sr` ([Pi](#pi)), and with `--dry-run` the
settings it would write, given for that run only. The
harness works in a temporary directory, is asked to run `echo vsr-check` and
answer with its output, and is stopped after 120 seconds. This spends a few
short model calls on your login or key. The harness reads its own credentials
as it always does, and the lines shown hide the gateway key, the value of any
variable whose name says it holds a secret, and bearer tokens. Each step gets a
line, and a failed one fails the command:

- `Harness`: found on `PATH`, with its version.
- `Login`: the gateway answered, so the key or login the harness sent was
  accepted.
- `Tool round trip`: the command ran and the model answered with its output.
- `Approval review`, for Codex with a `review_model`: Codex runs with
  `--approve-for-me` and is asked to request approval for the command. The line
  says whether a request for the review entrypoint appeared in the Router's
  replay list, read from the management API on the gateway's host (at
  `global.services.management_api.port`, 8080 by default, plus
  `VLLM_SR_PORT_OFFSET`, with `VSR_MGMT_TOKEN`). When `enabled: false`
  turns replay off for the review recipe, or when the model runs the command
  without asking, it reads `not observed`, which does not fail the check.
- `Exit`: the harness finished in time, exited with status 0, and reported no
  failed turn or error result. A run that ends any other way fails the check,
  even after the steps above passed.

`vllm-sr connect status` reads the undo records beside Claude Code's user and
project settings, in the Codex home and in Pi's agent directory. For each
connection it shows the files written, the gateway, the entrypoint, the review and memory models (Codex), the context
window and the login, whether each file still holds what connect wrote
(`as written`, `changed` with the keys that differ, or `missing`; tables Codex
appends to the profile count as written, as for `--undo`), and whether the
gateway answers a basic reply now. That sends one short request with the key
the harness would send and waits up to 5 seconds. It sends nothing, and says
why, once Claude Code's settings name another gateway or model than connect
wrote, or the Codex profile, Codex's shared `config.toml` (`--app`) or Pi's
`models.json` no longer holds what connect wrote, since the key they send may
then be for another gateway. A Pi connection fails when Pi's `auth.json` holds
a key of its own for `vllm-sr`, which Pi sends in its place. Under
`overridden here` it names the project settings in the current directory that
win over what connect wrote, the ones connect warns about after writing: a
`.claude/settings.json` or `.claude/settings.local.json` that sets one of the
keys of a user scope connection, and a `.codex/config.toml` that sets `model`.
Run it in the directory you start the harness in. These do not fail the
command, since the gateway itself still answers. `--json` prints the same as
JSON, the settings under `overrides`, and the command exits with status 1 when
a gateway check fails.

### Try it without writing anything

```bash
vllm-sr connect claude-code --config claude-code.yaml --launch
vllm-sr connect codex --config codex.yaml --launch -- exec "fix the failing test"
```

`--launch` runs the same checks and the same probe, then starts the harness
through the gateway for that one session and exits with its status. No setting
is written: Claude Code gets the settings with `--settings` from a temporary file
only you can read, and Codex gets them as `-c` overrides. Both read the key from
`VLLM_SR_API_KEY` in their own environment (Claude Code through an
`apiKeyHelper` that prints it), so it never appears on the command line or in a
file. The temporary file is removed when the harness exits, also when its
terminal closes. Both win over the user's and the project's settings, so the
overrides above do not apply. Arguments after `--` go to the harness, and a
gateway that fails the probe means the harness is not started. Claude Code's
`--resume` hint on exit names the removed file, so resume a launched session
with `vllm-sr connect claude-code --launch -- --resume <session id>`.

### Launch in a worktree or another home

```bash
vllm-sr connect codex --config codex.yaml --launch --worktree new:fix-auth
vllm-sr connect codex --config codex.yaml --launch --worktree ../feature-x
vllm-sr connect claude-code --config claude-code.yaml --launch \
  --worktree new --home ~/.claude-work
```

`--worktree new:NAME` starts the harness in a new worktree. Once the gateway
passes the probe, the command runs `git worktree add -b vllm-sr/NAME
<checkout>-NAME HEAD` in the git checkout it is run from, so the worktree sits
beside that checkout on a branch of its own and starts from the committed
`HEAD`; uncommitted changes stay in the checkout. `new` alone names it after the
harness and the time, such as `vllm-sr/codex-20261116-153012`. A name takes
letters, digits, `.`, `_` and `-`, and a branch or directory that already
exists is refused before anything is made. Give a directory called `new` as
`./new`. `--dry-run` prints the `git worktree add` command and makes nothing.

`--worktree <path>` starts the harness in an existing git worktree instead, and
only checks that git sees a work tree there. Neither form switches or removes a
branch or a worktree. The harness keeps its own home, `$CLAUDE_CONFIG_DIR` or
`~/.claude` and `$CODEX_HOME` or `~/.codex`, and with it its own login.
`--home` gives it another existing home for that session, such as one signed in
to a second account. Nothing is copied into it, so the harness signs in there as
that home's account does.

Before the harness starts, the command prints the session's header:

```text
Session 0b6f8c2e-3d1a-4f6b-9a57-2c8e4d1f7a90
  harness   claude-code
  model     vllm-sr/auto
  gateway   http://localhost:8899
  worktree  /src/app-fix-auth (made by connect from 1a2b3c4d5e6f)
  branch    vllm-sr/fix-auth
  home      /home/me/.claude
  started   2026-11-16T15:30:12+09:00
  requests  x-vsr-session-id, x-claude-code-session-id: 0b6f8c2e-3d1a-4f6b-9a57-2c8e4d1f7a90
  replay    files its calls under this session
```

Every model call of a launched session carries its id in `x-vsr-session-id`:
Claude Code sends it through `ANTHROPIC_CUSTOM_HEADERS`, keeping the headers
your shell sets there, and Codex through `http_headers` in its provider
settings. Claude Code also starts its conversation with `--session-id` set to
the same id, so its own `x-claude-code-session-id` carries it too, and
`-- --resume <id>` resumes it later. The Router reads `x-vsr-session-id` before
the headers `global.router.headers.session_id_headers` names, but Router Replay
and protection file a call under the header that
`global.router.learning.protection.identity.headers.session` names when the
call carries it. With `--config`, the `replay` line says which applies: for
Codex on the `session-id` the [Codex settings](#router-settings-for-codex-cli)
name, Replay files the calls under the session id Codex makes itself, one for
each Codex session. A Claude Code launch that resumes or continues a
conversation keeps that conversation's own id. Otherwise
`GET /api/v1/observability/replays?session_id=<id>` returns every call of the
launch, its subagents included.

### List and clean up launched sessions

```bash
vllm-sr connect sessions
```

Each launch is recorded in `$XDG_STATE_HOME/vllm-sr/sessions` (by default
`~/.local/state/vllm-sr/sessions`), and `connect sessions` lists them, the
newest first, with the header above and what is left of each one now: whether
the worktree still exists, how many uncommitted changes it has, and for a
worktree connect made, how many commits on its branch no other branch and no
remote has. `--json` prints the same as JSON.

The command removes nothing. For a worktree connect made it prints the steps
that remove it, and only steps that lose no work:

```text
  state     no uncommitted changes, no unmerged commits
  cleanup   git -C /src/app worktree remove /src/app-fix-auth, then git -C /src/app branch -d vllm-sr/fix-auth
```

While the worktree has uncommitted changes it says to commit or discard them
first, and while the branch has unmerged commits it says to push or merge the
branch before deleting it; `git worktree remove` and `git branch -d` refuse
those cases too. `git worktree remove` does delete the files git ignores there,
such as build output. A launch with `--worktree <path>` or none is listed but
never offered for removal, since that launch did not make the worktree. The
record stays after the cleanup, so the list keeps the session id to look up in
Insights.

### Fetch the key with a command

```bash
vllm-sr connect claude-code --config claude-code.yaml --key-command "pass show vllm-sr/gateway"
vllm-sr connect codex --config codex.yaml --key-command "pass show vllm-sr/gateway"
```

Where exporting `VLLM_SR_API_KEY` into a shell is awkward, as for a service or
an IDE that starts the harness, `--key-command` names a shell command that
prints the listener's key. The command writes it as the harness's own key
helper in place of the key or a variable, and the harness runs it:

- Claude Code: `apiKeyHelper` in the settings, with `ANTHROPIC_AUTH_TOKEN` and
  `ANTHROPIC_API_KEY` set empty, since either would win over it. Claude Code
  runs it in the system shell, sends what it prints as both `Authorization`
  and `x-api-key`, and runs it again after five minutes
  (`CLAUDE_CODE_API_KEY_HELPER_TTL_MS`). Without `--key-command` the command
  takes any `apiKeyHelper` out of the settings it writes, since a helper would
  send its key in place of Claude Code's own login on a caller gateway, and
  `--undo` puts it back.
- Codex: a `[model_providers.vllm-sr.auth]` table in the profile, with
  `command = "/bin/sh"` and `args = ["-c", "<the command>"]`, in place of
  `env_key`, which Codex 0.166 does not take beside it. Codex sends what it
  prints as the bearer key, waits up to 5 seconds for it, and runs it again
  after five minutes and after a refused key. Windows has no `/bin/sh`, so for
  the Codex app on Windows use `--api-key-env`, or run its agent in WSL.
- Pi: `"apiKey": "!<the command>"` for the `vllm-sr` provider in
  `models.json`. Pi runs it in the system shell before each request, waits up
  to 10 seconds for it, and sends what it prints as the bearer key.

The command runs it only to probe the gateway, for `--check` and for
`connect status`, and never prints or writes what it prints. A command that
fails, prints nothing, or takes longer than the harness waits for it (5 seconds
for Codex, 10 for Pi) stops it before anything is written. `--launch` passes
the same setting for that session and runs the command for its probe as the
harness will, in the `--worktree` and with the `--home` given, so the key
probed is the key the harness sends. `--undo` removes it with the rest. A
gateway that takes the harness's own login has no key to fetch, so the option
is refused there. The vSR plugin takes a stored key or a variable, not a
command, so vSR is not set up with it.

### The Codex app

```bash
vllm-sr connect codex --config codex.yaml --app --key-command "pass show vllm-sr/gateway"
```

The Codex app reads `~/.codex/config.toml` (`$CODEX_HOME/config.toml`), which
also holds the app's own settings, and no profile: Codex 0.166 layers a
`<name>.config.toml` profile over it only when `--profile` names one, and
refuses a `profile` key in it. `--app` therefore writes the gateway there, as
the default, in place of the profile:

- the top-level `model`, `model_provider`, `model_context_window`,
  `model_catalog_json` and `web_search`, and the `[model_providers.vllm-sr]`
  table, with its `auth` table for `--key-command`. The catalog is
  `vllm-sr-app.models.json`, so a profile's catalog is never replaced. A key of these that the
  command has no value for, such as a `model_context_window` of yours, is taken out, since it would
  describe another model.
- every other line, comment and table stays as it is. A file it cannot edit
  that way, such as one where a key it sets spans several lines or one that is
  not valid TOML, is refused and left alone. Reading TOML needs Python 3.11 or
  later.
- the preview is a diff of the whole file with no context lines: the lines the
  command writes or takes out, at their line numbers, so `patch` applies it.
  It never prints a line of yours, which can hold a secret no setting name
  gives away, such as an MCP server's `env`. One it changes, such as a last line
  that gets its missing newline, reads `(a line that is not connect's)`.

This changes the default for the Codex CLI too: every `codex` started without
`--profile` uses the gateway, and the command says so before it writes and
after. It never reads or writes `~/.codex/auth.json`. `--undo` gives each key
and the provider table their value from before, leaves those changed since as
they are, and keeps what you or Codex added. `--check` runs `codex exec`
without `--profile`, and `connect status` lists the connection as
`codex (config.toml, its default)`. `--launch` writes nothing, so it does not
take `--app`.

Restart the app afterwards: Codex reads `model_catalog_json` only when it
starts. An app started from the desktop often does not see the variables your
shell exports, so give it `--key-command` rather than `VLLM_SR_API_KEY`.

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
```

Claude Code's tool search marks tools for loading on demand. Without
`tools.defer_loading` the Router refuses the first calls of every session, and
Claude Code goes on without tool search. Toward an `openai` backend, also list
`tool_result.is_error`, `messages[].role` and `content.file`, as
[protocol compatibility](protocol-compatibility#validation-and-failure-behavior)
describes. An `anthropic` backend receives those as sent.

Protection reads the session and subagent ids Claude Code sends,
`x-claude-code-session-id` and `x-claude-code-agent-id`, with no identity
headers configured, so each session and each subagent keeps its own model.
Insights then names each subagent in a session's routing history.

### Claude Code on its own Claude login

Claude Code signed in to a Claude subscription can use the gateway without an
API key: give the backends of its Claude models `auth: {credential: caller}`
([Security Hardening](security-hardening#send-the-clients-own-login-to-its-provider)),
and point Claude Code at the listener without a gateway token, so it sends its
own login:

```bash
env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN -u CLAUDE_CODE_OAUTH_TOKEN \
  ANTHROPIC_BASE_URL=http://localhost:8899 \
  ANTHROPIC_MODEL=vllm-sr/auto \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=vllm-sr/auto \
  ANTHROPIC_DEFAULT_SONNET_MODEL=vllm-sr/auto \
  ANTHROPIC_DEFAULT_OPUS_MODEL=vllm-sr/auto \
  CLAUDE_CODE_MAX_CONTEXT_TOKENS=1000000 \
  claude
```

`CLAUDE_CODE_MAX_CONTEXT_TOKENS` is the smallest context window among the
models the entrypoint reaches; without it Claude Code does not know
`vllm-sr/auto` and says so at start. `vllm-sr connect claude-code --config
<your config>` writes the same settings for such a configuration: it leaves
`ANTHROPIC_AUTH_TOKEN` empty, so neither a gateway key nor one your shell
exports replaces the login, and it does not probe the gateway, which answers
only on that login.

The Router then routes each call among Claude models on that login, and
Insights shows the session's calls with the model each decision picked. A
cost is the tokens at the API's list price, not what the plan charges. A model
without `pricing` takes its list price from the built-in catalog, through its
`catalog` card or through the `provider_model_id` its `anthropic` backend calls,
such as `claude-sonnet-5-5` or `claude-haiku-5-5`
([list prices](catalog-backed-models#list-prices)). A model the catalog does
not price has no cost. The login stays its owner's: serve this
to your own Claude Code, or to your organization's members, never to other
people as a service
([Security Hardening](security-hardening#send-the-clients-own-login-to-its-provider)).

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

Protection reads the `session-id` and `thread-id` Codex sends with no identity
headers configured, so the root thread and each subagent or review child keep
their own models, and Replay files their calls under the Codex session.

### Codex on its own ChatGPT login

Codex signed in with ChatGPT can use the gateway without an API key as well:
give the backends of its OpenAI models `provider: openai`, the endpoint
`https://chatgpt.com/backend-api/codex` and `auth: {credential: caller}`
([Security Hardening](security-hardening#send-the-clients-own-login-to-its-provider)),
and add `requires_openai_auth = true` to the gateway's provider in the Codex
profile, so Codex sends its own login with the request:

```toml
[model_providers.vllm-sr]
name = "vLLM Semantic Router"
base_url = "http://localhost:8899/v1"
wire_api = "responses"
requires_openai_auth = true
```

Keep `web_search = "disabled"` in the profile: the Router takes function tools
only. `vllm-sr connect codex --config <your config>` writes this profile for
such a configuration, with `requires_openai_auth` and without a key variable,
and does not probe the gateway, which answers only on Codex's own login. Each
`connect` command stops before writing anything for a gateway whose caller
backends name the other provider: the Router takes an `anthropic` login only on
Messages and an `openai` one only on Responses or Chat, so every call would be
refused. The plan's usage limits apply to each call, and a cost and the login
work as for Claude Code above, with `gpt-6.1-sol` on an `openai` backend for
example.

### Approve for me reviews

`vllm-sr connect codex` keeps approval reviews on the gateway. Without an
entrypoint's `review_model`, Codex's **Approve for me** uses the session model.
The generated catalog contains only that selected model, so Codex cannot pick
an unrelated bundled `codex-auto-review` model the gateway does not serve.

To choose a separate reviewer, publish a second entrypoint and set the main
entrypoint's `review_model` to one of its model names. This complete config
answers on `gpt-6-luna` and reviews with Codex's own `codex-auto-review`, both
on the caller's ChatGPT login:

```yaml
version: v0.5
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
    apis:
      responses: {}
providers:
  defaults:
    model: gpt-6-luna
  models:
    - name: gpt-6-luna
      provider_model_id: gpt-6-luna
      api_format: responses
      context_window_size: 272000
      backend_refs:
        - name: chatgpt
          endpoint: https://chatgpt.com/backend-api/codex
          provider: openai
          auth:
            credential: caller
    - name: codex-auto-review
      provider_model_id: codex-auto-review
      api_format: responses
      context_window_size: 272000
      backend_refs:
        - name: chatgpt
          endpoint: https://chatgpt.com/backend-api/codex
          provider: openai
          auth:
            credential: caller
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: main
    default: true
    review_model: vllm-sr/review
  - model_names: [vllm-sr/review]
    recipe: review
recipes:
  - name: main
    decisions:
      - name: answer
        priority: 1
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: gpt-6-luna}]
  - name: review
    decisions:
      - name: approve
        priority: 1
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: codex-auto-review}]
global:
  router:
    learning:
      enabled: true
      protection:
        enabled: true
        scope: session
  services:
    response_api:
      enabled: true
      store_backend: memory
```

Serve it, run `vllm-sr connect codex --config config.yaml`, then start
`codex --profile vllm-sr` (or use `connect codex --launch`). Connect prints where reviews go and keeps its catalog and provider settings in
separate profile files; `--undo` restores them.

The catalog lists `vllm-sr/auto` and hides `vllm-sr/review`, setting the main
model's `auto_review_model_override` to the review entrypoint. The review entry
retains the base model's reasoning levels, defaults to `low` effort, and gets
its own context window from the models its recipe can reach. Reviews use the
ordinary Responses API, including function tools and JSON schema output;
Responses Lite is not needed.

Any other model can be the reviewer: point the review recipe at that model's
provider binding instead. The target must be another entrypoint in the same
API namespace, and must have neither a `review_model` nor a `memory_model`
itself. A reviewer cannot refer back to the main entrypoint or form a chain.
With several recipes, keep `default: true` on the main entrypoint. Re-run
connect after changing the review binding to refresh the catalog.

### Codex memories

Codex's memories feature (off by default; `[features] memories = true` turns
it on) summarizes earlier interactive sessions in the background. It asks the
configured provider, the gateway here, for its own models: `gpt-5.6-luna` to
extract and `gpt-5.6-terra` to consolidate, which the gateway does not serve,
so no memory would be written. `vllm-sr connect codex` therefore sets
`[memories] extract_model` and `consolidation_model` in the profile, in
`config.toml` with `--app` (beside the `[memories]` settings you have, which
`--undo` restores), and on the command line with `--launch`. Without an
entrypoint's `memory_model` they name the connected entrypoint, and the Router
routes these calls like any other request to it.

To send memory calls to a model of their own, such as a small local one that
spends nothing from a plan's limits, publish a second entrypoint and set the
main entrypoint's `memory_model` to one of its model names. This complete
config answers on `gpt-6-luna` on the caller's ChatGPT login and writes
memories with a local model served by vLLM:

```yaml
version: v0.5
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
    apis:
      responses: {}
providers:
  defaults:
    model: gpt-6-luna
  models:
    - name: gpt-6-luna
      provider_model_id: gpt-6-luna
      api_format: responses
      context_window_size: 272000
      backend_refs:
        - name: chatgpt
          endpoint: https://chatgpt.com/backend-api/codex
          provider: openai
          auth:
            credential: caller
    - name: local-memory
      provider_model_id: Qwen/Qwen3-8B
      api_format: chat_completions
      context_window_size: 32768
      backend_refs:
        - name: local
          endpoint: http://localhost:8000/v1
          provider: vllm
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: main
    default: true
    memory_model: vllm-sr/memory
  - model_names: [vllm-sr/memory]
    recipe: memory
recipes:
  - name: main
    decisions:
      - name: answer
        priority: 1
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: gpt-6-luna}]
  - name: memory
    decisions:
      - name: remember
        priority: 1
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: local-memory}]
global:
  router:
    learning:
      enabled: true
      protection:
        enabled: true
        scope: session
  services:
    response_api:
      enabled: true
      store_backend: memory
```

Serve it, run `vllm-sr connect codex --config config.yaml`, then start
`codex --profile vllm-sr` (or use `connect codex --launch`). Connect prints
where memories go and sets both memory models to `vllm-sr/memory`. It also
adds `vllm-sr/memory` to the catalog as a hidden entry, so it stays out of the
model picker, with its own context window from the models its recipe can
reach. Codex cuts each session it extracts to fit that window.
`vllm-sr connect status` shows the memory model.

Extraction asks for JSON schema output, and consolidation runs as an agent
that edits the memory files with shell commands, so the local server must
accept function tools (for vLLM, `--enable-auto-tool-choice` with the model's
tool call parser). The target follows the same rules as a review target:
another entrypoint in the same API namespace with neither a `review_model` nor
a `memory_model` of its own. One entrypoint can take both reviews and
memories. Re-run connect after changing the memory binding.

Router Replay records the reviews by default, under the recipe that answers
them: the `review` recipe above, or the main one when reviews use the session
model. A `router_replay` plugin with `enabled: false` on that recipe leaves
them out.

Each review's record holds the planned action, the verdict with its risk and
rationale, the reviewed session, the review model, latency, tokens and cost.
A failed or interrupted review is recorded as failed or cancelled, never as a
deny. The command and the rationale are request and response text, so they are
kept only with `prompt_capture: raw` and body capture on. Open **Insights** and
switch to **Reviews** for the approval rate, reviews over time, cost by review
model and a table of reviews; a session's routing history marks the turns
whose actions were reviewed. See
[Agent approval reviews](../tutorials/plugin/router-replay#agent-approval-reviews)
for the fields and the API filters.

### Mixing a login with other models

A pool on the caller's login can sit beside models with a key of their own or
none, so hard turns, easy turns and reviews each go to a model of their own:
here hard work goes to `gpt-6.1-sol` and reviews to `codex-auto-review` on the
ChatGPT login, and easy turns to a local model, with `gpt-6-luna` for the ones
too long for it.

```yaml
version: v0.5
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
    apis:
      responses: {}
providers:
  defaults:
    model: gpt-6-luna
  models:
    - name: gpt-6.1-sol
      provider_model_id: gpt-6.1-sol
      api_format: responses
      context_window_size: 500000
      max_output_tokens: 128000
      backend_refs:
        - name: chatgpt
          endpoint: https://chatgpt.com/backend-api/codex
          provider: openai
          auth:
            credential: caller
    - name: gpt-6-luna
      provider_model_id: gpt-6-luna
      api_format: responses
      context_window_size: 272000
      max_output_tokens: 128000
      backend_refs:
        - name: chatgpt
          endpoint: https://chatgpt.com/backend-api/codex
          provider: openai
          auth:
            credential: caller
    - name: codex-auto-review
      provider_model_id: codex-auto-review
      api_format: responses
      context_window_size: 272000
      max_output_tokens: 128000
      backend_refs:
        - name: chatgpt
          endpoint: https://chatgpt.com/backend-api/codex
          provider: openai
          auth:
            credential: caller
    - name: local-coder
      provider_model_id: Qwen/Qwen3-Coder-30B-A3B-Instruct
      api_format: chat_completions
      context_window_size: 65536
      max_output_tokens: 16384
      backend_refs:
        - name: local
          endpoint: http://localhost:8000/v1
          provider: vllm
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: agent
    default: true
    review_model: vllm-sr/review
  - model_names: [vllm-sr/review]
    recipe: review
recipes:
  - name: agent
    candidate_requirements:
      context: known_limits
    signals:
      keywords:
        - name: hard_work
          description: Work that needs the strongest model.
          operator: OR
          method: regex
          keywords: ['(?i)\b(refactor|architecture|migrate|debug)\b']
    decisions:
      - name: hard
        priority: 200
        rules: {operator: AND, conditions: [{type: keyword, name: hard_work}]}
        modelRefs: [{model: gpt-6.1-sol}]
        plugins:
          - type: request_params
            configuration: {default_max_tokens: 32000}
      - name: easy
        priority: 100
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: local-coder}, {model: gpt-6-luna}]
        plugins:
          - type: request_params
            configuration: {default_max_tokens: 8192}
  - name: review
    decisions:
      - name: approve
        priority: 1
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: codex-auto-review}]
global:
  router:
    learning:
      enabled: true
      protection:
        enabled: true
        scope: session
  services:
    response_api:
      enabled: true
      store_backend: memory
```

- **Who pays.** The login reaches only the caller models. The local model
  receives neither the login nor the `chatgpt-account-id` beside it, and
  answers without one, so easy turns keep working when the plan's limit is
  reached. One model keeps one payer: its backends all take the login or none
  does. A decision may list models of both kinds, as `easy` does, and then
  either may answer.
- **What each decision may do.** The login rules bind only the decisions that
  can reach a caller model: there no `system_prompt` and no in-process calls.
  A decision whose models all take no login may use both, and a recipe-wide
  `response_cache` stays refused, since a request may name a caller model
  itself.
- **Long conversations.** With `candidate_requirements.context: known_limits`
  on the recipe, a request that no longer fits the local model's 65,536 tokens
  goes to `gpt-6-luna`, the larger model of the same decision. That needs each
  model's `context_window_size` and `max_output_tokens` and an output bound on
  the decision (`request_params.default_max_tokens`, since Codex sends none).
  `vllm-sr connect` then advertises 272,000 tokens, the smallest of each
  decision's largest window, instead of the local model's, so Codex does not
  compact at 65,536. Without `known_limits` a request too long for its model
  is refused, and connect advertises the smallest window of all.
- **What picks hard or easy.** A decision's rules read the last user message,
  so a task keeps its decision through the tool calls that follow it, and the
  session protection in `global.router.learning` keeps the session on its
  model. A System One rule
  reads more when its `input` asks for the history or tool results.

Claude Code on a Claude login mixes the same way: Claude models with
`provider: anthropic` and `auth: {credential: caller}` for hard turns and a
local model for easy ones, on a `messages` listener.

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
vllm-sr connect claude-code --config router.yaml
vllm-sr connect codex --config router.yaml --vsr-source /path/to/semantic-router
```

When the listener publishes a System One model, one command configures the
gateway (System Two) and vSR (System One); `--no-vsr` leaves vSR out. The
plugin installs from `--vsr-source`, this repository on GitHub by default, in
the agent's own CLI (`claude plugin` or `codex plugin`), and only when it is
missing. For Claude Code the command then sets the plugin's options through
`claude plugin configure`: a remote runtime, the gateway's `/v1/decisions`, the
first published System One model, and `router: false` so the gateway chooses the
models. When the API key variable is set it also passes `runtimeToken`, which
Claude Code keeps in its credential store, not in `settings.json`; the preview
names the variable and never shows the key. For Codex it writes
`runtimeUrl`, `runtimeModel` and `managementUrl` to `$CODEX_HOME/vsr.json`
(default `~/.codex/vsr.json`), keeping its other settings. `managementUrl` is
the Router's management API on the gateway's host, at
`global.services.management_api.port` (8080 by default) moved by the stack's
port offset.

Both harnesses preview everything with `--dry-run`, the plugin commands
included, which run only when writing. `--undo` sets back the plugin options the
command changed (an option that was never set returns to its default) but leaves
the plugin installed and the token in place. The gateway settings are written
even when vSR cannot be: without the agent's CLI on `PATH`, when the plugin's
options cannot be read, or when the plugin holds a token from before and no key
is given (Claude Code cannot clear a stored token, and the plugin would send it to
the new listener), the command says why and what to run. `--launch` writes
nothing, so it leaves vSR alone.

The listener can also set which of vSR's apps run, for every plugin pointed at
it, under `systemone.vsr`; the Dashboard shows each field as Default, On or Off
under Config › Global Config › vSR agent plugins, where Default leaves the field
out. Each field set there wins over the plugin's own option, so turning
`redact` off here turns it off in every session that uses this listener, from
its next prompt a minute or more later. A choice a person makes in Claude Code
with `/vsr` stays theirs until `/vsr follow`:

```yaml
      systemone:
        models: [vela3-lite]
        vsr: {redact: true, guard: true, router: false}
```

#### Which models served a Codex turn

Codex knows only the entrypoint it asked for, such as `vllm-sr/auto`. At the
end of each turn the Codex plugin asks Router Replay for the turn's records and
shows one line:

```text
↳ Hook · ▚▞ vSR turn · 4 calls: gpt-6-luna (3), qwen3-coder (1) · decisions: coding, simple · 2 reviews: 1 allowed, 1 denied by codex-auto-review
```

Codex sends its turn metadata with every model call, in the
`x-codex-turn-metadata` header and the request's `client_metadata`, and the
Router records the call's `thread_id`, `turn_id` and `parent_turn_id` from it.
An approval review and a subagent name the turn that started them as their
parent turn, so `GET /api/v1/observability/replays?turn_id=<id>` returns the
turn's calls, reviews and subagent calls, and the plugin asks for the turn id
its `Stop` hook receives. The line does not use the session. Replay files a
Codex call under the `session-id` Codex sends when protection reads it, as it
does by default under `scope: session`
([Session identification](../api/session-identification#choose-the-identity-you-need)),
and under a session the Router derives otherwise.

- **Router Replay** records the turn and its reviews by default; a recipe
  whose `router_replay` plugin sets `enabled: false` leaves them out. A
  principal without `replay.detail` permission still sees the models,
  decisions and verdicts.
- **The management API** must be reachable at `managementUrl`. With bearer
  auth, set `VSR_MANAGEMENT_TOKEN` to a token with `replay.read`.
- **No answer within 1.5 seconds**, or no record of the turn, ends the turn
  without a line. Set `managementUrl` to `""` in `vsr.json` to turn it off.

#### Rules of your own for guard and redact

`vsr.rules` binds rules of one recipe to guard and redact. A rule is an
ordinary signal of that recipe: guard takes any [keyword
rule](../tutorials/signal/heuristic/keyword) and a [System
One](../tutorials/signal/systemone) `noul` rule, or `choice` rule with
`labels`; redact takes a rule that finds text, an `OR` keyword rule of literal
or regex keywords without `fuzzy_match`, `count` or `density`, or a System One
`span` rule. An entrypoint must reach the recipe, since the Router prepares the
rules of a reachable recipe only.

```yaml
      systemone:
        models: [vela3-lite]
        vsr:
          rules:
            recipe: default
            guard:
              - {type: keyword, name: prod_database, enforce: required}
              - {type: systemone, name: deletes_data}
            redact:
              - {type: keyword, name: employee_id, enforce: required}
```

In the Dashboard, Config › Global Config › vSR agent plugins › Rules of your
own picks the recipe and lists its rules that fit each app, each Off, Advisory
or Required. Once an app binds a rule, the rules left Off fold behind a
"Show N more rules" button. Try it under each app sends a sample command or output through
the Dashboard, which adds the listener's key on the server, and says which
rules match or what they would hide, as the Router runs the saved rules.

`enforce: required` makes a rule fail closed for the plugin that applies it:
when the rule cannot be checked, the command waits for the person and the
output is withheld. `advisory`, the default, lets an unchecked rule step aside
as the plugins' own checks do. An app binds at most 16 rules, each name once.

The listing (`GET /v1/systemone/models`) hands each rule's name, description
and `required` to the plugins under `vsr.rules`, with a `revision` that changes
with any rule's definition, binding or enforcement. A plugin that gets an
answer for another revision, such as after a change in the Dashboard, reads the
listing again and asks once more in the same call, so a change applies to a
running session at once. A rule's pattern or
question never leaves the Router. To evaluate them, a client sends the texts
to the same listener, with the same key as any System One call:

```bash
curl -s http://localhost:8899/v1/systemone/rules \
  -H 'Authorization: Bearer <listener key>' \
  -d '{"app": "redact", "parts": [{"id": "out-1", "text": "Paid by E-104233"}]}'
```

```json
{"revision": "3f9c0a1b2d4e", "parts": [{"id": "out-1", "coverage": "complete",
  "matches": [], "spans": [{"start": 8, "end": 16, "rule": "employee_id"}], "unchecked": []}]}
```

Each part comes back with the guard rules it matches, or the redact spans in
code points, overlapping spans of different rules merged into one under the
rule that starts first, and the rules the Router could not check with a reason.
`coverage` is `complete`, `partial` or `failed`. A request holds at most 16
parts and 256 KiB of text; a listener evaluates 8 such requests at once and
answers `429` with `Retry-After` beyond that, `404 rules_not_published` for an
app it binds no rules to, and `401` without its key. Every call is logged with
the rule names, counts and latency, never the text.

The vSR plugins for Claude Code and Codex read these rules from the listing and
call this endpoint themselves. Guard sends the whole command beside its own
questions and holds it when a rule matches; redact sends the tool output as the
tool gave it and puts `[vsr:<rule name>]` where a rule found text. A required
rule that cannot be checked holds the command or withholds the output in both;
the rules last read from a listener are kept, so an outage does not drop one
once a plugin has read the listing. Before that first read a plugin knows no
rules, and its checks step aside when the listener does not answer.
Turning guard or redact off for a session turns their rules off too, so a
protection nobody may turn off belongs in a decision or in masking at the
gateway.

A Router rule about what the agent reads, such as a prompt injection in a file
or a web page, should read the tool results and leave the user turn out
(`input: {tool_results: true, request: false}`). Claude Code's user turn carries
its own instructions and reminders, and a rule that reads it also judges those.

Turn the plugin's router app off (`/vsr router off`) so the plugin and the
Router do not both choose a subagent's model, and with it the plugin's routing
of your own turns (`routerTurns`), which is for Claude Code talking to Anthropic
directly. Its escalate app still asks Claude
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

`CLAUDE_CODE_MAX_OUTPUT_TOKENS` is 32000, or the smallest `max_output_tokens`
of the models the entrypoint reaches when that is lower, since a model refuses
a request for more.

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

Codex does not know `vllm-sr/auto`, so without a catalog entry for it, it warns
that it uses fallback model metadata. `vllm-sr connect` writes one next to the
profile, `vllm-sr.models.json`, named by `model_catalog_json`. The file
replaces Codex's own catalog and holds only the selected gateway model and its
optional review and memory entrypoints. All adapt the installed Codex's `gpt-5.5` entry
(from `codex debug models --bundled`), with tool search and the `apply_patch`
custom tool turned off, since the Router does not serve the first and a
Messages backend refuses the second. The main entry uses Codex's fallback
reasoning settings, without the base model's upgrade prompt and Fast tier.
Codex then edits files through its shell. A name Codex already knows, such as
`--model gpt-5.5`, gets the same gateway adaptation. A Codex that bundles no
`gpt-5.5` entry gets no catalog and keeps the warning. Run `vllm-sr connect codex`
again after updating Codex, so the catalog follows its release.
`vllm-sr connect` also sets `model_context_window` and the main entry's window
from the configuration, or from `--context-window`, so the budget follows your
backends. See [Approve for me reviews](#approve-for-me-reviews) to select a
reviewer and [Codex memories](#codex-memories) to select a memory model.

Codex 0.166 also says it runs without its shared background server: a profile,
like `-c` overrides, starts a Codex process of its own. The session works as
usual.

The Codex TUI sends a `namespace` tool on every turn. A Router that refuses
namespace tools answers its first turn with `unsupported_tool`; on such a
Router, run `codex exec` with `[features] multi_agent = false` instead.

## Pi

```bash
export VLLM_SR_API_KEY=<the listener's API key>   # leave unset when it has none
vllm-sr connect pi --config pi.yaml
pi --provider vllm-sr --model vllm-sr/auto
```

[Pi](https://github.com/earendil-works/pi) reads custom providers from
`models.json` in its agent directory, `~/.pi/agent` or `$PI_CODING_AGENT_DIR`.
The command adds a `vllm-sr` provider there and leaves your other providers
and keys as they are. It shows the change, writes it all or nothing, and
`--undo` puts back the provider you had under that name, or removes it, as for
the other harnesses. Pi is not made the default: select the provider as
above, or with `/model`, where Ctrl+S saves it for new sessions.

- Pi talks to the gateway over Chat Completions (`api: openai-completions`),
  the API its documentation uses for OpenAI-compatible servers such as vLLM.
  On it Pi keeps its own model ID on each reply and records the model ID the
  reply names, the backend's `provider_model_id`, as `responseModel`.
- The file holds no key. `apiKey` is `$VLLM_SR_API_KEY`, which Pi reads from its
  environment before each request, or `!<the command>` with `--key-command`. A
  listener without keys gets a placeholder value, since Pi lists a model only
  once its provider has a key. A key Pi stored for `vllm-sr` in `auth.json`
  wins over `apiKey`, so the command refuses to connect while one is there.
- The model gets the context window the command works out as `contextWindow`
  (Pi assumes 128,000 tokens without one), and as `maxTokens` the output limit
  the command gives Claude Code: 32000, or the smallest `max_output_tokens` of
  the models the entrypoint reaches when that is lower. Pi lowers it for a
  request that leaves less room in the window.
- `compat.sendSessionAffinityHeaders` with `sessionAffinityFormat: "openrouter"`
  has Pi send its session ID as `x-session-id`, which session protection reads.

Pi accepts comments in `models.json`; a file with comments is refused rather
than rewritten without them. The command probes `/v1/chat/completions` with a
basic reply and a streamed tool loop. `--check` runs `pi --mode json` with only
its `bash` tool, no session file, and no extensions, skills or instruction
files. Pi has no allowlist for a single command, so it runs whatever the model
asks for, in the check's empty temporary directory. With `--dry-run` the check
gives Pi an agent directory of its own that holds only the `models.json` the
command would write, so it runs without your Pi login and settings, except
on [Pi's own login](#pi-on-its-own-claude-or-chatgpt-login).

There is no `--launch` for Pi: it reads providers only from `models.json`, so
it cannot reach a gateway for one session without one being written. There is
no vSR plugin for Pi. A gateway that presents each caller's own login upstream
gets Pi's own login, as the next section describes.

### Pi on its own Claude or ChatGPT login

Pi signed in with `/login` (or `pi auth login anthropic`, `pi auth login
openai-codex`) keeps that login on a gateway whose caller backends name its
provider, as [Claude Code](#claude-code-on-its-own-claude-login) and
[Codex](#codex-on-its-own-chatgpt-login) do. `vllm-sr connect pi --config
<your config>` then gives the provider Pi's own command for that login as its
key:

```json
{
  "providers": {
    "vllm-sr": {
      "name": "vLLM Semantic Router",
      "baseUrl": "http://localhost:8899",
      "api": "anthropic-messages",
      "apiKey": "!pi auth print-bearer-token --provider anthropic",
      "compat": { "supportsEagerToolInputStreaming": false },
      "models": [
        { "id": "vllm-sr/auto", "maxTokens": 32000, "contextWindow": 200000 }
      ]
    }
  }
}
```

- Before each request Pi runs `pi auth print-bearer-token`, the command Pi
  offers external clients: it prints the OAuth login Pi stored in `auth.json`,
  and refreshes it first when it expires within 30 minutes. The login stays in
  Pi's store and its refresh stays Pi's. `connect` reads only which kind of
  credential each provider holds there; it never runs the command, so it never
  holds or shows the login.
- A login goes only to the provider that issued it. A gateway whose caller
  backends name `anthropic` gets Pi's `anthropic` login, and one that names
  `openai` its `openai-codex` login (ChatGPT for Codex), or else its `openai`
  login (Sign in with ChatGPT). The command stops before writing anything when
  Pi holds no OAuth login of that provider, a login of the other provider or a
  stored API key included. A gateway key in `$VLLM_SR_API_KEY` is not used.
- The Router takes an Anthropic login only on Messages, so on one Pi talks to
  the gateway over Messages (`api: anthropic-messages`, the listener without
  `/v1` as `baseUrl`), with the login, Claude's OAuth beta and Claude Code's
  identity as Pi sends them to Anthropic. The Router refuses Pi's default
  `tools[].eager_input_streaming` there, so the provider turns it off and Pi
  asks for fine-grained tool streaming with its beta header instead. Pi sends
  no session header on this login, so session protection cannot hold its
  model for a session, and Messages keeps thinking signatures: the Router
  settings below do not apply.
- On an `openai` login Pi stays on Chat Completions. It sends the ChatGPT login
  without the `chatgpt-account-id` header Codex sends beside it.

The gateway is not probed, since it answers only on Pi's login. `--check` runs
Pi on that login. With `--dry-run` the provider in the check's own agent
directory runs the command with `PI_CODING_AGENT_DIR` set to yours, so it reads
your login there. `vllm-sr connect status` shows `Pi's own login (anthropic)`
and sends nothing, and `--undo` takes the provider back as for a key.

### Router settings for Pi

```yaml
global:
  router:
    # only when a decision can pick an anthropic model
    allow_lossy_translation: [reasoning.signature]
    learning:
      enabled: true
      protection:
        enabled: true
        scope: session
```

Pi names no subagent, and it sends its own output limit
(`max_completion_tokens`), so decisions need no `default_max_tokens`. The
command reports what the configuration given with `--config` lacks of these.

### Pi settings by hand

The command writes this provider into `models.json`; these settings follow the
`models.json` reference of Pi 1.5.2:

```json
{
  "providers": {
    "vllm-sr": {
      "name": "vLLM Semantic Router",
      "baseUrl": "http://localhost:8899/v1",
      "api": "openai-completions",
      "apiKey": "$VLLM_SR_API_KEY",
      "compat": {
        "sendSessionAffinityHeaders": true,
        "sessionAffinityFormat": "openrouter"
      },
      "models": [
        { "id": "vllm-sr/auto", "maxTokens": 32000, "contextWindow": 131072 }
      ]
    }
  }
}
```

## VS Code

The [VS Code extension](https://github.com/vllm-project/semantic-router/tree/main/src/agent-harness/vscode)
puts the gateway's entrypoints, such as `vllm-sr/auto`, in VS Code's chat model
picker, for chat and agent mode. It sends each request to the gateway's Chat
Completions API, and the gateway does all the routing. After each reply the
status bar shows the model the gateway chose, from `x-vsr-selected-model`, and
its tooltip the decision, from `x-vsr-selected-decision`.

It needs VS Code 1.104 or later and installs from source, as its README
describes. Set `vllm-sr.gatewayUrl` to the inference address, run **vLLM
Semantic Router: Set Gateway API Key** when the listener has `api_keys`, and
set `vllm-sr.maxInputTokens` as the next section describes. Each VS Code window
sends one `x-session-id`, and no `x-conversation-id`.

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

### Keep long and failed calls in the turn

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

In standalone mode and behind the Envoy the CLI renders, a backend that
refuses a connection does not fail the call: the Router retries an attempt
the backend never received twice, and never resends one it did. An Envoy
configured elsewhere retries only as its routes say (see
[model configuration](model-configuration#tune-timeouts-retries-and-endpoint-health)).
A model that answers with an error, or keeps refusing, fails the call unless
its decision falls back to its next model. Fallback is off by default, because
the next model answers without the first one's prompt cache, and its default
timeouts (30 seconds in all, 10 for each model) end an agent's slower answer;
see [Fall back to another model](model-configuration#fall-back-to-another-model).

`vllm-sr connect` reports both for the entrypoint it connects: each model, or
decision override, whose whole call the Router bounds, with the
`total_timeout: 0s` to add; each decision whose fallback is off or too short
for an agent call, with the block to add where it is set (the decision's own
`fallback`, its recipe's, or `global.router.fallback`); and each decision that
names one model:

```text
- fallback on decision all, so a model call that fails moves to the decision's next model ... with 1200s for it: ...
  global:
    router:
      fallback:
        enabled: true
        total_timeout: 1200s
        per_attempt_timeout: 1200s
- total_timeout: 0s on qwen-msg (300s, the listener's timeout): the Router cuts a call off ...
  providers:
    models:
      - name: qwen-msg
        reliability:
          total_timeout: 0s
```

Behind Envoy a fallback answer also ends at the listener's `timeout` (300
seconds by default), since the Router holds the response while the fallback
call runs; raise it there for agents that fall back on long calls.

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
Codex CLI sends `session-id` and `thread-id`. Protection reads both without
identity headers configured; see
[Agents that delegate to subagents](../tutorials/learning/protection#agents-that-delegate-to-subagents)
for how they hold a run and its subagents.

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
asks for adaptive thinking with its effort setting on every call. When the
decision turns reasoning off for a Messages backend, the Router also drops the
thinking display the harness asked for, since no thinking comes back. For a
Chat or Responses backend the display stays, and the Router keeps the
backend's reasoning out of the harness's answer as it asked. Set
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

## Troubleshooting

Each row starts from what the harness shows. `vllm-sr connect status` sends one
short request with the key the harness would send, and the Router log has a
line for each request it refused. The messages are those of Claude Code
2.1.325, Codex CLI 0.166.0, Pi 1.5.2 and the VS Code extension in VS Code 1.146.

| Symptom | Check | Fix |
| --- | --- | --- |
| The gateway refuses the key. `claude -p` prints nothing for about three minutes while it retries, then `Failed to authenticate. API Error: 401 Unauthorized: missing or invalid API key`. Codex: `unexpected status 401 Unauthorized`. Pi: `401: {"message":"Unauthorized: missing or invalid API key"`. VS Code: `The gateway answered 401: Unauthorized: missing or invalid API key.` | `vllm-sr connect status` says `the gateway did not accept the key`. `claude -p --output-format stream-json --verbose` shows each retry as an `api_retry` event with `"error_status":401`. The Router logs an `access` line with `"response_code":401`. | The key is wrong, or the listener's `api_keys` changed. Claude Code keeps the key in its settings: export the current key as `VLLM_SR_API_KEY` and run `vllm-sr connect claude-code` again. Codex and Pi read the variable when they start, so export it in the shell that starts them. In VS Code, run **vLLM Semantic Router: Set Gateway API Key**. |
| Codex says `` Missing environment variable: `VLLM_SR_API_KEY`. `` Pi says `No API key found for vllm-sr.` | `vllm-sr connect status` says `not probed: $VLLM_SR_API_KEY is not set`. | Export the listener's key as `VLLM_SR_API_KEY` before you start the harness, or connect again with `--key-command` ([Fetch the key with a command](#fetch-the-key-with-a-command)). |
| The gateway does not answer at that address. `claude -p` retries for about three minutes, then `API Error: Connection refused`. Codex keeps printing `Reconnecting... waiting for network`. Pi: `Connection error.: connect ECONNREFUSED`. VS Code: `The gateway at <address> refused the connection`. | `vllm-sr connect status` says `no connection (Connection refused). Is the gateway running there?` after the URL it tried. | Start the Router, or run connect again with the `--config` the Router serves or with `--base-url`. In VS Code, set `vllm-sr.gatewayUrl`. |
| A `--key-command` fails or prints nothing. `vllm-sr connect` stops with `--key-command exited with status 1`, `--key-command printed no key` or `--key-command did not finish in 5 s`, and writes nothing. When it stops working after connect: Claude Code says `apiKeyHelper failed: did not return a value`, then `Your apiKeyHelper script is failing`. Codex: ``provider auth command `/bin/sh` produced an empty token``. Pi: `API key auth failed for provider vllm-sr`. | `vllm-sr connect status` runs the command as the harness does and says `--key-command printed no key` or `--key-command exited with status 1`. Run the command in a shell to see its error. | Make the command print the key, for example by unlocking the password store, and start a new session. The settings keep the command, so connect does not need to run again. |
| The harness asks for a model the gateway does not serve. Claude Code: `API Error: 400 the requested model is not available`. Codex: `the requested model is not available`, after a warning that it found no model metadata. Pi: `400: {"type":"invalid_request_error","code":"model_not_found"`. | `GET /v1/models` lists what the gateway serves. The Router logs `specified_model_not_found` with the `model` the harness sent. `codex exec` names it on its `model:` line. | Ask for an entrypoint such as `vllm-sr/auto`, or add the ID to an entrypoint's `model_names` ([Model names the harness picks itself](#model-names-the-harness-picks-itself)). When a project's settings name the model, see the next two rows. |
| Claude Code started in one directory asks for another model. | `vllm-sr connect status` run in that directory shows `overridden here` with the file and keys, such as `.claude/settings.json sets ANTHROPIC_MODEL`, and `vllm-sr connect claude-code` there warns that `Claude Code started in this directory takes those over` the user settings. | Remove those keys from the project's settings, connect there with `--scope project`, or start with `--launch` ([Try it without writing anything](#try-it-without-writing-anything)). |
| Codex in a trusted project asks for another model: its `model:` line names it, and the gateway answers `the requested model is not available`. | Run `vllm-sr connect codex` in that directory. It warns that `.codex/config.toml sets model, which Codex asks for instead of vllm-sr/auto once it trusts this project`. | Remove `model` from the project's `.codex/config.toml`, or start with `--launch`. |
| Pi answers `401` although `VLLM_SR_API_KEY` holds the right key. | `vllm-sr connect status` says `Pi holds a key of its own for vllm-sr in auth.json, which it sends in place of the one models.json names`. | Remove the `vllm-sr` entry from Pi's `auth.json`. |
| The request is larger than the context window. Claude Code: `Prompt is too long`. Codex: `This model's maximum context length is`, in an error with code `context_length_exceeded`. VS Code: `The gateway answered 400: This model's maximum context length is`. `pi -p` prints nothing and exits with status 0, and `pi --mode json` shows the error in `errorMessage`. | When the Router refuses it on its own count, it logs `context_window_exceeded` with the window and the count. `vllm-sr connect status` shows the `context window` the harness was given. | Give the harness the smallest window of the models the entrypoint reaches: run connect again after those models change, or pass `--context-window`. In VS Code, set `vllm-sr.maxInputTokens` ([Set budgets](#set-budgets-before-running-a-task)). |
| `vllm-sr connect codex --check` shows `not observed` on its `Approval review` line. | The Router logs a `routing_decision` line with `"original_model":"vllm-sr/review"` for a review that reached it. If `GET /api/v1/observability/replays?model=vllm-sr/review` on the management API then lists nothing, replay is off for the review recipe. Without that log line, the model ran the command without asking for approval. | Remove `enabled: false` from the `router_replay` plugin of the review recipe to record reviews. A model that runs the command without asking does not fail the check. |
| Approve for me reviews go to the session model after you set a `review_model`. | `vllm-sr connect status` shows `the session model` as the review model. | Run connect again after you set or change `review_model` ([Approve for me reviews](#approve-for-me-reviews)). |
| Insights shows no requests, although the harness gets answers. | `GET /api/v1/observability/replays` on the management API answers 404 with `router replay is disabled or unavailable`. | Replay is on by default. Remove `enabled: false` from the `router_replay` plugin in `global.plugins`, or set it to `true`, and restart the Router. |
| Insights lost the requests from before the Router restarted. | The list answers `"count":0` right after a restart, and either `global.services.router_replay.store_backend` is `memory` or the Router logged at startup that its records `stay in memory`. | Remove `store_backend: memory` so the default `file` store keeps them, make the configuration's directory writable, or configure Postgres or Redis ([Keep records across restarts](../tutorials/plugin/router-replay#keep-records-across-restarts)). |
| `vllm-sr connect` stops before writing anything, with `so every call would be refused`, for a gateway that presents each client's own login. | The configuration's `auth: {credential: caller}` backends name `openai` while Claude Code sends an `anthropic` login, or the reverse for Codex. For Pi the message names the provider it has no login for, such as `Pi holds no anthropic login`. | Point Claude Code at a gateway whose caller backends name `anthropic`, and Codex at one whose caller backends name `openai`. Sign Pi in with `pi auth login anthropic` or `pi auth login openai-codex`, and run connect again. |
| `claude -p` prints `[claude-code:unrecognized_model] {"model":"vllm-sr/auto","query_source":"sdk"}` before the answer. | The answer follows. | Nothing to fix: Claude Code does not know the entrypoint's name. |
