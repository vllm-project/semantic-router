# Developing vSR for Copilot CLI

## Layout

| Path | What it holds |
| --- | --- |
| `plugin.json` | The manifest: name `vsr-copilot` and the hooks file. The root `.github/plugin/marketplace.json` lists this folder as `vsr-copilot@vllm-sr`. |
| `hooks.json` | Two command hooks on `Bash`: `PreToolUse` runs `hooks/vsr.py pre` (15-second timeout) and `PostToolUse` runs `hooks/vsr.py post` (30 seconds). |
| `hooks/vsr.py` | The hooks: guard, then redact and shield. Settings come from `vsr.json` in the Copilot home and `VSR_*` variables. Python 3 standard library only. |
| `hooks/framing.py` | The Claude Code plugin's questions, copied from `claude-code/skills/setup/scripts/framing.py`; `test_framing_copy.py` fails when the two differ. |
| `hooks/test_vsr.py` | The hooks against a mocked decision runtime. |

The same marketplace file lists the Claude Code plugin as `vsr`. Copilot reads only the first
marketplace file it finds, and `.github/plugin/` comes before `.claude-plugin/`, so without that
entry Copilot users would lose the Claude Code plugin's skills. Keep that entry's name and source the
same as in `.claude-plugin/marketplace.json`.
`hooks.json` has no `statusMessage` because the Copilot hooks reference does not list one.

## Run and check

Start a runtime in one terminal:

```bash
vllm-srun serve vllm-sr/vela3-lite --device cpu --host 127.0.0.1 --port 8790
```

In another, install the plugin from this checkout and start Copilot:

```bash
copilot plugin marketplace add "$(git rev-parse --show-toplevel)"
copilot plugin install vsr-copilot@vllm-sr
copilot
```

A marketplace added from a local path loads the plugin from the checkout, so after a change the next
session runs the new code with no reinstall. On a CPU a decision call takes about 4 seconds: put
`{"timeoutSeconds": 12}` in `vsr.json` in the Copilot home first, because with the default 3 seconds
every call goes through unchecked. An interactive session also asks to trust the folder; `copilot -p`
does not.

To try one hook without Copilot, pipe an event into it from this folder:

```bash
echo '{"hook_event_name":"PreToolUse","tool_name":"Bash","tool_input":{"command":"git push --force origin main"}}' \
  | VSR_RUNTIME_URL=http://127.0.0.1:8790/v1/decisions python3 hooks/vsr.py pre
printf '%s' '{"hook_event_name":"PostToolUse","tool_name":"Bash","tool_input":{"command":"cat customers.csv"},"tool_result":{"result_type":"success","text_result_for_llm":"name,email\nMina Park,mina.park@example.com\n<shellId: 0 completed with exit code 0>"}}' \
  | VSR_RUNTIME_URL=http://127.0.0.1:8790/v1/decisions python3 hooks/vsr.py post
```

The first prints a `permissionDecision` of `deny`; the second prints a `modifiedResult` with the name
and email replaced by `[vsr:name]` and `[vsr:email]`. Without the `timeoutSeconds` setting above, a CPU
runtime answers too late and the first prints only a `systemMessage` saying the guard skipped the call.

Before a change is done:

```bash
python3 -m unittest discover -s hooks
```

CI's Agent Plugins contract runs these tests, with the other plugins' tests, through
`make agent-plugins-test` whenever this folder or `.github/plugin/` changes.

## What Copilot CLI 1.0.114 does with each hook answer

Measured on Copilot CLI 1.0.114 with claude-haiku-4.5, in `copilot -p` runs and in interactive
sessions, against `vllm-srun` serving vela3-lite:

| Hook answer | What happens |
| --- | --- |
| `PreToolUse` deny | The command does not run. The transcript shows `Denied by preToolUse hook: <reason>`, the session log records the call as failed with `error.code: "denied"`, and the model reads the reason. No `PostToolUse` hook runs. |
| `PostToolUse` `modifiedResult` | Replaces the output in place: the model and the transcript both show the new text. |
| `PostToolUse` `additionalContext` | Appended after the output, under `Additional guidance from postToolUse hooks:`, when it is a top-level key. Nested under `hookSpecificOutput` it did not reach the model in two runs. |
| `PostToolUse` `decision: "block"` | The model reads the `reason` in place of the output, and the transcript shows `Tool result blocked: <reason>`. |
| `systemMessage` | Not drawn in `copilot -p` or in an interactive session. It is in the debug log. |
| A command that exits non-zero | `PostToolUse` still runs and the rewrite applies. |

A `Bash` event's `tool_input` is `{"command": "...", "description": "..."}`, and `PostToolUse`'s
`tool_result` is `{"result_type": "success", "text_result_for_llm": "..."}`. The exit code is only in
the footer that ends the text, `<shellId: N completed with exit code M>`. `vsr.py` sets the footer
aside before it asks anything and puts it back last, and any rewrite must keep it last. Copilot sets
`PLUGIN_ROOT` (and `COPILOT_PLUGIN_ROOT` and `CLAUDE_PLUGIN_ROOT`) to the plugin's folder.

## Limits to know

- **Copilot's file viewer is not covered.** Redact and shield read the `Bash` tool's output only. In
  a measured run where Copilot read a file with its `view` tool instead of `cat`, the file's emails
  and phone numbers reached the model as they were. Grep, glob, web fetch and MCP tools were not
  measured and probably behave the same. Covering them is a follow-up: the debug log shows the `view`
  result as `textResultForLlm` text, so the same rewrite should fit, but the post hook would have to
  accept more tool names and the stdin fields for them were not observed.
- **The `Bash` matcher does not stop the hook from starting.** In that `view` run both hook
  processes still started. `main()` returns `{}` for any other tool before it reads settings or asks
  anyone, so nothing is asked and nothing is slowed; do not rely on the matcher alone. Copilot's
  reference maps both `bash` and `powershell` to `Bash`; Windows was not tried.
- **Failures.** A runtime that is down, slow or answering an error leaves the call alone: the hook
  prints a `skipped` line and Copilot runs the call as it would have. A crash inside `vsr.py` is
  caught in `main()`, which prints `vsr <mode>: skipped, <ErrorName>` to stderr and lets the call run,
  because the [hooks reference](https://docs.github.com/en/copilot/reference/hooks-reference) says a
  `preToolUse` command that crashes or exits non-zero is denied and that a timeout is always let
  through. A `vsr.json` key of the wrong type is ignored and a malformed `vsr-listener.json` reads as
  empty.
- **Timeouts.** `timeoutSeconds` bounds each call to the runtime. The guard hook may also read the
  listener's listing (up to 1.5 seconds) and ask for listener rules before its own question, so keep
  `timeoutSeconds` well under the 15 seconds `hooks.json` gives it; the post hook has 30.
- **Output past 16,000 characters is not read.** The hook reports the unchecked part in a
  `systemMessage` line, which Copilot 1.0.114 does not draw.
- **No turn line.** The Codex plugin ends each turn with the models the Router served. Copilot's
  `Stop` payload carries no turn id and the reference gives `Stop` no way to show the person a line,
  so this plugin has no `Stop` hook.
- **Guard denies; it does not ask.** `preToolUse` can also answer `permissionDecision: "ask"`, which
  the reference says is denied in autopilot mode and in the cloud agent because nobody can answer.
  Denying is what was measured, as in the Codex plugin. Trying `ask` is a follow-up.
- **Shield's effect is not shown.** With claude-haiku-4.5 the model refused the injected command with
  and without the plugin, so the runs show that the note arrives and what it says, not that it
  changes the outcome.
- **The listener path is tested against a mock only.** `test_vsr.py` covers the listing, the cache
  and bound rules. The live runs used `vllm-srun`, which serves only `/v1/decisions`, so the listing
  read there fails open to the defaults.
- **Docker and the Copilot app were not used.** The live runs started `vllm-srun` directly, which
  serves the same `/v1/decisions` contract as `vllm-sr serve`, and ran Copilot CLI only. A
  marketplace added from GitHub rather than a local path was not tried either.
- **The Claude Code plugin's hooks file is not a Copilot hooks file.** Its `hooks/hooks.json` is a
  module list, so Copilot's debug log carries an `Invalid hooks config for plugin "vsr"` error when
  it is installed. Its skills install and its hooks never run. This is a property of that plugin.
