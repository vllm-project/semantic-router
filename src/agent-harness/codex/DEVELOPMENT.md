# Developing vSR for Codex

## Layout

| Path | What it holds |
| --- | --- |
| `.codex-plugin/plugin.json` | The manifest: name `vsr`, the hooks file, the icon. The root `.agents/plugins/marketplace.json` lists this folder as `vsr@vllm-sr`. |
| `hooks.json` | Three command hooks: on `Bash`, `PreToolUse` runs `hooks/vsr.py pre` and `PostToolUse` runs `hooks/vsr.py post`; `Stop` runs `hooks/vsr.py stop` at the end of each turn. `statusMessage` names each hook on Codex's review screen. |
| `hooks/vsr.py` | The hooks: guard, then redact and shield, and the turn line. Settings come from `vsr.json` in the Codex home and `VSR_*` variables. Python 3 standard library only. |
| `hooks/framing.py` | The Claude Code plugin's questions, copied from `claude-code/skills/setup/scripts/framing.py`; `test_framing_copy.py` fails when the two differ. |
| `hooks/test_vsr.py` | The hooks against a stub decision runtime and a stub Router management API. |

## Run and check

Start a runtime, install the plugin from this checkout into a Codex home, and trust its hooks on the
first session:

```bash
vllm-srun serve vllm-sr/vela3-lite --device cpu --host 127.0.0.1 --port 8790
codex plugin marketplace add "$(git rev-parse --show-toplevel)"
codex plugin add vsr@vllm-sr
codex
```

After a change, `codex plugin add vsr@vllm-sr` again copies it into the Codex home, and the next
session asks to trust the changed hooks. To try one hook without Codex, pipe an event into it:

```bash
echo '{"tool_name": "Bash", "tool_input": {"command": "git push --force origin main"}}' \
  | VSR_RUNTIME_URL=http://127.0.0.1:8790/v1/decisions python3 hooks/vsr.py pre
echo '{"hook_event_name": "Stop", "turn_id": "<a turn id from Router Replay>"}' \
  | VSR_MANAGEMENT_URL=http://127.0.0.1:8080 python3 hooks/vsr.py stop
```

Before a change is done:

```bash
python3 -m unittest discover -s hooks
```

CI's Agent Plugins contract runs them, with the Claude Code plugin's tests, through
`make agent-plugins-test` whenever either plugin changes.

## What Codex 0.166 does with each hook answer

Measured on Codex CLI 0.166.0 through a Router listener with vela3-lite:

| Hook answer | What happens |
| --- | --- |
| `PreToolUse` `permissionDecision: "deny"` | The command does not run; the model reads `Command blocked by PreToolUse hook: <reason>`, and the transcript shows `Blocked by hook` with the `systemMessage` and the reason. |
| `PreToolUse` `permissionDecision: "ask"` | Rejected as unsupported. |
| `PostToolUse` `decision: "block"` | The model reads the `reason` in place of the output; the person still sees the output the command printed. |
| `PostToolUse` `additionalContext` | Added for the model; the `systemMessage` shows as `↳ Hook · …`. |
| Untrusted hooks in `codex exec` | Skipped without a word. `--dangerously-bypass-hook-trust` runs them, for tests only. |

A `Bash` event's `tool_input` is `{"command": "..."}`, and `PostToolUse`'s `tool_response` is the
output as one string. Codex sets `PLUGIN_ROOT` (and `CLAUDE_PLUGIN_ROOT`) to the installed copy.

The `Stop` contract is read from Codex 0.166's source, not measured: the hooks crate's
`stop.command.input` schema and `core/src/hook_runtime.rs`. The event carries `session_id`,
`turn_id`, `cwd`, `model`, `permission_mode`, `stop_hook_active`, `transcript_path` and
`last_assistant_message`. `turn_id` is the turn's `sub_id`, the same id Codex puts in the
`x-codex-turn-metadata` header and `client_metadata` of every model call of the turn, which is what
the turn line asks Router Replay for. The answer takes `systemMessage`, shown as `↳ Hook · …`, and
the universal keys; any other key, such as `hookSpecificOutput`, fails the hook. Root turns run
`Stop`; a thread-spawned subagent's turns run `SubagentStop`, which vSR does not hook.
