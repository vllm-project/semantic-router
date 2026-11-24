# Developing vSR for Copilot CLI

## Layout

| Path | What it holds |
| --- | --- |
| `plugin.json` | The manifest: name `vsr`, the hooks file. The root `.github/plugin/marketplace.json` lists this folder as `vsr@vllm-sr`. |
| `hooks.json` | Two command hooks, both on `Bash`: `PreToolUse` runs `hooks/vsr.py pre` and `PostToolUse` runs `hooks/vsr.py post`. `statusMessage` names each hook on Copilot CLI's review screen. There is no `Stop` hook, unlike the Codex plugin: Copilot CLI's `Stop` payload carries no turn id, so there is nothing for an end-of-turn summary line to key off. |
| `hooks/vsr.py` | The hooks: guard, then redact and shield. Settings come from `vsr.json` in the Copilot home and `VSR_*` variables. Python 3 standard library only. |
| `hooks/framing.py` | The Claude Code plugin's questions, copied from `claude-code/skills/setup/scripts/framing.py`; `test_framing_copy.py` fails when the two differ. |
| `hooks/test_vsr.py` | The hooks against a stub decision runtime. |

## Run and check

Start a runtime, add this repository as a marketplace, and install the plugin:

```bash
vllm-srun serve vllm-sr/vela3-lite --device cpu --host 127.0.0.1 --port 8790
copilot plugin marketplace add "$(git rev-parse --show-toplevel)"
copilot plugin install vsr@vllm-sr
copilot
```

The first command finds `.github/plugin/marketplace.json`, ahead of
`.claude-plugin/marketplace.json` in Copilot CLI's own discovery order, so it
never resolves to the Claude Code plugin's unrelated `vsr` entry there. A
plugin installed from a local path marketplace loads live from the checkout
on `/restart` or a new session, so a change here needs no reinstall. To try
one hook without a session, pipe an event into it:

```bash
echo '{"tool_name": "Bash", "tool_input": {"command": "git push --force origin main"}}' \
  | VSR_RUNTIME_URL=http://127.0.0.1:8790/v1/decisions python3 hooks/vsr.py pre
```

Or load the plugin straight from this folder, isolated from your own Copilot
home, the way the folder was first proven against a real session:

```bash
COPILOT_HOME=/tmp/vsr-copilot-home copilot --plugin-dir "$(pwd)" -p "run echo hi"
```

Before a change is done:

```bash
python3 -m unittest discover -s hooks
```

CI's Agent Plugins contract runs this, with the Claude Code and Codex
plugins' tests, through `make agent-plugins-test` whenever any of the three
plugin folders change.

## What Copilot CLI 1.0.114 does with each hook answer

| Hook answer | What happens |
| --- | --- |
| `PreToolUse` `permissionDecision: "deny"` | The command does not run; the model reads the `permissionDecisionReason`, and the transcript shows it was denied by the hook. Confirmed live: pointed at a real `vllm-srun serve vllm-sr/vela3-lite`, `git push --force origin main` came back `✗ Shell Denied by preToolUse hook: vSR guard held this command: it looks destructive (risk score 0.98 ≥ 0.90)...`, and no push was attempted. |
| `PostToolUse` `modifiedResult: {resultType: "success", textResultForLlm}` | The model reads `textResultForLlm` in place of the real output; the person still sees the output the command printed. Confirmed live the same way: `cat` on a file with a planted name and email came back to the model as `name,email` / `[vsr:name],[vsr:email]`, while the file on disk still held the real values. |
| `PostToolUse` top-level `additionalContext` | Appended to the tool output the model reads, on the same turn. Per GitHub's hooks reference, not independently re-run here. |
| `PostToolUse` `additionalContext` nested under `hookSpecificOutput`, the way Claude Code and Codex read it | Not read for `PostToolUse` here: it did not reach the model in the probe behind https://github.com/vllm-project/semantic-router/issues/7326, and GitHub's own hooks reference documents only the flat, top-level field for this event. |
| `systemMessage` | Not shown in `-p` output or the session's own event log; it only showed up in Copilot CLI's debug log in that same probe. Interactive display is untested, so vSR still sets it: it costs nothing when unread. |

A `Bash` event's `tool_input` is `{"command": "..."}`, matching Codex's
shape, so `guard()` needs no adapter there. `PostToolUse`'s output sits at
`tool_result.text_result_for_llm`, not the top-level `tool_response` Codex
reads, because hooks.json's `PascalCase` event names select Copilot CLI's
"VS Code compatible" snake\_case payload format. Copilot CLI sets
`PLUGIN_ROOT` for a command hook the same way Codex does.
