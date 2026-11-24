# vSR Copilot CLI plugin

- This folder is a Copilot CLI plugin (`vsr`), listed in the root
  `.github/plugin/marketplace.json` as `vsr@vllm-sr`. It is a client of a
  decision runtime's public `/v1/decisions` API; it never imports router or
  runtime code, and nothing from `../claude-code` or `../codex`.
- `hooks/vsr.py` handles both hook events and uses only the Python standard
  library, since Copilot CLI runs it with the user's `python3`. There is no
  Stop hook here: Copilot CLI's Stop payload carries no turn id, so there is
  nothing for a turn-summary line (the one the Codex plugin ends each turn
  with) to key off.
- `hooks/framing.py` is a copy of the Claude Code plugin's questions. Change
  the Claude Code plugin first, then copy the file here; `test_framing_copy.py`
  fails while the two differ.
- A changed hook command or script makes every user trust the plugin's hooks
  again, so keep `hooks.json` stable.
- Reply to a hook event with Copilot CLI's own documented flat fields
  (`permissionDecision`, `modifiedResult`, `additionalContext`), not Claude
  Code's nested `hookSpecificOutput`: Copilot CLI accepts the nested form for
  some replies but silently drops a `PostToolUse` `additionalContext` nested
  that way.
- Before a change is done, run `python3 -m unittest discover -s hooks`. CI
  runs it through the `copilot-cli-plugin` entry in `tools/agent/domains.yaml`.
