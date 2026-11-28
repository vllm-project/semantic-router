# vSR Copilot CLI plugin

- This folder is a Copilot CLI plugin (`vsr-copilot`), listed in the root
  `.github/plugin/marketplace.json` as `vsr-copilot@vllm-sr`. It is a client of
  a decision runtime's public `/v1/decisions` API; it never imports router or
  runtime code, and nothing from `../claude-code` or `../codex`.
- Copilot CLI reads only the first marketplace file it finds, and
  `.github/plugin/marketplace.json` comes before `.claude-plugin/marketplace.json`.
  That is why the file also lists the Claude Code plugin as `vsr`; keep that
  entry's name and source the same as in `.claude-plugin/marketplace.json`.
- `hooks/vsr.py` handles both hook events and uses only the Python standard
  library, since Copilot CLI runs it with the user's `python3`. There is no
  Stop hook here: Copilot CLI's Stop payload carries no turn id, so there is
  nothing for a turn-summary line (the one the Codex plugin ends each turn
  with) to key off.
- `hooks/framing.py` is a copy of the Claude Code plugin's questions. Change
  the Claude Code plugin first, then copy the file here; `test_framing_copy.py`
  fails while the two differ.
- `main()` in `hooks/vsr.py` catches every error and prints `{}`: Copilot CLI
  denies a command when its `PreToolUse` hook crashes or exits non-zero, so a
  bug here would stop every shell call. Keep it, and keep any new setting
  tolerant of a wrong-typed value in `vsr.json`.
- `hooks.json` keeps the `Bash` matcher and the two timeouts (15 seconds before
  the command, 30 after it) that were measured on Copilot CLI 1.0.114. A hook
  that outruns its timeout is skipped and the command runs unchecked, so keep
  the `timeoutSeconds` default in `hooks/vsr.py` below them.
- Reply to a hook event with Copilot CLI's own documented flat fields
  (`permissionDecision`, `modifiedResult`, `additionalContext`), not Claude
  Code's nested `hookSpecificOutput`: Copilot CLI accepts the nested form for
  some replies but silently drops a `PostToolUse` `additionalContext` nested
  that way.
- Before a change is done, run `python3 -m unittest discover -s hooks`. CI
  runs it through the `copilot-cli-plugin` entry in `tools/agent/domains.yaml`.
