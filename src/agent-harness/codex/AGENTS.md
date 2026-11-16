# vSR Codex plugin

- This folder is a Codex plugin (`vsr`), listed in the root
  `.agents/plugins/marketplace.json` as `vsr@vllm-sr`. It is a client of a
  decision runtime's public `/v1/decisions` API; it never imports router or
  runtime code, and nothing from `../claude-code`.
- `hooks/vsr.py` handles both hook events and uses only the Python standard
  library, since Codex runs it with the user's `python3`.
- `hooks/framing.py` is a copy of the Claude Code plugin's questions. Change
  the Claude Code plugin first, then copy the file here; `test_framing_copy.py`
  fails while the two differ.
- A changed hook command or script makes every user trust the plugin's hooks
  again, so keep `hooks.json` stable.
- Before a change is done, run `python3 -m unittest discover -s hooks`. CI
  runs it through the `codex-plugin` entry in `tools/agent/domains.yaml`.
