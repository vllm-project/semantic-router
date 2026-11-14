# vSR: vLLM Semantic Router in Codex

vSR puts a small decision model from the vLLM Semantic Router project beside Codex CLI, as System One
beside the agent: before Codex runs a shell command it asks whether a person should look first, and
before Codex reads a command's output it asks whether the output holds personal data or tries to give
the agent orders. Codex's own requests still go to its model as usual. Three of the Claude Code
plugin's apps work in Codex:

| App | What it does in Codex |
| --- | --- |
| guard | Blocks a shell command the decision model scores at or over `guardReview` (0.90), and tells Codex why, so it asks you instead of running it. |
| redact | Replaces personal data in a command's output with tags such as `[vsr:email]` before Codex reads it, and tells Codex the tags are vSR's and the values are hidden on purpose, not to be read another way. |
| shield | Tells Codex to read an output that tries to give an AI agent instructions as data, when the injection score is at or over `shieldInject` (0.90). |

## Install

You need Codex CLI 0.166 or later, Python 3, and a decision runtime: `vllm-srun serve` on your
machine, or a Router listener that publishes a System One model.

```bash
codex plugin marketplace add vllm-project/semantic-router
codex plugin add vsr@vllm-sr
```

The next Codex session says `2 hooks are new or changed`. Choose **Review hooks** to read the two
commands, or **Trust all and continue**. Codex runs a plugin's hooks only once they are trusted, and
asks again after each update, since it hashes the plugin's files.

## Configure

vSR reads `vsr.json` in the Codex home (`$CODEX_HOME`, by default `~/.codex`). Every key is optional:

```json
{
  "runtimeUrl": "http://127.0.0.1:8790/v1/decisions",
  "runtimeModel": "vela3-lite",
  "guard": true,
  "redact": true,
  "shield": true,
  "guardReview": 0.9,
  "redactPii": 0.7,
  "shieldInject": 0.9,
  "timeoutSeconds": 3
}
```

`VSR_RUNTIME_URL` and `VSR_RUNTIME_MODEL` override the file, and `VSR_RUNTIME_TOKEN` is sent as a
bearer token; keep the token out of `vsr.json`. With Codex pointed at a Router through
`vllm-sr connect codex`, set `runtimeUrl` to that listener's `/v1/decisions` and `runtimeModel` to the
System One model it publishes, so one listener serves both the agent's calls and vSR's questions.

That listener can also turn `guard`, `redact` and `shield` on or off for every plugin pointed at it,
under `apis.systemone.vsr` in the Router's config or as checkboxes in the Dashboard (Config › Global
Config › vSR agent plugins).
vSR reads the listener's listing (`GET /v1/systemone/models`) at most once a minute and keeps the
answer in `vsr-listener.json` in the Codex home. An app you name in `vsr.json` stays yours; the
listener sets the others.

## What you see

Each decision is one line in the transcript, starting with vSR's mark:

```text
• Blocked by hook
  └ ▚▞ vSR held · git push --force origin main · looks destructive · risk score 0.98 ≥ 0.90
↳ Hook · ▚▞ vSR shielded · the output of cat vendor/NOTES.md tried to give Codex instructions · injection 0.95 ≥ 0.90
• Blocked by hook
  └ ▚▞ vSR redacted · 9 items hidden: email, name, phone · the output of head -5 customers.csv
```

"Blocked by hook" is Codex's label for a hook that replaced what the model reads. For redact, the
command ran and you see its full output; Codex's model reads the copy with the tags, shown under the
line.

## How it differs from Claude Code

- **guard blocks rather than asks.** Codex 0.166 rejects a hook's `ask` (`unsupported
  permissionDecision:ask`), so a held command is denied with the reason, and Codex tells you what it
  would have run. Run it yourself if you want it. Guard asks about the whole command line, where the
  Claude Code plugin checks a chain part by part and holds its fixed catastrophic commands without a
  question.
- **redact reads shell output only.** Codex reads files through shell commands, so every read goes
  through it, but it does not hide secrets by pattern as the Claude Code plugin does, and it reads the
  first 16,000 characters of an output; past them the line says the rest went unchecked.
- **Not in Codex yet:** router, escalate and check, the `/vsr` pane and its answers, the decision log
  and `/vsr:calibrate`.

A decision runtime that does not answer never stops Codex: the call goes on as Codex would run it, and
the line says what went unchecked (`▚▞ vSR guard skipped · the decision model is not answering`).
