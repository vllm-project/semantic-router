# vSR: vLLM Semantic Router in Copilot CLI

vSR puts a small decision model from the vLLM Semantic Router project beside Copilot CLI, as
System One beside the agent: before Copilot CLI runs a shell command it asks whether a person
should look first, and before Copilot CLI reads a command's output it asks whether the output
holds personal data or tries to give the agent orders. Copilot CLI's own requests still go to
its model as usual. Three of the Claude Code plugin's apps work here:

| App | What it does in Copilot CLI |
| --- | --- |
| guard | Blocks a shell command the decision model scores at or over `guardReview` (0.90), and tells Copilot CLI why, so it asks you instead of running it. |
| redact | Replaces personal data in a command's output with tags such as `[vsr:email]` before Copilot CLI reads it, and tells it the tags are vSR's and the values are hidden on purpose, not to be read another way. |
| shield | Tells Copilot CLI to read an output that tries to give an AI agent instructions as data, when the injection score is at or over `shieldInject` (0.90). |

There is no end-of-turn summary line here, unlike the Codex plugin: Copilot CLI's `Stop` event
carries no id that ties its calls to a turn, so there is nothing for such a line to key off.

Copilot CLI already installs the Claude Code plugin's `/vsr:setup` and `/vsr:calibrate` skills
from this repository's `.claude-plugin/marketplace.json`; this plugin ships hooks only; install
the Claude Code plugin alongside it for the skills.

## Install

You need Copilot CLI 1.0.114 or later, Python 3, and a decision runtime: `vllm-srun serve` on
your machine, or a Router listener that publishes a System One model.

```bash
copilot plugin marketplace add vllm-project/semantic-router
copilot plugin install vsr@vllm-sr
```

This repository's `.github/plugin/marketplace.json` is Copilot CLI's own canonical location for
a repo-hosted marketplace, read ahead of `.claude-plugin/marketplace.json` in its discovery
order, so the install resolves to this plugin rather than the Claude Code plugin's own `vsr`
entry there. Copilot CLI may ask to trust the plugin's hooks the first time they run; accept to
turn guard, redact and shield on.

## Configure

vSR reads `vsr.json` in the Copilot home (`$COPILOT_HOME`, by default `~/.copilot`). Every key
is optional:

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

`VSR_RUNTIME_URL` and `VSR_RUNTIME_MODEL` override the file, and `VSR_RUNTIME_TOKEN` is sent as
a bearer token; keep the token out of `vsr.json`. `runtimeTokenEnv` names another variable to
read the token from when `VSR_RUNTIME_TOKEN` is unset, for a listener with `api_keys`. Point
`runtimeUrl` at that listener's `/v1/decisions` and `runtimeModel` at the System One model it
publishes to have one listener serve both Copilot CLI's calls and vSR's questions.

That listener can also turn `guard`, `redact` and `shield` on or off for every plugin pointed at
it, under `apis.systemone.vsr` in the Router's config or as Default, On or Off in the Dashboard
(Settings › Router settings › vSR agent plugins). vSR reads the listener's listing
(`GET /v1/systemone/models`) at most once a minute and keeps the answer in `vsr-listener.json` in
the Copilot home. An app you name in `vsr.json` stays yours; the listener sets the others.

Rules the listener binds to guard and redact (`vsr.rules`) come with the same listing, by name
and description only, and the listener evaluates them. Guard sends the whole command beside its
own question: a rule that matches denies it with the rule's description, and the person can run
it. Redact sends the Bash output whole and the decision model reads its first 16,000 characters,
the same text, and what either finds is hidden together, a rule's find as `[vsr:<rule name>]`. A
`required` rule the listener could not check denies the command or withholds the output; an
advisory one adds a line saying it was not checked. Before vSR has read a listener's listing once
with this token it knows no rules, and when the listing does not answer, its checks step aside as
they do for any outage.

## What you see

guard's deny reaches Copilot CLI's model as `permissionDecisionReason`, and the person sees the
command blocked with that reason:

```text
✗ Shell  Denied by preToolUse hook: vSR guard held this command: it looks destructive (risk
  git push --force origin main
  score 0.98 ≥ 0.90). Do not run it or a variant of it. Tell the person what it does and why
  it was held; they can run it themselves if they want it.
```

redact reaches the model as the tool result Copilot CLI shows it; the person still sees the
command's own, unredacted output in the transcript, and only the model's copy carries the tags:

```text
$ Shell  Read customers.csv exactly 7 lines…
  cat customers.csv
```

> The visible command output was:
>
> ```text
> name,email
> [vsr:name],[vsr:email]
> ```
>
> The name and email were redacted; those placeholders are not the original text.

Each decision also carries a `systemMessage` naming it, starting with vSR's mark (`▚▞ vSR`); this
showed up in Copilot CLI's own debug log in the probe behind
https://github.com/vllm-project/semantic-router/issues/7326, not in `-p` output or the session's
own event log, so treat it as a debugging aid rather than something every surface shows.

A decision runtime that does not answer never stops Copilot CLI: the call goes on as it would
without vSR, and the `systemMessage` says what went unchecked (`▚▞ vSR guard skipped · the
decision model is not answering`).

## How it differs from Codex and Claude Code

- **No Stop hook.** Codex ends each turn with one line naming the models a Router gateway served
  it with, read from the turn id Codex sends with every call. Copilot CLI's `Stop` payload has no
  such id, so this plugin has nothing to build that line from and registers no `Stop` hook.
- **Flat reply fields, not nested `hookSpecificOutput`.** Claude Code and Codex nest a
  `PostToolUse` reply's extra context under `hookSpecificOutput.additionalContext`. Copilot CLI's
  own documented schema puts `additionalContext` and `modifiedResult` at the top level, and a
  probe behind https://github.com/vllm-project/semantic-router/issues/7326 found the nested form
  silently dropped for this event, so this plugin uses only the flat, documented fields.
- **redact reads shell output only**, the way the Codex plugin does: Copilot CLI reads files
  through shell commands, so every read goes through it, but it does not hide secrets by pattern
  the way the Claude Code plugin does, and it reads the first 16,000 characters of an output;
  past them the `systemMessage` says the rest went unchecked.
- **Not in this plugin yet:** router, escalate and check, the end-of-turn line, and
  `/vsr:setup`/`/vsr:calibrate` (install the Claude Code plugin alongside for those skills).
