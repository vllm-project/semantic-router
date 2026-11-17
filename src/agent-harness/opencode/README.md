# vSR: vLLM Semantic Router in OpenCode

vSR gives OpenCode's subagents a lighter model when their task is easy, holds
risky shell commands for you to confirm, and hides personal data and secrets
before the model reads them. A small decision model from the vLLM Semantic
Router project (vela3-lite, about 2.3 GB of RAM) makes these calls on your
machine or on a runtime you host; your requests still go to your own provider.
It does six things, the same six as the Claude Code and Codex plugins:

| App      | What it does                                                                                                       |
| -------- | ------------------------------------------------------------------------------------------------------------------ |
| router   | Gives each new subagent without a model of its own the model and effort, from your three tiers, that fit its task. |
| escalate | When you push back on an answer, the agent re-checks it and the next subagent goes one step up.                    |
| redact   | Hides personal data and secrets in `read` and shell output before the model reads them.                            |
| shield   | Warns the model when a file or web page tries to give it instructions.                                             |
| guard    | Holds a risky shell command for you to confirm in OpenCode's own permission prompt.                                |
| check    | Marks a final answer that is not backed by what the model read that turn. Off by default.                          |

## Install

You need OpenCode 2.2.0 or later and Node 22. vSR is not on npm yet, so point
OpenCode at this folder of a checkout. From this folder:

```bash
npm ci
```

Then add it to your global configuration, `~/.config/opencode/opencode.jsonc`
(or a project's `opencode.json`):

```json
{
  "plugins": [
    {
      "package": "/absolute/path/to/semantic-router/src/agent-harness/opencode",
      "options": {
        "routing": {
          "tiers": [
            { "model": "your-provider/light" },
            { "model": "your-provider/middle" },
            { "model": "your-provider/strong" }
          ]
        }
      }
    }
  ]
}
```

The tiers are three models of one provider in your OpenCode configuration,
lightest first, and your session runs on one of them: the router moves a
subagent between them only when it starts from one of the three. Without
`routing`, router and escalate stay off. OpenCode applies a saved change
within a few seconds; type `/reload` to apply it at once.

`opencode plugin list` lists only plugins installed from npm or Git, so it does
not show a plugin loaded from a folder.

## Set up the decision runtime

vSR needs a decision runtime: `vllm-srun serve` on this machine, or a Router
listener that publishes a System One model.

1. **No runtime yet?** Ask the agent to use the `vsr-setup` skill. It looks at
   the machine, installs `vllm-srun` pinned to the commit vSR was tested against
   in `~/.local/share/vsr/venv`, and starts vela3-lite on port 8790 on the GPU
   or CPU, for this login or as a user service. OpenCode asks before each
   command it runs. It is the same runtime the Claude Code plugin installs, so
   one runtime serves every agent on the machine. It runs on Linux and macOS;
   a Mac runs it on the CPU, since an Apple GPU cannot run vela3 yet. On Windows,
   connect to a runtime on another machine instead.
2. **Connect it.** Run `/connect` and pick **vSR decision runtime**: **Runtime
   without a token** for `http://127.0.0.1:8790/v1/decisions`, or **Runtime with
   a token** for one that needs a key. The form asks the `/v1/decisions` URL and,
   when the runtime serves several, the model. OpenCode keeps the token in its
   own credential store, and vSR switches to the connected runtime at once,
   without a restart. Without a connection, vSR uses the runtime in the
   [options](#options), by default `http://127.0.0.1:8790/v1/decisions`.
3. **Check it.** Run `/vsr-check` in a session. vSR lists the runtime's models and
   asks one case per enabled app through the path that app takes (a force push
   for guard, a support ticket for redact, a page with hidden instructions for
   shield, an easy task for router, pushback for escalate, an invented summary
   for check). Each line says whether the app decides it as it does on
   vela3-lite and how long it took, with the median. An app with no answer means
   the runtime is offline, refuses the token, or is slower than the app's wait.

## What you see

While the agent runs, the line above the prompt shows the newest decision, so
it sits under the step it was made on:
`vSR  REDACTED ▒ read …/customer.txt  email ×1 · phone ×1  · 9 decisions · /vsr`.
It names what redact hid, the model the router picked, or the score against its
threshold, and counts in red the decisions to look at. The sidebar (OpenCode's
"Show sidebar", `<leader>b`) lists the session's decisions in order, one line
each. A held command is OpenCode's own permission prompt, with the reason vSR
held it. OpenCode draws its transcript itself, so vSR cannot add lines under
each step there.

The decisions panel, `/vsr`, is for looking back:

- At the top: whether the runtime answers, which apps are on, each app's wait,
  the router's goal, when you last calibrated, and the tokens and estimated cost
  of the session's subagents (from OpenCode's prices; a model with no known
  price is counted apart).
- Then each decision, newest first: its outcome in color (red needs a look,
  amber went through only in part, quiet let things be), what it was about (the
  command, the subagent's task, your prompt, or the file, page or command an
  output came from), how long it took, and its score as a bar with the
  thresholds in force when it was made marked on it. The router's score is a
  difficulty level from 0 to 2, marked with its light and strong lines.
  - `routed`: vSR picked the subagent's model and effort from your tiers.
    `→` names the model vSR chose, then `ran`, or the model the subagent
    actually ran on when it differs.
  - `kept`: the subagent stays on its model: the router was not sure enough, the
    goal is `quality`, the model is not one of your tiers, or the agent has a
    model of its own (`agent model`).
  - `escalated`: you pushed back; the agent re-checks and the next subagent
    goes one step up.
  - `held`, then `approved` or `declined`: a shell command waited for your
    answer, and what you answered.
  - `redacted`: personal data or secrets were hidden from an output.
  - `shielded`: a file or page tried to instruct the model, which was told to
    treat it as data.
  - `unverified`: the answer claims things that what the model read does not
    back. Look twice. `verified` when it does.
  - `partial`: an output was checked only in part, and the rest went to the
    model unchecked.
  - `withheld`: an app set to wait until done could not finish, so the model
    got nothing from that output.
  - `skipped`: an app got no answer in time; the model went on without it.
  - `allowed`: a command guard let run.

Every control is a command; type `/vsr` to list them. The panel takes the
keyboard when it opens, and `Esc` closes it. To keep the panel open and type,
press `ctrl+x` then `←` to return to the prompt; there `Esc` interrupts the
model as usual, and `/vsr` closes the panel.

| Command          | What it does                                                           |
| ---------------- | ---------------------------------------------------------------------- |
| `/vsr`           | Open the decisions panel, or close it                                  |
| `/vsr-apps`      | Turn an app on or off, or hand the apps back to a Router listener      |
| `/vsr-waits`     | Choose how long redact, shield and guard wait: 1 to 3 s, or until done |
| `/vsr-feedback`  | Say whether a decision was right; for router, whether the model fit    |
| `/vsr-check`     | Check the decision runtime: one case per enabled app                   |
| `/vsr-sessions`  | Show another session's decisions                                       |
| `/vsr-goal`      | Choose the router's goal: balanced, quality or save                    |
| `/vsr-refresh`   | Refresh the panel                                                      |
| `/vsr-calibrate` | Fit an app's thresholds to your answers                                |

What you change in `/vsr-apps`, `/vsr-waits` and `/vsr-goal` is saved and applies to every session,
also after a restart. Changing the same setting in the options later wins over
it.

When the runtime cannot be reached, the panel says `offline`, and vSR steps
aside: OpenCode works as it would without vSR, except that guard's fixed rules
still hold, and vSR tries again a minute later. An app that waits until done
does not step aside: redact and shield withhold what they cannot check, and
guard holds every command.

## How the router decides

For each new subagent without a model of its own, `general` and Explore alike,
vSR asks how difficult the task is and how many reasoning steps it needs. In
OpenCode such a subagent otherwise runs on your session's model. The Claude
Code plugin routes only general-purpose subagents and leaves Explore on the
model its definition picks. An agent you gave a model keeps it, recorded as
`kept`.

- Easy tasks go to your light tier, hard ones to your strong tier, the rest to
  the middle. With `efforts` on a tier, the effort follows the steps answer.
- A tier lighter than your session's is used only when the decision model is
  sure (confidence 0.6 or more); below 0.5 the subagent keeps your session's
  model. `routing.goal` changes this: `quality` never uses a lighter tier,
  `balanced` is the default, `save` uses one whenever the answer is clear enough
  to route at all.
- When you push back on an answer (escalate), the next subagent goes one tier
  up, never below your session's, and one effort higher.

vSR never changes your session's own model, and picks a subagent's model only
when the subagent starts. A subagent that names a model, a resumed one, an
agent with its own model, and a session on a model outside your tiers keep
their choice.

## Guard

Before a shell command runs, vSR splits it into the commands it runs and asks
about each in turn whether it needs a person's review. When a part scores 0.9
or more, OpenCode's permission prompt asks you, with the reason. All parts share
one wait (3 s by default); a part it could not check in that time lets the
command run, recorded as `skipped`. Waiting until done (up to 60 s) holds a
command it could not check instead. A command of more than 12 parts is always
held. Guard only tightens: your own deny and ask rules in OpenCode still apply.

Some commands are always held, whatever the score and even while the runtime is
offline: recursive deletes of `/`, your home directory or a top-level system
directory, `mkfs`, `dd` to a device, fork bombs, recursive permission changes
on `/`, and scripts from `curl` or `wget` fed to a shell or an interpreter.
These rules see through pipes, subshells, `sudo`, `env`, `nohup` and similar
wrappers, and `bash -c` strings. They are the Claude Code plugin's rules.

Guard covers OpenCode's `shell` tool; commands run by other tools, or typed in
your own terminal, are outside it.

## Redact and shield

Redact hides names, emails, phone numbers, card numbers, addresses, ID numbers
and secrets (API keys, tokens, private keys, passwords in config files and
connection URLs) in `read` and shell output, including a failed command's
output. Secrets in known formats are hidden by fixed rules before anything else
sees the text, the decision model included; the rest is the decision model's
call. Dates, domain names and version numbers stay. Each hidden value becomes a
tag such as `[vsr:email]`, and the model is told the tags are vSR's and the
values hidden on purpose. If the agent then tries to write a tag into a file as
if it were the real value, vSR refuses the write and asks it to change only the
lines it needs.

Shield reads what `read` and `webfetch` return. When a file or page tries to
give the agent instructions, the model gets it quoted as data with a warning.

Both wait for the decision model, redact 3 s and shield 2 s by default; the
part of a long output they could not read in that time goes on with a note
(`partial`). What happens when a check cannot finish:

| When                                          | Waiting 1 to 3 s                                                     | Waiting until done     |
| --------------------------------------------- | -------------------------------------------------------------------- | ---------------------- |
| The runtime is slow, offline or errs          | The fixed rules still apply; the rest goes on, marked unchecked      | The output is withheld |
| An output is longer than 128,000 characters   | The decision model reads the first 128,000; the fixed rules read all | The output is withheld |
| An output is longer than 1,000,000 characters | The output is withheld                                               | The output is withheld |
| An output carries images                      | They go on, marked unchecked                                         | The output is withheld |

## Check

Check is off by default because it is slow without a GPU: it reads the answer
and what the model read that turn, each up to 8,000 characters, and a CPU
runtime takes several seconds for a long answer, past check's 1.5 s limit. Past
the limit it steps aside (`skipped`); it never holds the agent up and never
changes the answer. Turn it on with `/vsr-apps` or `apps.check` when the runtime has a
GPU.

It checks the main session's final answer against the `read` and shell output
the model received during that turn, after redact and shield. A score of 0.5 or
more marks the answer `unverified`. `verified` means backed by what the model
read, not that it is true. Short answers (under 40 characters) and turns that
read nothing are not checked.

## Calibrate from your own answers

The thresholds above are measured on vela3-lite. To fit them to your own work,
vSR keeps the newest 5,000 decisions it can learn from, across sessions and
restarts, in OpenCode's plugin storage: each with its subject, the scores it
held against its thresholds and the thresholds in use then. A rule's
decision, a skipped or withheld one, and a route outside your tiers are not
kept. Run `/vsr-calibrate` and pick an app:

1. It asks about the unanswered decisions nearest the app's line, on both sides:
   "Should vSR have asked you before this command ran?" with what vSR did and
   how near the line it was; for router, whether the routed model was too weak,
   right or stronger than needed. Skip any you are unsure of, or stop.
2. For apps other than router, pick what matters more: balanced, fewer false
   alarms or catch more.
3. It shows the new lines, and how many of your answers the current and the new
   ones agree with. Apply, or keep the current ones.

Some answers come without asking: `/vsr-feedback` on a decision, a held command you
declined (it should have asked), and pushback within 15 minutes of a route to
the lighter two tiers (it needed a stronger one). An app needs 8 answers before
its lines move, and each line moves at most 0.15 per calibration, the same fit
as the Claude Code plugin's `/vsr:calibrate`. Escalate only logs when it
escalates, so its lines only rise.

Calibrated thresholds apply at once and after a restart. Changing a threshold
in the options drops the calibration in favor of what you set. The panel's
Calibration line says when you last calibrated and how many decisions came
since, and suggests `/vsr-calibrate` after 300.

## A Router listener's switches and rules

Pointed at a Router listener that publishes a System One model, vSR reads the
listener's listing (`GET /v1/systemone/models`) at the first prompt, and again
at a prompt a minute or more later. Its `vsr` profile (`apis.systemone.vsr` in
the Router's config, or Default, On or Off in the Dashboard under Config ›
Global Config › vSR agent plugins) turns each app on or off for every plugin
pointed at it. An app you name under `apps` in the options, or turn on or off
with `/vsr-apps`, stays yours; the listener sets the others. "Follow the Router
listener" in `/vsr-apps` hands the apps you changed there back to the listener. A plain
decision runtime sets nothing.

Rules the listener binds to guard and redact (`vsr.rules`) come with the same
listing, and the listener checks them beside the decision model's questions,
within the same wait. A rule that matches a command holds it, with the rule's
description; a rule's find in an output is hidden as `[vsr:<rule name>]`. A
`required` rule the listener could not check holds the command or withholds the
output, whatever the wait; an advisory one is recorded as skipped. vSR keeps the
last rules it read in plugin storage, so an outage, or a restart during one,
never drops a required rule.

## Privacy

With a local runtime nothing leaves your machine for vSR's decisions. A remote
runtime receives everything the enabled apps check: each `read` and shell
output (after the fixed secret rules, when redact is on), fetched page, prompt,
subagent task, shell command and answer.

vSR keeps no output. Of a decision it keeps a subject: the command, subagent
task, prompt, or for an output its source (file, address or command), on one
line, at most 200 characters, with secrets in known formats hidden. The
session's decisions are kept in OpenCode's plugin storage (500 per session, 64
sessions), so the panel and sidebar still show them after `/reload` or a
restart; deleting a session deletes them. The calibration log keeps its 5,000
there too. A token is never shown in the
panel or returned by the plugin.

## How it differs from Claude Code

- **Your session's model never changes.** The Claude Code plugin can also route
  your own turns; here the router picks subagents' models only.
- **Your tiers, your provider.** Instead of Haiku, Sonnet and Opus, the router
  uses the three models you list, and only when your session runs on one of
  them.
- **One panel instead of cards.** Decisions are in the `/vsr` panel and the line
  above the prompt, not under each row of the conversation.
- **OpenCode's own setup.** The runtime is connected through `/connect`, its
  token stays in OpenCode's credential store, and options live in
  `opencode.jsonc` instead of `/config`.
- **No savings estimate.** The panel shows what subagents actually used, and
  OpenCode's own cost estimate for it.

## Options

Every option is optional. An unknown option is an error, not ignored.

| Option                                                 | Default                              | What it sets                                                                   |
| ------------------------------------------------------ | ------------------------------------ | ------------------------------------------------------------------------------ |
| `runtimeUrl`                                           | `http://127.0.0.1:8790/v1/decisions` | The runtime, when none is connected through `/connect`                         |
| `runtimeModel`                                         | the runtime's only model             | One model of a runtime that serves several                                     |
| `runtimeTokenEnv`                                      | `VSR_RUNTIME_TOKEN`                  | The environment variable of the OpenCode server that holds the runtime's token |
| `apps`                                                 | all on except `check`                | Each app on or off, for example `{ "check": true }`                            |
| `waits.redact`, `waits.shield`, `waits.guard`          | `3000`, `2000`, `3000`               | How long each waits: `1000`, `2000`, `3000` milliseconds, or `"done"`          |
| `routing.tiers`                                        | none                                 | Three `{ "model": "provider/model" }`, lightest first, with optional `efforts` |
| `routing.goal`                                         | `"balanced"`                         | `"quality"` never routes lighter, `"save"` routes lighter whenever it can      |
| `routing.easyBelow`, `routing.hardFrom`                | `0.65`, `1.5`                        | The difficulty (0 to 2) under which a task is easy, and from which it is hard  |
| `routing.minConfidence`, `routing.downgradeConfidence` | `0.5`, `0.6`                         | The confidence to route at all, and to route lighter than your session         |
| `routing.feedbackThreshold`, `routing.repeatThreshold` | `0.7`, `0.8`                         | Escalate's lines for "that's wrong" and "again"                                |
| `routing.deadlineMs`                                   | `1500`                               | The router's wait                                                              |
| `guard.reviewThreshold`                                | `0.9`                                | The score that holds a command                                                 |
| `redact.piiThreshold`                                  | `0.7`                                | The score that hides a span of personal data                                   |
| `shield.injectionThreshold`                            | `0.9`                                | The score that marks an output as instructions                                 |
| `check.unsupportedThreshold`, `check.deadlineMs`       | `0.5`, `1500`                        | The score that marks an answer unverified, and check's wait (at most 10000)    |
| `maxDecisions`, `maxSessions`                          | `500`, `64`                          | How many decisions per session, and sessions, the panel keeps                  |

Tokens, URL query tokens and passwords in a URL are refused as options; use
`/connect` or `runtimeTokenEnv`. vSR does not follow a runtime's redirects, so a
token cannot be sent to another host.

## Troubleshooting

- **`/vsr-check` says an app has no answer.** The runtime is not answering at the
  connected URL, refuses the token, or is slower than the app's wait. Check
  that `vllm-srun serve` is running, or ask for `vsr-setup` again.
- **A configuration change has no effect.** Type `/reload`. If it still has
  none, run `opencode service restart`.
- **Reads feel slower.** On a CPU runtime, redact and shield add up to their
  waits to a long output, and guard a fraction of a second per command part. A
  GPU runtime is several times faster; you can also shorten a wait (`/vsr-waits`) or
  turn off apps you do not need (`/vsr-apps`).
- **A subagent stayed on its model.** The panel's `kept` row says why: the
  router was not sure enough, `routing.goal` is `quality`, or your session's
  model is not one of your tiers.

## Uninstall

1. Take the plugin out of your OpenCode configuration and save it.
2. Remove the credential in `/connect`, if you connected a runtime.
3. If `vsr-setup` installed a runtime and the Claude Code plugin does not use
   it, stop it and delete `~/.local/share/vsr`:

   ```bash
   # Started for one login:
   kill "$(lsof -ti tcp:8790)"
   # Started as a Linux user service:
   systemctl --user disable --now vsr-runtime
   rm -rf ~/.local/share/vsr
   ```

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for the checks, the tests and how each app
works in detail.
