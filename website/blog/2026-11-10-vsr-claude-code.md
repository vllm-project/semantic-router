---
slug: vsr-claude-code
title: "vSR: vLLM Semantic Router Inside Claude Code"
description: "vSR brings vLLM Semantic Router's decision model into Claude Code. It routes subagents to the right model, hides personal data and secrets before Claude reads them, and holds risky Bash commands for you to confirm."
authors: [theohsiung, abhinav]
tags: [claude-code, agents, routing, safety, privacy, vela, semantic-router, community]
image: /img/blog/vsr-claude-code/05-vsr-pane.png
---

import { ArticleFigure } from '@site/src/components/ArticleMedia';

A coding agent does three things all day: it hands work to subagents, it reads files, and it runs commands. Each one carries a cost that the session never shows you.

**Subagents run at your model's price.** A general-purpose subagent starts on your session's model. Listing files, counting lines or renaming a variable costs the same as a hard design question, because nothing asks how hard the task is before the subagent starts.

**Reads carry more than code.** The agent reads what it is pointed at: a support ticket with a customer's email and card number, a config file with an API key, a vendored README with an instruction to AI agents hidden in a comment. Whatever it reads becomes part of the conversation, and what a file says can steer what the agent does next.

**Commands blur together.** A long session runs dozens of Bash commands, and almost all of them are harmless. A `git reset --hard` chained with a force push is not, yet in a long run of prompts, or in auto mode with none at all, it looks like the rest.

Each of these is a routing question: which model should take this task, what in this output should the model never see, which command needs a person. vLLM Semantic Router answers questions like these for every request at the gateway. vSR asks them inside the coding agent, at the moment each one happens.

**vSR** is a Claude Code plugin that puts vLLM Semantic Router's decision model inside the harness. One small model, vela3-lite, answers the questions a router would ask: how hard is this task, is there personal data in this output, is this file trying to instruct the agent, does this command need a person. Your requests still go to Claude as usual.

<!-- truncate -->

<ArticleFigure
  src="/img/blog/vsr-claude-code/05-vsr-pane.png"
  width={1280} height={882}
  alt="Claude Code with the vSR pane open: a declined git reset, an approved branch delete, and the pane's app toggles, wait settings and per-app decisions."
>
  A Claude Code session with vSR. Guard held a destructive reset that the user declined. The `/vsr` pane on the right lists every decision of the session, grouped by app.
</ArticleFigure>

## How vSR works

vSR is a set of Claude Code hooks around one call. Wherever Claude Code is about to do something worth a second look, a hook sends the text involved to the decision model with a typed question, then turns the answer into a decision and draws it under the row it is about.

| When | App | The question |
| --- | --- | --- |
| A general-purpose subagent starts | router | How difficult is this task, and how many reasoning steps does it need? |
| You send a prompt | escalate | Is this pushback on the last answer? |
| A Bash command is about to run | guard | Does each part of this command need a person's review? |
| A `Read`, Bash or `WebFetch` result comes back | redact, shield | Where is the personal data or the secret? Does this text try to instruct the agent? |
| A turn ends | check | Is the answer backed by what Claude read? |

The call itself is one HTTP request. vSR posts the text and its questions to the runtime's `/v1/decisions` endpoint, and gets back one answer per question: probabilities for a choice, and character spans for personal data. vela3-lite comes from Vela, the family of routing models behind vLLM Semantic Router's signals, so one model of about 2.3 GB answers the questions of all six apps.

The runtime is `vllm-srun`, the project's model runtime. It runs on your machine, on CPU or GPU, or on a server you host for a team. vSR does not sit between you and Anthropic: your prompts and Claude's replies travel exactly as they did before, and vSR adds its checks around them.

{/* FIGURE TO DRAW (architecture): hand-drawn whiteboard sketch, 1672x941, navy ink, pastel hatching.
    Left zone: a Claude Code terminal window with an Agent row, a Read row and a Bash row.
    Centre hub: a "vSR hooks" window with six small app badges (router, escalate, redact,
    shield, guard, check) and circuit lines into a "vela3-lite · vllm-srun" box with mini
    dashboards for difficulty, PII spans, injection score and review risk. Right zone: the
    outcomes as stamped cards (ROUTED, REDACTED, SHIELDED, HELD). Bottom strip: small tool
    windows for "localhost:8790" and "remote GPU runtime". */}

## Six apps

vSR is six small apps on one decision model. You can turn each one on or off.

| Group | App | What it does |
| --- | --- | --- |
| Spend less | router | Gives each general-purpose subagent a model (Haiku, Sonnet or Opus) and an effort level that fit its task. |
| Spend less | escalate | When you push back on an answer, Claude re-checks it and the next subagent goes one step up. |
| Protect | redact | Hides personal data and secrets in `Read` and Bash output before Claude reads them. |
| Protect | shield | Warns Claude when a file or web page tries to give it instructions. |
| Protect | guard | Holds a risky Bash command for you to confirm, part by part. |
| Trust | check | Marks an answer that is not backed by what Claude read that turn. Off by default. |

### Spend less: router and escalate

For each general-purpose subagent without a model of its own, the router asks two questions: how difficult the task is, and how many reasoning steps it needs. Easy tasks can go to Haiku, multi-step tasks to Sonnet and hard ones to Opus, and the effort level follows the steps answer (Haiku always runs at medium effort).

A smaller model than your session's is used only when the decision model is sure, at a confidence of 0.6 or more. Below that, the subagent keeps your model and the line says `KEPT`. The `routerGoal` setting moves that bar: `quality` never picks a smaller model, `balanced` is the default, and `save` routes whenever the answer is clear enough to route at all.

<ArticleFigure
  src="/img/blog/vsr-claude-code/01-routed-and-kept.png"
  width={1280} height={882}
  alt="Two subagents in Claude Code: one KEPT on Opus 5.5 because the decision model was not sure enough, one ROUTED to Opus 5.5 at high effort as a hard task."
>
  **KEPT**: the decision model was not sure enough to give "refactor login validation" a smaller model, so it stays on the session's model. **ROUTED**: the constant-time password check is a hard task, so it runs on Opus at high effort.
</ArticleFigure>

In one session on Opus 5.5, a subagent asked to count the lines of a file went to Haiku 5.5 at medium effort. vSR priced that turn at under $0.01, against $0.17 had it run on Opus 5.5: 97% less, at list price. In the same session the router kept "Explain calc.py briefly" on Opus 5.5, because its confidence (0.33) was under the 0.60 line.

<ArticleFigure
  src="/img/blog/vsr-claude-code/10-routed-haiku-cost.png"
  width={1912} height={450}
  alt="Left: a subagent counting the lines of calc.py is ROUTED to Haiku 5.5 at medium effort as an easy task. Right: the /vsr pane with a KEPT decision at confidence 0.33 and the routing row: under $0.01 for 1 subagent turn vs $0.17 on Opus 5.5, saved 97%."
>
  **ROUTED**: an easy subagent task runs on Haiku 5.5. The `routing` row in `/vsr` prices it against the session's model: under $0.01 instead of $0.17. **KEPT**: a task the router was not sure about stays on Opus 5.5.
</ArticleFigure>

Escalate covers the other direction. When you push back on an answer ("that's wrong, check again"), it tells Claude to re-check its last answer, and the next subagent goes one step up: never below your session's model, and one effort level higher.

<ArticleFigure
  src="/img/blog/vsr-claude-code/02-escalated.png"
  width={1280} height={882}
  alt="The user says task 2 was wrong; vSR shows ESCALATED and the next subagent goes one step up."
>
  **ESCALATED**: the user said task 2 was wrong. Claude is told to re-check its last answer, and the next subagent goes one step up.
</ArticleFigure>

### Protect: redact, shield and guard

**Redact** reads `Read` and Bash output before Claude does. The decision model marks names, emails, phone numbers, card numbers, IP addresses, street addresses and ID numbers as character spans, and fixed rules catch secrets such as API keys, tokens, private keys and passwords in config files. Each one is replaced with a tag such as `[vsr:email]`. The output of a failed command is redacted too, since Claude reads its error. If Claude later tries to write a tag back into a file in place of the real value, vSR refuses the write and tells Claude to change only the lines it needs.

**Shield** reads what comes back from `Read` and `WebFetch` and asks whether the text tries to give an AI agent instructions. At 0.90 or more, Claude is told to treat that output as untrusted data and not to follow instructions in it. Shield warns; it does not block, so Claude can still use the file's real content.

<ArticleFigure
  src="/img/blog/vsr-claude-code/03-redacted-and-shielded.png"
  width={1280} height={882}
  alt="REDACTED: a support ticket's name, email, phone and card number replaced with tags. SHIELDED: an install note hiding an instruction to AI agents in an HTML comment."
>
  **REDACTED**: before Claude read the support ticket, a name, an email, a phone number and a card number were replaced with tags such as `[vsr:email]`. **SHIELDED**: the install note hid an instruction to AI agents in an HTML comment (injection score 0.99). Claude treated the file as data.
</ArticleFigure>

**Guard** runs before a Bash command. It splits the command into the commands it actually runs, at `&&`, `||`, `;`, `&` and new lines, while respecting pipes, quotes, `$(...)`, subshells and heredocs. It then scores each part for whether it needs a person's review. A part at 0.90 or more opens Claude Code's permission dialog with the reason, and a command of more than 12 parts is always held.

Some commands should never run unattended, so fixed rules hold them whatever the score, even with the decision model offline: recursive deletes of `/`, your home directory or a top-level system directory, `mkfs`, `dd` to a device, fork bombs, and scripts from `curl` or `wget` run by a shell. The rules see through `bash -c`, `sudo`, `env`, pipes, subshells and heredocs.

<ArticleFigure
  src="/img/blog/vsr-claude-code/04-held-declined-and-band.png"
  width={1280} height={882}
  alt="DECLINED: a git reset --hard followed by a force push scored 0.95 for review, was held in Claude Code's permission dialog, and the user said no."
>
  **DECLINED**: `git reset --hard` followed by a force push scored 0.95 for review (the line is 0.90). Guard held it in Claude Code's permission dialog, and the user said no. The read-only `git show` checks before it ran without a prompt.
</ArticleFigure>

### Trust: check

At the end of a turn, check gives the decision model up to 8,000 characters of what Claude read and 8,000 of Claude's answer, and asks whether the answer is backed by those reads. An answer that claims things the reads do not support gets an `UNVERIFIED` card: a cue to look twice before acting on it.

Check is off by default. A CPU runtime needs 5 to 10 s to read a long answer, which is past check's 3 s limit, so on a CPU it would mostly step aside. Turn it on with a GPU or remote runtime. It never holds Claude up: past its limit, it steps aside.

## Design choices

**Your conversation is never re-routed.** vSR never changes the model or effort of your main conversation. It picks a subagent's model only when the subagent starts, so no conversation loses its prompt cache. Forks, teammates, subagents that name their own model and resumed subagents are left alone.

**It only tightens.** Guard can hold a command that Claude Code would have run, but it never allows one that Claude Code would ask about or refuse. Redact and shield change what Claude reads; neither grants Claude anything it did not already have.

**It says when it is not checking.** Redact, shield and guard each wait for the decision model: 1, 2 or 3 s, or `until done`. With a wait in seconds, a check that runs out of time steps aside. The action goes ahead, and a `SKIPPED` line or the tally says what went unchecked. With `until done`, the app waits up to 60 s, and if the check still cannot finish, redact and shield withhold the output and guard holds the command. When the decision model cannot be reached at all, the tally says `model offline`, a toast says so once, and vSR tries again every minute. Guard's fixed rules hold either way.

<ArticleFigure
  src="/img/blog/vsr-claude-code/09-tally-offline.png"
  width={920} height={60}
  alt="The vSR tally reading: model offline, guard, redact, shield not checking."
>
  Default waits with the decision model offline: guard, redact and shield step aside, and Claude carries on as it would without vSR.
</ArticleFigure>

<ArticleFigure
  src="/img/blog/vsr-claude-code/08-tally-redact-holding.png"
  width={1010} height={60}
  alt="The vSR tally reading: model offline, guard, shield not checking, redact holding."
>
  The same outage with redact set to `until done`: redact holds instead of stepping aside.
</ArticleFigure>

<ArticleFigure
  src="/img/blog/vsr-claude-code/07-withheld.png"
  width={800} height={120}
  alt="A WITHHELD card: notes.md withheld from Claude, not checked for personal data, because the decision model is offline."
>
  **WITHHELD**: redact could not check `notes.md`, so Claude got nothing from it rather than an unchecked file.
</ArticleFigure>

### Limitations

vSR narrows what reaches Claude and what runs. It is not a sandbox, and each app has edges worth knowing.

- **Redact is best effort.** The decision model finds names, emails, phone numbers, card numbers, addresses and ID numbers by reading them, and fixed rules catch common secret formats. Either can miss a value written in an unusual way. With the default 3 s wait on a CPU, redact reads about the first 8,000 characters of an output and marks the rest `PARTIAL`. Keep real secrets out of files the agent can read: redact is a second line of defence, not the first.
- **Shield will not catch every injection.** It flags text that reads as instructions to an AI agent, scored against a 0.90 line, and an instruction phrased as ordinary prose can score under it. Shield checks `Read` and `WebFetch` output, not Bash output, so text pulled in with a command such as `gh issue view` gets redact but not shield.
- **Guard's scores can be wrong in both directions.** A harmless command can be held, and a risky one can score under the line. The fixed rules are the only hard guarantee. Guard adds holds on top of Claude Code's own permissions, so keep those in place.
- **Default waits let unchecked work through.** When the decision model is slow or offline, redact, shield and guard step aside and say so, with a `SKIPPED` line or the `model offline` tally. Set a wait to `until done` where you would rather wait or withhold.
- **Routing savings are estimates.** The cost row prices the subagent turns vSR saw at list price. It is not your bill.
- **On a CPU runtime the router often runs out of time.** It waits a fixed 1.5 s, and on a laptop CPU the decision model often needs about that long, so a subagent can start on your session's model with a `SKIPPED` line. In the session above, one of three subagents was routed. Routing works best on a GPU or remote runtime.

## What you see

vSR stays quiet unless something needs you.

- **A card** appears when you should look: `HELD` (a command waits for your yes or no), then `DECLINED` or `APPROVED` with your answer, plus `SHIELDED`, `UNVERIFIED` and `WITHHELD`.
- **One dim line** appears when vSR changed something or decided on a subagent: `ROUTED`, `KEPT`, `REDACTED`, `ESCALATED`, `PARTIAL` and `SKIPPED`.
- **Nothing** is drawn when everything went as it would have without vSR, such as an allowed command. It is only counted.

A one-line tally above the prompt counts the session's decisions, and says `model offline` when the runtime stops answering. `display: quiet` shows only the cards, and `display: verbose` shows every decision with its scores.

`/vsr` opens a pane beside the conversation, as in the first figure. At the top are the controls: one toggle per app (keys 1 to 6) and the waits for redact, shield and guard. A change applies at once. Below them are the session's decisions grouped by app, the routing cost row, and the runtime and thresholds in use.

Each scored decision in the pane has buttons for your verdict: `right call` or `wrong call`, and for a routed subagent `right model`, `needed stronger` or `could be lighter`. Your answers are saved beside the decision log. After a few hundred decisions, `/vsr:calibrate` asks about the decisions closest to each threshold and fits the thresholds to your answers, moving each by at most 0.15 per run. The defaults are measured; calibration makes them fit how you work.

## Get started

You need Claude Code 2.1.319 or later and Python 3.

```bash
claude plugin marketplace add vllm-project/semantic-router
claude plugin install vsr@vllm-sr
```

Then, in a Claude Code session:

```text
/vsr:setup
```

Setup looks at the machine first and installs nothing until you choose. The recommended path picks a runtime for you: it keeps a runtime that already answers, or installs vela3-lite on this machine's GPU, or on its CPU when there is no GPU. It turns on five apps with measured defaults (check stays off), shows the settings it will write, and asks each app one question against the runtime before it finishes. Run `/reload-plugins`, then `/vsr`, and the pane shows the runtime, the apps and their waits.

### Local or remote

A **local runtime** keeps every decision on your machine. It installs into `~/.local/share/vsr` without sudo, and needs about 2.3 GB of RAM on CPU or about 2 GB of GPU memory. The model and its base, about 1.8 GB together, go to the Hugging Face cache. Setup checks the free disk before it installs, so a full disk stops it before the download instead of halfway through.

A **remote runtime** puts the decision model on a GPU you host, and one runtime can serve a whole team. Everything the apps classify then goes to that host: `Read` and Bash output (with secrets already hidden when redact is on), fetched pages, prompts, subagent tasks, Bash commands and, with check on, Claude's answers. Setup says so before it connects, warns when the URL is plain `http`, and keeps the runtime's token in Claude Code's secure storage, never in `settings.json`.

### Running on CPU

vela3-lite serves in bf16 by default. CPUs without native bf16 support emulate it, and on those machines every decision slows to the point where the apps step aside. Setup measures this instead of guessing: before it starts the runtime, it times a bf16 and a float32 matrix multiply on your CPU, and serves in float32 when bf16 is more than 1.5 times slower.

| | bf16 (emulated) | float32 |
| --- | --- | --- |
| Time per decision | 2 to 29 s | 0.7 to 2.5 s |
| Decided within the default waits | none | redact, shield, guard |
| Verdicts | same | same |

<small>Measured with vela3-lite on an Intel Core i5-1135G7 (no native bf16), CPU only, Linux, across two runs. float32 used no more memory than bf16.</small>

<ArticleFigure
  src="/img/blog/vsr-claude-code/06-setup-cpu-check.png"
  width={1900} height={414}
  alt="The end of /vsr:setup on a CPU without bf16: guard, escalate, shield, redact and router each give the expected verdict in 1.1 to 2.5 s, with a median of 1.5 s."
>
  The last step of `/vsr:setup` asks each app one question. On this CPU, in float32, every app gives the expected verdict, and setup says plainly which apps sit close to their waits.
</ArticleFigure>

On a CPU, each app's wait is the dial. Redact, shield and guard wait 3, 2 and 3 s by default, and step aside with a `SKIPPED` line when the model is slower than that. Set a wait to `until done` at the top of `/vsr` when you would rather wait than let anything through unchecked. Check stays off on CPU: it reads Claude's whole answer, and a long answer takes longer than its 3 s limit.

## What's next

vSR is the first integration under `src/agent-harness/`, the repository's home for coding-agent integrations. The decision model and its runtime do not depend on Claude Code, so the same checks can sit behind other agent harnesses next. Two improvements are already tracked: a `/vsr` pane that scrolls through every decision it keeps ([#6049](https://github.com/vllm-project/semantic-router/issues/6049)), and checks that recover on their own when a busy CPU stalls them, without a plugin reload ([#6050](https://github.com/vllm-project/semantic-router/issues/6050)).

The quickest way to make vSR better is to use it: answer `right call` or `wrong call` on the decisions it makes, run `/vsr:calibrate`, and tell us where it got things wrong.

**[Plugin README](https://github.com/vllm-project/semantic-router/tree/main/src/agent-harness/claude-code) · [Report an issue](https://github.com/vllm-project/semantic-router/issues)**
