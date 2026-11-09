---
slug: vsr-claude-code
title: "vSR: vLLM Semantic Router Inside Claude Code"
description: "vSR brings vLLM Semantic Router's decision model into Claude Code. It routes subagents to the right model, hides personal data and secrets before Claude reads them, and holds risky Bash commands for you to confirm."
authors: [theohsiung, aayushsaini101, abhinav]
tags: [claude-code, agents, routing, safety, privacy, vela, semantic-router, community]
image: /img/blog/vsr-claude-code/05-vsr-pane.png
---

import { ArticleFigure } from '@site/src/components/ArticleMedia';

{/* DRAFT SKELETON. Each section has a note on what it covers and the facts it can use.
    Measured numbers stay as ⟦placeholders⟧ until a real run fills them. Pick up any section in the PR. */}

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

{/* 2. ARCHITECTURE. Claude Code hooks (subagent start, before and after a tool, prompt submit,
    turn end) send a typed question to vela3-lite over /v1/decisions on a vllm-srun runtime.
    The answer is a score plus labels. vSR turns it into a decision and draws it under the row
    it is about. Tie back to the router: the same signal model family that feeds routing
    decisions in vLLM Semantic Router, here asked one question per hook.
    FIGURE (architecture): hand-drawn whiteboard sketch, 1672x941, navy ink, pastel hatching.
    Left zone: a Claude Code terminal window with an Agent row, a Read row and a Bash row.
    Centre hub: a "vSR hooks" window with six small app badges (router, escalate, redact,
    shield, guard, check) and circuit lines into a "vela3-lite · vllm-srun" box with mini
    dashboards for difficulty, PII spans, injection score and review risk. Right zone: the
    outcomes as stamped cards (ROUTED, REDACTED, SHIELDED, HELD). Bottom strip: small tool
    windows for "localhost:8790" and "remote GPU runtime". */}

- ⟦Hook points and the question each one asks.⟧
- ⟦One model, several typed questions: how this maps to vLLM Semantic Router signals.⟧
- ⟦Runs on your machine (about 2.3 GB of RAM on CPU, about 2 GB of GPU memory) or on a runtime you host.⟧

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

{/* 3a. Router asks two questions per subagent: difficulty and reasoning steps. Easy goes to
    Haiku, multi-step to Sonnet, hard to Opus; effort follows the steps answer. A smaller model
    is used only at confidence 0.6 or more; routerGoal quality / balanced / save changes that.
    Escalate: pushback moves the next subagent one step up, never below the session's model.
    Use a real routing cost row from a measured session: ⟦routing row⟧. */}

<ArticleFigure
  src="/img/blog/vsr-claude-code/01-routed-and-kept.png"
  width={1280} height={882}
  alt="Two subagents in Claude Code: one KEPT on Opus 5.5 because the decision model was not sure enough, one ROUTED to Opus 5.5 at high effort as a hard task."
>
  **KEPT**: the decision model was not sure enough to give "refactor login validation" a smaller model, so it stays on the session's model. **ROUTED**: the constant-time password check is a hard task, so it runs on Opus at high effort.
</ArticleFigure>

<ArticleFigure
  src="/img/blog/vsr-claude-code/02-escalated.png"
  width={1280} height={882}
  alt="The user says task 2 was wrong; vSR shows ESCALATED and the next subagent goes one step up."
>
  **ESCALATED**: the user said task 2 was wrong. Claude is told to re-check its last answer, and the next subagent goes one step up.
</ArticleFigure>

- ⟦Router: the two questions and the confidence rule, in prose.⟧
- ⟦routerGoal: quality, balanced, save.⟧
- ⟦Measured: the routing cost row from a real session, ⟦$ routed⟧ vs ⟦$ on session model⟧.⟧

### Protect: redact, shield and guard

{/* 3b. Redact: personal data and secrets replaced with tags such as [email] before Claude reads
    Read or Bash output. Shield: a file or page that tries to instruct the agent; Claude is told
    to treat it as data. Guard: a command is split into the commands it runs (&&, ||, ;, &, new
    lines; quotes, $(...), subshells and heredocs respected); each part is scored, and 0.9 or
    more opens Claude Code's permission dialog with the reason. Fixed rules (rm -rf of / or ~,
    mkfs, dd to a device, fork bombs, curl | sh and friends) hold even when the model is offline,
    and see through bash -c, sudo, env, pipes and heredocs. */}

<ArticleFigure
  src="/img/blog/vsr-claude-code/03-redacted-and-shielded.png"
  width={1280} height={882}
  alt="REDACTED: a support ticket's name, email, phone and card number replaced with tags. SHIELDED: an install note hiding an instruction to AI agents in an HTML comment."
>
  **REDACTED**: before Claude read the support ticket, a name, an email, a phone number and a card number were replaced with tags such as `[email]`. **SHIELDED**: the install note hid an instruction to AI agents in an HTML comment (injection score 0.99). Claude treated the file as data.
</ArticleFigure>

<ArticleFigure
  src="/img/blog/vsr-claude-code/04-held-declined-and-band.png"
  width={1280} height={882}
  alt="DECLINED: a git reset --hard followed by a force push scored 0.95 for review, was held in Claude Code's permission dialog, and the user said no."
>
  **DECLINED**: `git reset --hard` followed by a force push scored 0.95 for review (the line is 0.90). Guard held it in Claude Code's permission dialog, and the user said no. The read-only `git show` checks before it ran without a prompt.
</ArticleFigure>

- ⟦Redact: what it hides, in prose, with one before/after line.⟧
- ⟦Shield: the hidden-instruction example.⟧
- ⟦Guard: parts, scores and the fixed rules.⟧

### Trust: check

{/* 3c. Check reads up to 8,000 characters of what Claude read and 8,000 of its answer at turn
    end, and marks an answer that is not backed (UNVERIFIED). Off by default because it is slow
    on CPU; turn it on with a GPU runtime. Never holds Claude up: past its limit it steps aside. */}

⟦Check: what UNVERIFIED means and when to turn it on.⟧

## Design choices

{/* 4. The rules that make it safe to leave on:
    - Never changes the main conversation's model or effort; picks a subagent's model only when
      it starts, so no prompt cache is lost. Forks, teammates and named-model calls are left alone.
    - Only tightens: never allows what Claude Code would ask about or refuse.
    - Offline: the tally says "model offline", a toast says so once, and vSR steps aside; guard's
      fixed rules still hold, and it retries every minute.
    - Waits: redact, shield and guard each wait 1, 2 or 3 s, or "until done" (up to 60 s), which
      withholds or holds what it could not check instead of stepping aside. */}

- ⟦Your main conversation is never touched.⟧
- ⟦It only tightens.⟧
- ⟦Step aside or wait until done: the trade-off, in one paragraph.⟧

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

## What you see

{/* 5. Cards only when something needs your eyes (HELD, DECLINED/APPROVED, SHIELDED, UNVERIFIED,
    WITHHELD); one dim line when vSR changed something (ROUTED, KEPT, REDACTED, ESCALATED,
    SKIPPED); nothing when all went as it would have without vSR. The tally above the prompt.
    The /vsr pane: app toggles (keys 1 to 6), wait settings, decisions by app, and right call /
    wrong call buttons. /vsr:calibrate fits the thresholds to those answers. */}

- ⟦Cards, lines and the tally.⟧
- ⟦The `/vsr` pane and its controls (see the first figure).⟧
- ⟦Feedback and `/vsr:calibrate`: the thresholds learn from your answers.⟧

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

{/* 7. Where it goes from here: calibration from feedback across users, more apps on the same
    decision model, remote runtimes for teams. Link the plugin README and the issue tracker. */}

⟦What's next, and where to send feedback.⟧

**[Plugin README](https://github.com/vllm-project/semantic-router/tree/main/src/claude-code) · [Report an issue](https://github.com/vllm-project/semantic-router/issues)**
