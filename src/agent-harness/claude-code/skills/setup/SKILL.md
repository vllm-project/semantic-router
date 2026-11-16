---
name: setup
description: Guided setup for the vSR plugin. Runs the vLLM Semantic Router classifier on this machine's GPU or CPU or connects a remote runtime, picks which of vSR's six apps run, sets thresholds, writes the settings, and checks each app against the runtime. Use when vSR was just installed, when its tally says classifier offline or its rows say SKIPPED, or when the user wants to change vSR's runtime, apps, or thresholds.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/detect.py *) Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/runtime.py *) Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/settings.py id *) Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/settings.py show *) Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/settings.py plan *)
---

# Set up vSR

vSR asks a small classifier questions on every subagent, prompt, read and Bash command. This skill gets that classifier answering and writes vSR's settings. The scripts in `${CLAUDE_SKILL_DIR}/scripts/` do every check and every write; you run them, read their output, and ask the user only what they decide. Show script output as it prints, and keep your own summary to a few lines.

Settings go under `pluginConfigs["<plugin id>"].options` in the user's `~/.claude/settings.json` (or `$CLAUDE_CONFIG_DIR/settings.json`). Claude Code reads `pluginConfigs` only from user settings, so there is no project scope. The token for a remote runtime is the one value that never goes there: the user saves it with `/plugin configure <plugin id>`, which stores it in secure storage.

Apps and thresholds start at vSR's defaults. Every app is on by default except check: it reads Claude's whole answer, so on a CPU runtime a long answer outlasts its 3 s limit. Recommend check only with a GPU or remote runtime.

## Steps

1. **Plugin id.** Run `python3 ${CLAUDE_SKILL_DIR}/scripts/settings.py id --plugin-root "${CLAUDE_PLUGIN_ROOT}"`. It prints `vsr@<marketplace>` for an installed copy, or `vsr@inline` for one loaded with `--plugin-dir`. Then run `python3 ${CLAUDE_SKILL_DIR}/scripts/settings.py show --id <id>`. If `otherIds` lists another vSR id, tell the user those values stay under that id. Done when you know the id and what is already saved.

2. **Look first.** Run `python3 ${CLAUDE_SKILL_DIR}/scripts/detect.py --id <id>` and show its six lines (machine, GPU, Python, vllm-srun, runtime, suggestion). It installs nothing. When a runtime already answers, the runtime line names the device it runs on when the runtime reports one and how long one question takes, and the suggestion line carries `keep runtime=<value>`, the value that keeping it saves, with the reason in parentheses. Done when the user has seen what this machine has and what detect suggests.

3. **Recommended or custom.** Ask one AskUserQuestion:
   - **Recommended setup (Recommended)**: in the description, say what it does on this machine, from detect's suggestion: keep the runtime that already answers (`existing`), or install and run vela3-lite on this machine's GPU (`local-gpu`) or CPU (`local-cpu`). Apps and thresholds at their defaults (say check stays off and can be turned on later with a GPU or remote runtime). The user still sees the settings change before anything is written.
   - **Custom**: the user picks where the classifier runs, which apps run, and the thresholds.

   When detect suggests `remote`, this machine cannot run the classifier, so skip the question and go on as Custom. Done when the user picked one. For Recommended, take detect's suggestion as the step 5 branch, then go to step 8 with every app and threshold at its default. For Custom, go to step 4.

4. **Where the classifier runs.** Custom only. Ask one AskUserQuestion. Put detect's suggestion first, marked "(Recommended)", with its reason as the description:
   - **Keep the running one**, only when detect found a ready runtime: it stays as it is.
   - **Local GPU**, only when detect found a GPU: fastest, uses about 2 GB of GPU memory for vela3-lite.
   - **Local CPU**: about 2.3 GB of RAM, 0.1 to 0.25 s per question as measured on x86 CPUs with bf16 (setup times bf16 against float32 and runs a CPU without it in float32, 0.7 to 2.4 s on a 4-core laptop); an Apple Silicon CPU is several times slower, about 2 s, which is too slow for vSR's hooks.
   - **Remote runtime**: a runtime on another machine, reached by URL, with an optional token. Everything the apps classify goes to that host: each Read and Bash output (keys hidden by the secret rules when redact is on), each fetched page (shield), your prompts (escalate), each subagent's task (router), each Bash command (guard), and Claude's answers with what it read that turn (check).

   Done when the user picked one.

5. **Get the runtime answering.** Follow the branch the user picked, or detect's suggestion on the Recommended path:
   - **Keep the running one**: the URL is the one detect checked. Set `runtime` to the `keep runtime=` value on detect's suggestion line. When that line says detect could not tell the device, tell the user which value it kept and that Custom can change it. When the value is `remote`, tell the user what leaves this machine, as step 4 lists for a remote runtime, then run the Remote branch's `health` and probe commands and handle a `warning:` or an `HTTP 401` the way that branch does.
   - **Local GPU or Local CPU**: read [local-runtime.md](local-runtime.md) and follow it. On the Recommended path, use vela3-lite and skip its question about larger models.
   - **Remote**: ask for the `/v1/decisions` URL. Tell the user plainly what leaves this machine from now on: every Read and Bash output, fetched page, prompt, subagent task, Bash command and answer the enabled apps classify, as step 4 lists. Run `python3 ${CLAUDE_SKILL_DIR}/scripts/runtime.py health --url <url>` and `python3 ${CLAUDE_SKILL_DIR}/scripts/runtime.py probe --smoke --apps guard --url <url>`, both without a token. When either prints a `warning:` that requests travel unencrypted, tell the user that all of that, and the token if the runtime takes one, crosses the network in plain text, recommend an `https` URL, and ask them to confirm before you go on. When `health` lists more than one model, or the smoke probe shows `HTTP 400` for a runtime that answers only a named model (a hosted System One API does both), ask which of the listed models to use, add `--model <id>` to every `runtime.py` command from here on, and set `runtimeModel` to it in step 8. When either shows `HTTP 401` or `HTTP 403`, the runtime needs a token. Only then ask for it, and say why there are two steps: these checks are scripts outside Claude Code and cannot read its secure storage, so they read the token from a file; vSR itself reads it only from secure storage, which `/plugin configure` fills at the end. Ask the user to put the token in a file only they can read (`chmod 600`) and give you the path; they can delete the file once setup is done. Never ask them to paste the token into the chat. Add `--token-file <path>` to every `runtime.py` command from here on, and rerun `health` and the probe with it.

   Then time it on every branch: run `python3 ${CLAUDE_SKILL_DIR}/scripts/runtime.py probe --smoke --url <url>` (with `--token-file` when the runtime needs a token), which asks one question per app. When it prints `warning: slow runtime`, tell the user plainly, before anything is written: each vSR hook waits only a few seconds at most for an answer, so with this runtime the apps will often step aside without deciding; a GPU or remote runtime is recommended. Ask whether to keep this runtime or go back to step 4.

   Done when `runtime.py health --url <url>` prints `ready` (with `--token-file` when the runtime needs a token), the user confirmed any `warning:`, and the smoke probe came back without an error.

6. **Apps.** Custom only. Ask one AskUserQuestion call with two multiSelect questions, one line per option, marking the apps that are on by default as recommended:
   - Question "Which subagent and prompt apps?": **router** (picks Haiku, Sonnet or Opus for each general-purpose subagent by difficulty, and its starting effort by how many steps it needs; two questions in one request per subagent), **escalate** (after you push back, Claude re-checks and the next subagent goes one step up: Haiku to Sonnet, otherwise at least the session's model with one effort level more; one or two questions per prompt), **check** (marks an answer its reads do not back; one question per answer that read files).
   - Question "Which read and command apps?": **redact** (hides personal data and secrets in Read and Bash output; one question on every Read and Bash output), **shield** (warns Claude when a file or page tries to give it instructions; one question on every Read and WebFetch), **guard** (asks you before risky Bash commands; one question per chained part of each command).

   Add to the question text: each question costs about 0.1 to 0.2 s on an x86 CPU (several times more on an Apple Silicon CPU), less on a GPU, and redact and shield run on every Read. Done when every app is on or off.

7. **Thresholds.** Custom only. Ask with AskUserQuestion: **Defaults (Recommended)**, measured on vela3-lite, or **Advanced**. For Advanced, read [thresholds.md](thresholds.md), show its rows for the enabled apps only, each with its value under `effective` in step 1's `show` output, and let the user name the values to change. Done when you have the list of changed thresholds, possibly empty.

8. **Write.** Build one command with a `--set name=value` for each of: `runtime`, `runtimeUrl`, `runtimeModel` (the model step 5 chose, or empty, which clears one saved for an earlier runtime), the six app switches as chosen, and each threshold the user changed. The script saves only values that differ from vSR's defaults, and a value set back to its default clears the saved one, so those options follow the defaults when a later vSR changes them. Leave unchanged thresholds out: a saved one may hold a calibrated value. Run `python3 ${CLAUDE_SKILL_DIR}/scripts/settings.py plan --id <id> --set ...` and show the diff it prints. It writes nothing. Ask the user to confirm, then run the same command with `apply` in place of `plan`. It merges into the existing file, keeps every other key, and saves the previous copy beside it. When every value is already saved or at its default, `plan` prints `no change` and there is nothing to apply. Done when `apply` prints `wrote`, or `plan` printed `no change`.

9. **Verify.** Run `python3 ${CLAUDE_SKILL_DIR}/scripts/runtime.py probe --smoke --id <id> --url <url> --apps <enabled apps, comma-separated>`, with `--token-file` when step 5 used one. Report each app's verdict and latency, and the median latency, in a few lines. `ok` means the score is within 0.15 of the measured one and the verdict matches. Explain any `drift`, `flip` or `error`. Done when every enabled app has a verdict.

10. **Hand over.** Tell the user, in this order:
    - **First, when the runtime needs a token: run `/plugin configure <id>`** and enter the token in the masked field. vSR reads the token only from there; until it is saved, vSR sends requests without it and the runtime refuses them. The token file from step 5 can go after that.
    - Run `/reload-plugins`, then `/vsr`. The pane's `classifier` and `apps` lines must match what step 8 wrote. If they do not, restart Claude Code.
    - After a few hundred decisions, `/vsr:calibrate` tunes the thresholds from their own decisions.

    Done when the user has these steps.
