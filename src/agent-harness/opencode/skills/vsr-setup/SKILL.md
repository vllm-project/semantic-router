# Set up vSR's decision runtime

vSR asks a small decision model about every child task, prompt, read, shell command and answer. This skill gets that model answering on this machine, or points vSR at one that already answers, picks which apps run and their thresholds, writes them, and checks each app. The scripts in `${VSR_SKILL_DIR}/scripts/` do every check and every install (where a step names one short, as `local.py start`, run it as `python3 "${VSR_SKILL_DIR}/scripts/local.py" start`); run them, show what they print, and ask the user only what they decide, with OpenCode's question tool. Keep your own summary to a few lines. Nothing here needs sudo; if a step seems to, stop and tell the user why.

vSR's two tools write and check: `vsr_settings` says vSR's settings with no input and sets what the user chose with input, kept over the plugin options as `/vsr-config` keeps them; `vsr_check` asks one case per enabled app through that app's own path.

The runtime is the Claude Code plugin's: `~/.local/share/vsr/venv`, port 8790, and its `vsr-runtime` service, so one runtime serves every agent on this machine.

## Steps

1. **Look first.** Call `vsr_settings` with no input, then run `python3 "${VSR_SKILL_DIR}/scripts/detect.py" --url <the runtime URL it names>` and show its six lines (machine, GPU, Python, vllm-srun, runtime, suggestion). It installs nothing. Done when the user has seen what this machine has and what detect suggests.

2. **Recommended or custom.** Ask one question:

   - **Recommended setup (recommended)**: say what it does on this machine from detect's suggestion: keep the runtime that answers (`existing`), install the pinned runtime over an older one and restart it (`upgrade`), or install and run vela3-lite on this machine's GPU (`local-gpu`) or CPU (`local-cpu`). Every app at its default (check stays off; turn it on later with a GPU or remote runtime) and the default thresholds.
   - **Custom**: the user picks where the model runs, which apps run, and the thresholds.

   When detect suggests `remote`, this machine cannot run the model, so skip the question and go on as Custom. Done when the user picked one. On Recommended, take detect's suggestion as step 4's branch, then go to step 7 with nothing to change. On Custom, go to step 3.

3. **Where the model runs.** Custom only. Ask one question, detect's suggestion first and marked recommended, with its reason:

   - **Keep the running one**, only when a runtime answers: nothing is installed. When detect suggests `upgrade`, offer **Upgrade it** first and say why (detect's reason).
   - **This machine's GPU**, only when detect found one: fastest; vela3-lite uses about 2 GB of GPU memory.
   - **This machine's CPU**: about 2.3 GB of RAM. The runtime computes in float32 on a CPU. On an idle 12-core desktop CPU (an i7-12700) a short question takes about 1 s, inside the 1.5 to 3 s waits, and an output of 8,000 characters 6 to 9 s, past shield's 2 s and redact's 3 s, so a long output is checked only in part unless the wait is until done. Other work on the CPU makes each about twice as slow. An ARM CPU is slower still: about 4 s a question on a GB10's. A Mac runs it on the CPU: the runtime cannot use an Apple GPU yet.
   - **A runtime on another machine**: nothing is installed. Everything the enabled apps check goes to that host: each read and shell output and the images one carries, fetched page, prompt, child task, shell command and answer.

   Done when the user picked one.

4. **Get the runtime answering.** Follow the branch picked, or detect's suggestion on the Recommended path.

   **Keep the running one**: nothing to do.

   **This machine's GPU or CPU, or Upgrade it**: the device is `cuda` (NVIDIA), `rocm` (AMD) or `cpu`; for an upgrade, the one detect's suggest line names, or when it names none, `cuda` or `rocm` for the GPU detect's gpu line shows and `cpu` without one. Every `local.py` command but `install` and `wait` needs it. When detect's vllm-srun line says the runtime is not installed from git, it is someone's own build: install nothing over it.

   1. Run `python3 "${VSR_SKILL_DIR}/scripts/local.py" install --dry-run --device <device>` (with `--python <path>` when detect's `python3` is older than 3.10 and it lists a newer one) and show the commands. They create the venv, install torch (the CPU wheels on Linux without a GPU, which save several GB) and the runtime pinned to the commit vSR was tested against with its `multimodal` extra (Pillow, which reading images needs); over an older install they first force the pinned one in. Say the download: about 200 MB of CPU torch or 2 to 3 GB of CUDA or ROCm torch, plus 1.8 GB of model weights on first start. For a new runtime, ask one question: **Install and start it for this login (recommended)**, which stops at reboot, and at logout where the system ends a user's processes; **Install and keep it running across reboots**, as a systemd user service on Linux or a launchd agent on macOS, which starts it each time the user logs in; or **Don't install** (back to step 3). For an upgrade, ask only whether to go ahead.
   2. Run the same command without `--dry-run`, in the background, since it takes minutes. When it prints `not installing: only N GiB disk free…`, tell the user and stop; only when they say the space is there (the model on another disk, say) rerun it with `--skip-disk-check`. On failure, show the last pip lines and stop. For `rocm`, torch from PyPI is a CUDA build: point the user to pytorch.org for the ROCm wheel index. Done when `local.py install --dry-run` prints `nothing to install`.
   3. Start it. For an upgrade: `python3 "${VSR_SKILL_DIR}/scripts/local.py" restart --device <device>`, which restarts the service or the runtime started for this login so it loads the new code; when it says to ask the user to restart theirs (a runtime setup did not start serves the port), do that. For this login: `local.py start --device <device>`. Across reboots: `local.py service --device <device>`, show the unit it prints, then rerun it with `--write` on the user's yes; a runtime started with `start` holds the port, so stop it first with `local.py stop`. Done when the command exited 0.
   4. Run `python3 "${VSR_SKILL_DIR}/scripts/local.py" wait` in the background; the first start downloads the model. Done when it prints `ready:` with how long a question takes; when the runtime exits or the wait times out, show the log lines it prints.

   A runtime on this machine needs no `/connect`: vSR asks `http://127.0.0.1:8790/v1/decisions` by default.

   **A runtime on another machine**: ask for its `/v1/decisions` URL and run `python3 "${VSR_SKILL_DIR}/scripts/runtime.py" health --url <url> --time`.

   - When it prints a `warning:` that requests travel unencrypted, tell the user that everything the apps check, and the token if the runtime takes one, crosses the network in plain text; recommend an `https` URL, and go on only when they confirm.
   - When it lists several models, ask which to use (one with the presets `pii, halu` that reads `image` uses every app), and rerun it with `--model <id>` to time that one; when it says the runtime refuses requests without a token, it needs one. Neither is typed here.
   - Then tell the user to run `/connect`, pick **vSR decision runtime**, then **Runtime with a token** (or **without a token**), and enter the URL, the model when it lists several, and the token in OpenCode's own form, which keeps it in OpenCode's credential store; never ask for the token in the chat. Wait until they say it is done.

   On every branch, when a script prints `warning: slow runtime` (detect for the running one, `local.py wait`, or `runtime.py health --time`), tell the user plainly before anything is written that the apps will often step aside, and ask whether to keep this runtime or go back to step 3. Done when the runtime answers (`ready`), or the user connected one on another machine.

5. **Apps.** Custom only. Ask one question, multiple choice, one line per app, the ones on by default marked recommended:

   - **router** (picks the model and effort of each new subagent from your three tiers by its task's difficulty), **escalate** (after you push back, the agent re-checks and the next subagent goes one tier up): both need `routing.tiers`, three models from one provider, in the plugin options in `opencode.jsonc`; when there are none, say so and offer to add them there.
   - **redact** (hides personal data and secrets in read and shell output), **shield** (warns the agent when a file or page tries to give it instructions), **guard** (asks you before risky shell commands).
   - **check** (marks an answer that claims more than the agent read; off by default, slow on a CPU runtime).

   Add to the question: each question costs about 0.05 s on a GPU and from a fraction of a second to several seconds on a CPU, and redact and shield run on every read. Done when every app is on or off.

6. **Thresholds.** Custom only. Ask one question: **Defaults (recommended)**, measured on vela3-lite, or **Advanced**. For Advanced, show the threshold lines step 1's `vsr_settings` listed for the enabled apps, each with its value and what it does, and let the user name the values to change. Done when you have the list of changed thresholds, possibly empty.

7. **Write.** Call `vsr_settings` once with what steps 5 and 6 chose: `apps` with each app on or off, and `thresholds` with each app whose lines changed (all of that app's lines, in the order `vsr_settings` lists them). On Recommended, call it with nothing to change. Show what it says. Done when it shows the settings the user chose.

8. **Check.** Call `vsr_check` and show each line, its `Warning:` line too. An app that has no answer, or decided otherwise than on vela3-lite, is said plainly with what to do: a runtime that does not answer (step 4), a slow one (a longer wait in `/vsr-config`, up to until done, which makes each check wait that long; or a GPU or remote runtime), a model that answers no score questions (the router steps aside), or a runtime line that says the model is not set (name one in `/connect`). Done when every enabled app answered, or the user knows why one did not.

9. **Hand over.** Tell the user, in this order:

   - `/vsr` opens the decisions panel; its top turns apps on and off and picks their waits.
   - `/vsr-config` changes any setting later, thresholds included, and its last row checks the runtime again; so does a click on the panel's runtime line.
   - After some use, `/vsr-calibrate` fits the thresholds to their own answers.

   Done when the user has these steps.
