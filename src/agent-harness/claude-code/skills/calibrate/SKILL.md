---
name: calibrate
description: Re-tunes the vSR plugin's thresholds from the user's own decision log. Tunes only the parts the user names (for example just the routing difficulty lines), asks about the decisions closest to each line in plain words, fits new lines to the answers (moving each at most 0.15 per run), explains the change and recommends whether to apply it, and applies only on approval. Use when vSR asks for a calibration, when vSR holds, hides, warns or escalates too often or too rarely, when its subagent routing feels too weak or too strong, or after a change of classifier model.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/calibrate.py report *) Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/calibrate.py sample *) Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/calibrate.py label *) Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/calibrate.py suggest *) Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/calibrate.py probe *) Bash(python3 ${CLAUDE_SKILL_DIR}/../setup/scripts/settings.py id *)
---

# Calibrate vSR

vSR logs every decision to its decision log with the score it compared and the line it used. A calibration asks the user about the decisions closest to the lines of the parts they chose, then moves those lines to fit their answers. `${CLAUDE_SKILL_DIR}/scripts/calibrate.py` does the sampling, the fitting and the writing, and prints everything in plain words. Every number you show comes from its output. Show its plain output as printed. Keep option names (`routerEasyBelow` and the like), scores and the maths out of what you tell the user unless they ask.

The parts ("apps") and their names for `--apps`: `router` (which model each subagent gets), `guard` (asks before risky commands), `redact` (hides personal data), `shield` (warns about injected instructions), `escalate` (reacts when the user pushes back), `check` (marks answers that claim more than Claude read). When the user asks what a line means, read [../setup/thresholds.md](../setup/thresholds.md) and answer in plain words. The router's effort lines are not calibrated yet. When the runtime needs a token, add `--token-file <path>` after the subcommand, with the same file only the user can read that setup used.

Every command below is `python3 ${CLAUDE_SKILL_DIR}/scripts/calibrate.py <subcommand> ...`. `<apps>` is the comma list chosen in step 1, for example `router`.

## Steps

1. **Scope.** Run `python3 ${CLAUDE_SKILL_DIR}/../setup/scripts/settings.py id --plugin-root "${CLAUDE_PLUGIN_ROOT}"` for the plugin id, then `report --id <id>`. When the log holds no decisions, say vSR has not logged any yet, and stop.
   - When the user's request already names what to tune, use it and do not ask: routing, difficulty, model choice, Haiku/Sonnet/Opus → `router`; risky commands, holds, confirmations → `guard`; personal data, hiding → `redact`; injection, instructions in files → `shield`; pushback, complaints, escalation → `escalate`; answer checking, unverified answers → `check`.
   - Otherwise ask one AskUserQuestion with `multiSelect: true`, header "Tune what", question "Which parts of vSR should I tune?". Give one option per app in the report, at most four: leave out first the apps with no decisions, then the ones marked "off in your settings". The label is the app's plain name from the report (for example "Routing"). The description is its purpose and its answer count, for example "which model each subagent gets; 23 answers". Add " (Recommended)" to the label of each app with at least 8 answers.

   Done when the apps are chosen.

2. **Report.** Run `report --id <id> --apps <apps>` and show it as printed. Done when the user has seen it.

3. **Label.** Run `sample --id <id> --apps <apps>`. It prints a JSON list of questions about the decisions closest to each line. When router or guard is chosen, tell the user once that the log already counts some answers for them: a held command they declined counts as "should ask", and pushback within 15 minutes of a Haiku or Sonnet route counts as "needed stronger". Ask the questions with AskUserQuestion, up to four per call, using each entry's `header`, `question` and `options` (`label` and `description`) as printed. After each call, save the answers in one command: `label --id <id> <decision id>=<option value> ...`. An answer the user typed as Other, or skipped, is not a label. Stop early when the user wants to. Done when every question was asked or the user stopped, and every answer is saved (`saved N label(s)`).

4. **Enough answers?** Each app needs 8 answers before its lines can move. Rerun `report --id <id> --apps <apps>`. When a chosen app is short, offer one more `sample` round for it. Done when the user is satisfied or nothing is left to ask.

5. **Goal, only for apps other than router.** When router is the only chosen app, skip this step: routing has no goal. Otherwise ask with AskUserQuestion: **Balanced (Recommended)** (as few wrong calls as possible either way), **Fewer false alarms** (vSR steps in only when it is sure, and may miss some), **Catch more** (vSR steps in on more borderline cases, with some false alarms). The values are `balanced`, `fewer-false-alarms` and `catch-more`. Done when the goal is chosen.

6. **Suggest.** Run `suggest --id <id> --apps <apps>`, and add `--goal <goal>` when step 5 asked for one. It writes nothing to the settings. Show its output as printed, down to the `Overall` line. For each app it says what each line does, what the change would do on the user's own log, how many of their answers agree before and after, and one recommendation. Add nothing of your own. Show the table and the settings diff only when the user asks for details: rerun with `--detail` and show the part under `Details`. When no line changes, say the current settings already fit the answers, and stop. Done when the user has seen the suggestion.

7. **Probe, after a model change.** When the report's "Decision model now" differs from the model of the last calibration, or the user asks, run `probe --id <id>`. It asks the runtime the built-in measured cases. A `flip` means the model now decides a known case the other way, and `drift` that its score moved more than 0.15. Before the user approves, tell them in plain words which cases flipped or drifted. Done when the probe ran or was not needed.

8. **Apply.** Ask one AskUserQuestion. Put the script's recommendation first, with " (Recommended)" added to its label:
   - One app: "Apply" and "Keep the current settings", in the order of its "Recommended:" line.
   - Several apps: "Apply the recommended" (the apps the `Overall` line names, when it names any), "Apply all changes", "Keep the current settings".
   - An app with "Not enough answers yet" has nothing to apply. When no chosen app has anything to apply, offer "Answer more questions" (back to step 3) and "Stop here" instead.

   Only on an apply answer, run `apply --id <id> --apps <apps to apply>`. It applies only what the saved suggestion holds, merges the new numbers into the settings, keeps every other key, saves the previous copy beside the file, and records the change in `calibrations.jsonl` beside the log. Then tell the user to run `/vsr calibrated`, so vSR records when it was calibrated, and `/reload-plugins` to load the new lines. Done when `apply` printed `wrote` and the user has those two commands, or the user chose to keep their settings.
