---
name: calibrate
description: Re-tunes the vSR plugin's thresholds from the user's own decision log. Asks the user about the decisions closest to each threshold, fits new thresholds to those labels (moving each at most 0.15 per run), shows before and after, and applies only on approval. Use when vSR asks for a calibration, when vSR holds, hides, warns or escalates too often or too rarely, when its subagent routing feels too weak or too strong, or after a change of classifier model.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/calibrate.py *) Bash(python3 ${CLAUDE_SKILL_DIR}/../setup/scripts/settings.py id *)
---

# Calibrate vSR

vSR logs every decision to its decision log (`logPath`, newest 5000) with the score it compared and the threshold it used. A calibration asks the user about the decisions closest to each line, then fits each enabled app's threshold to their answers. `${CLAUDE_SKILL_DIR}/scripts/calibrate.py` does the sampling, the fitting and the writing. Every number you show comes from its output; keep the math out of your own words.

What each threshold does, with measured examples, is in [../setup/thresholds.md](../setup/thresholds.md); read it when the user asks what a threshold means. Calibration moves the router's difficulty lines and confidence line; the three `routerSteps*` lines are not calibrated yet.

## Steps

1. **Report.** Run `python3 ${CLAUDE_SKILL_DIR}/../setup/scripts/settings.py id --plugin-root "${CLAUDE_PLUGIN_ROOT}"` for the plugin id, then `python3 ${CLAUDE_SKILL_DIR}/scripts/calibrate.py --id <id> report`. Show the report: per app, how many decisions, how many near the line, how many labels (theirs, including answers given with the buttons in `/vsr`, and implied by the log), and the score histogram with the threshold marked. When the log holds no decisions, say vSR has not logged any yet, and stop. Done when the user has seen the report.

2. **Label.** Run `python3 ${CLAUDE_SKILL_DIR}/scripts/calibrate.py --id <id> sample`. It prints a JSON list of questions: the unlabeled decisions closest to each enabled app's line, two on each side, apps without data left out. Tell the user once that the log already implies some labels: a guard hold they declined counts as "should ask", and an escalation within 15 minutes of a Haiku or Sonnet route counts as "that route was too low". Then ask the questions with AskUserQuestion, up to four per call, using each entry's `header`, `question` and `options` (`label` and `description`) as printed. After each call, save the answers in one command: `python3 ${CLAUDE_SKILL_DIR}/scripts/calibrate.py --id <id> label <decision id>=<option value> ...`. An answer the user typed as Other, or skipped, is not a label. Stop early when the user wants to. Done when every sampled question was asked or the user stopped, and every answer is saved (`saved N label(s)`).

3. **More labels?** An app needs 8 labels before its threshold can move. Rerun `report`; when an app the user cares about is short, offer one more `sample` round for it with `--apps <app>`. Done when the user is satisfied with the labels or no unlabeled decisions are left.

4. **Goal.** Ask with AskUserQuestion: **Balanced (Recommended)**, **Fewer false alarms** (vSR acts only when it is sure, at the cost of missing some), or **Catch more** (vSR acts on more borderline cases, at the cost of more false alarms). The values are `balanced`, `fewer-false-alarms` and `catch-more`. Done when the goal is chosen.

5. **Suggest.** Run `python3 ${CLAUDE_SKILL_DIR}/scripts/calibrate.py --id <id> suggest --goal <goal>`. Show its table and its settings diff as printed. For each changed threshold, say in one sentence what the user will notice, read from the table's "on the log" column (for example: "guard would have held 14 commands instead of 9, and agrees with 11 of your 11 labels instead of 9"). It writes nothing yet. When it suggests no changes, say so and stop. Done when the user has seen every change and its effect.

6. **Probe, after a model change.** When the report's "runtime model now" differs from the model of the last calibration, or the user asks, run `python3 ${CLAUDE_SKILL_DIR}/scripts/calibrate.py --id <id> probe`. It asks the runtime the built-in measured cases and judges them at the default thresholds they were measured at. A `flip` means the model now decides a known case the other way, and `drift` that its score moved more than 0.15. The `yours` column is the verdict under the user's current thresholds. Show the flips and drifts before the user approves. Done when the probe ran or was not needed.

7. **Apply.** Ask with AskUserQuestion whether to apply the suggested changes. Only on yes, run `python3 ${CLAUDE_SKILL_DIR}/scripts/calibrate.py --id <id> apply`. It merges the new numbers into the settings, keeps every other key, saves the previous copy beside the file, and records the change in `calibrations.jsonl` beside the log. Then tell the user to run `/vsr calibrated`, so vSR records when it was calibrated, and `/reload-plugins` to load the new thresholds. Done when `apply` printed `wrote` and the user has those two commands.
