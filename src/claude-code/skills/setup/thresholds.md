# vSR thresholds

Each line is one `userConfig` option: what crossing it does, and which way makes vSR act more often. Its default is in the plugin manifest; `settings.py show` prints the value in use under `effective`. Scores are the classifier's, 0 to 1 (router's difficulty and steps 0 to 2). "Measured" numbers come from vela3-lite on CPU.

| Option | Lower means | What it decides |
| --- | --- | --- |
| `routerEasyBelow` | fewer Haiku routes | A subagent task whose difficulty is under this goes to Haiku, always at medium effort. Measured: "List the files in this repo." 0.03. |
| `routerHardFrom` | more Opus routes | From this difficulty on, the task goes to Opus; between the two cuts it goes to Sonnet. Measured: a refactor plus README update 0.74, a timing-leak proof 1.75. |
| `routerMinConfidence` | more tasks routed | Under this confidence the answer is too spread to act on, and the subagent keeps the model it would have had. |
| `routerStepsLow` | fewer low-effort starts | Starting effort inside the model's band, from the `steps` question ("How many reasoning steps does this task need?", 0 to 2). Under this, Sonnet starts at low and Opus at high; Haiku is always at medium. Measured: "List the files in this repo." 0.17. |
| `routerStepsHigh` | more xhigh starts | From this on (many steps), Sonnet starts at high and Opus at xhigh; between the two lines Sonnet starts at medium and Opus at high. Opus never starts below the effort the subagent would have had. Measured: a refactor plus README update 0.52, a timing-leak proof 1.33. |
| `routerStepsMinConfidence` | more steps answers used | Under this confidence the steps answer is ignored and the band's default effort is used (Sonnet medium, Opus high). A missing or failed steps answer does the same and never switches the router off. Measured confidences: refactor 0.23 (so medium), timing-leak proof 0.35. |
| `escalateFeedback` | more escalations | When a prompt's "wrong answer" or "want something different" probability reaches this, the next subagent goes to Opus and Claude is told to re-check. Measured: "No, task 2 is wrong…" 0.92, "Thanks, that works perfectly." 0.00. |
| `escalateReask` | more escalations | When a prompt asks for the same thing as the one before with this probability or more, vSR escalates too. Measured: "That's not what I asked. Summarize src/login.py again…" 0.52. |
| `redactPii` | more hidden | A personal-data span found with this probability or more is replaced by its tag (`[vsr:email]`, `[vsr:name]`) before Claude reads it. Secret patterns are hidden whatever this is. |
| `shieldInject` | more warnings | When a file or page scores this or more as a prompt injection, Claude is warned to treat it as data. Measured: an HTML-comment injection 0.99, a plain README 0.08. |
| `guardReview` | more holds | When any part of a Bash command scores this or more for "a person should review it", vSR asks you before it runs. Measured: `git push origin main` 0.76, `rm -rf ~` 0.78 (under this line; the fixed rules below hold it), `git reset --hard HEAD~1` 0.95, `git push --force origin main` 0.98. |
| `checkUnsupported` | more answers marked | An answer scoring this or more for claims its reads do not back is marked unverified. Measured: a faithful summary 0.04, one inventing a lockout and refund 0.65. |
| `calibrateEvery` | more reminders | After this many decisions since the last calibration, `/vsr` suggests running `/vsr:calibrate`. |

**Always held.** guard asks you before a catastrophic command whatever it scores and whatever `guardReview` is: `rm -rf` of `/`, `~`, `$HOME` or a top-level system directory such as `/etc` or `/usr`; `mkfs`; `dd` writing to a block device; a fork bomb; `chmod -R` or `chown -R` on `/`; and `curl` or `wget` piped to a shell. These are fixed rules, not scores, so no threshold and no calibration changes them. The score still decides every other command, and it can miss: `rm -rf ~` scores 0.78 above, under the default line.

Moving a threshold by 0.05 to 0.1 is a real change; prefer letting `/vsr:calibrate` move it from your own decisions.
