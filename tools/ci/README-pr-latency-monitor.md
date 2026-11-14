# PR latency monitor

This read-only monitor checks PR workflow admission, the required `PR Gate`,
and Mergify queue transitions. It runs outside GitHub Actions so an Actions
outage cannot silence its own scheduler. It posts one grouped Slack message
for new incidents and one for recoveries; a local state file prevents repeated
alerts on every poll.

## Signals and limits

| Signal | Initial threshold |
| --- | --- |
| Current queued Pull Request run has zero jobs | 10 minutes |
| PR with two approvals has no PR Gate check | 10 minutes after latest approval |
| Two approvals and green PR Gate but no Mergify queue entry | 10 minutes after eligibility |
| Queued PR has no speculative check PR | 20 minutes |
| PR remains queued | 90 minutes |
| Branch-protection or update-related dequeue | Next poll, if within 10 minutes |

The monitor leaves an in-progress Gate alone, including `Waiting for committer
approval`. It never changes PRs, checks, reviews, queue state, or
repository settings. These thresholds are proposed in #6383 and should be
calibrated against observed latency. Mergify state comes from its current
bot comment; the monitor does not require a Mergify API credential.

The GitHub search index can lag, and a green same-name check may still be in
a failed or cancelled suite (#5956). Alerts are triage signals, not proof
that a PR is safe to merge. A Mac that is asleep, powered off, or disconnected
cannot send alerts; monitor availability needs an independent heartbeat if
that failure must be covered.

## Set up on a Mac

1. Install and authenticate the GitHub CLI with read access to this repository:
   `gh auth status`. Keep its credential in the user's normal secure storage.
2. Ask the Slack workspace to authorize a channel-scoped incoming webhook for
   the maintainer alert channel. Workspace app approval may be required.
   Save the URL in a file **outside this repository** and set mode `0600`.
3. Choose an absolute path outside the repository for the state JSON. Run once
   with `--bootstrap` to record existing incidents without flooding Slack,
   then run once with `--dry-run` to inspect the alert text.
4. Schedule the command every five minutes with a user LaunchAgent. Use
   absolute paths for Python, the script, the state file, and the webhook file;
   set `PATH` so the job can find `gh`. Capture standard output and error
   in local logs. Keep the Mac awake and signed in to the account that owns
   the LaunchAgent.

```bash
python3 tools/ci/pr_latency_monitor.py \
  --state-file /absolute/private/path/pr-latency-state.json \
  --bootstrap

python3 tools/ci/pr_latency_monitor.py \
  --state-file /absolute/private/path/pr-latency-state.json \
  --dry-run

python3 tools/ci/pr_latency_monitor.py \
  --state-file /absolute/private/path/pr-latency-state.json \
  --webhook-file /absolute/private/path/slack-webhook
```

The final command is what the LaunchAgent runs. Do not place the webhook URL
in its property list, command arguments, shell history, logs, or GitHub.
The script checks that its file is owner-only before using it. A failed poll
exits nonzero and leaves the previous state intact; investigate its error
log before interpreting silence as a healthy queue.
