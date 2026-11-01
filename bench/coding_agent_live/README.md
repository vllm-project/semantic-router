# Claude Code Lane

This lane runs Claude Code itself on Harbor tasks, such as Terminal-Bench, with
every model call going through the Router's Anthropic Messages ingress. It
answers what a contract test cannot: whether a real coding-agent session, with
its own system messages, failed tool calls, effort settings and subagents,
survives the Router to the end, which models the Router chose for the main
agent and for each subagent, and what the task's verifier says about the result.

Each trial reports the verifier reward, Harbor's exception (a timeout keeps the
reward the verifier gave), Claude Code's own result event (terminal reason,
turns, duration, subagents spawned by type), the API errors Claude Code printed,
and the Router's replay records for the session: calls and tokens per selected
model, response statuses, and protection actions. When the Router records agent
identity, calls are split into the main agent and its subagents, otherwise they
are reported together.

The lane needs a model backend, Docker and network access to install Claude
Code in each task container, so it runs locally and never in CI. The tests in
this directory check the command, the alias mapping and the report offline.

## Set up

Use a Python environment with Harbor installed. `vllm-sr bench` already sets one
up at a pinned revision for its Terminal-Bench harness, and the task checkout it
prepares can serve as the dataset.

The Router needs three things for the lane:

- a listener the task containers can reach, usually on the Docker bridge, with
  `identity.trust_headers` on and the bridge's network in `trusted_peers`, so
  Claude Code's session header reaches the Router;
- `global.services.router_replay.enabled: true` and the management API, so the
  report can read the session's records;
- an entrypoint `model_names` entry for every model name the lane sends.

The Router takes the session id of an Anthropic Messages request from
`x-claude-code-session-id` when no `x-session-id` is sent, so replay records and
Claude Code's result event share one id without extra configuration.

## Run

```bash
python bench/coding_agent_live/run.py \
  --dataset /path/to/terminal-bench-2-1/tasks \
  --task fix-git --task log-summary-date-ranges \
  --router-url http://172.17.0.1:8899 \
  --model vllm-sr/parent --haiku-model vllm-sr/small \
  --jobs-dir jobs --job-name lane-1

python bench/coding_agent_live/report.py jobs/lane-1 \
  --replay-url http://127.0.0.1:8080
```

`--model` is the Router model name of the main agent. `--opus-model`,
`--sonnet-model`, `--haiku-model` and `--subagent-model` map Claude Code's tier
aliases and its subagent default, and each defaults to `--model`. Claude Code's
Explore agent uses the haiku alias. An empty value leaves that variable unset,
so a subagent inherits its parent's model. `--claude-code-version` pins the
release, and `--dry-run` prints the Harbor command and the mapping.
`--agent-kwarg KEY=VALUE`, repeatable, passes an option of Harbor's Claude Code
agent, for example `reasoning_effort=xhigh` to pin the effort Claude Code sends,
or `append_system_prompt=...`.
`--agent-env KEY=VALUE`, also repeatable, sets an environment variable for
Claude Code in the task container. Claude Code 2.1.304 starts Agent subagents in
the background and a print-mode run can end before they report, so subagent
runs pass `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`.

The report prints one JSON row per trial and a final summary with the pass
count, the trials that ended on an API error, Harbor's exceptions, and the
subagents spawned.
