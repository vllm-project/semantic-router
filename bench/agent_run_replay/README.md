# Agent Run Replay

`replay.py` sends recorded agent runs through a running Router, step by step and
as the agent sent them, and reports what the Router did with each run: the model
it selected on every step, the switches inside the run, the session phases and
the protection actions. A backend started by the script answers each request
with the recorded reply, so no model is called and a run costs nothing. The
recordings come from public leaderboard trajectories, so the replay covers many
agents, models and tasks rather than one hand-written session
(`../agent_session_replay.py`).

The same recorded content can travel in two wire forms. In `text` form the
agent's commands and their outputs are assistant and user messages, the way
mini-SWE-agent and terminus send them. In `tool` form they are tool calls and
tool results. Protection treats the two differently, which is what the replay
makes visible.

## Get recordings

```bash
python bench/agent_run_replay/replay.py fetch-tbench --out runs.jsonl \
  --models deepseek-v3.2,gpt-5-mini --tasks 10
```

This reads the Terminal-Bench 2.0 leaderboard trajectories
([`yoonholee/terminalbench-trajectories`](https://huggingface.co/datasets/yoonholee/terminalbench-trajectories),
Apache-2.0) at a pinned revision and keeps the terminus-2 runs of the named
models. It needs `pandas` and `huggingface_hub`. The scrape leaves the task text
of most runs and some command outputs as unresolved references, which the
recordings carry as `[not recorded in the source]` (the task keeps its name).

Other sources only need to be written in the recording schema, one run per
line:

```json
{"schema_version": "agent-run-recording.v1", "run_id": "...", "task_id": "...",
 "model": "...", "outcome": true, "cost_usd": 0.12, "system": "...", "task": "...",
 "steps": [{"reply": "...", "command": "ls\n", "observation": "..."}]}
```

`outcome` and `cost_usd` may be `null`. A step without a command is a reply
only, and the last step usually has no observation. Keep data whose license
does not allow redistribution out of the repository.

## Replay runs

Start the Router on the bundled config, then replay:

```bash
make build-router
./bin/router -config bench/agent_run_replay/configs/router.yaml -gateway standalone
python bench/agent_run_replay/replay.py run --recordings runs.jsonl --wire text --out out/text
python bench/agent_run_replay/replay.py run --recordings runs.jsonl --wire tool --out out/tool
```

`configs/router.yaml` serves two models from the replay's backend on port 18771
and adds a probe decision that prefers `strong` once a request reaches 2K
tokens, so the decision changes inside a run as the history grows. Each run
uses its `run_id` as session and conversation id. Change
`global.router.learning.protection.scope` to compare protection scopes. The
Router writes `router-runtime.json` next to the config it serves, so delete it
afterwards or serve a copy of the config.

`out/<name>/runs.jsonl` has one row per run, and `summary.json` totals them:

| Field | Meaning |
| --- | --- |
| `selected_models` | Steps per model, from `x-vsr-selected-model` |
| `switch_steps` | Steps where the selected model differs from the step before |
| `phases` | Steps per `x-vsr-session-phase`, for example `tool_loop` |
| `learning_actions` | Steps per `x-vsr-learning-actions`, for example `protection=hold_current` |
| `first_replay_id` | Router Replay record of the run's first request |

On ten Terminal-Bench 2.0 tasks (30 runs, 425 requests) against a Router built
from main, the text form under conversation scope switched model inside 15 of
the 30 runs. The tool form under the same config held the first model through
the tool loop (391 `hold_current`, 1 switch), and the text form under session
scope switched in none.

## Learn from outcomes

`learn` streams tasks recorded for several models. The Router's choice on the
first request decides which model's recording the run follows, and with
`--post-outcomes` the recorded outcome of that run is posted to
`/api/v1/observability/outcomes` on the first request's replay record, within
the API's 60 per minute limit.

```bash
./bin/router -config bench/agent_run_replay/configs/router-learning.yaml -gateway standalone
python bench/agent_run_replay/replay.py learn --recordings runs.jsonl \
  --models deepseek-v3.2,gpt-5-mini --post-outcomes --out out/learn
```

The model names in `configs/router-learning.yaml` match the recordings' `model`
field, and its pricing is each provider's list price. `summary.json` adds the
model each run was served on, over all runs and over the second half, and the
status of each outcome post.

## Test

```bash
make test-agent-run-replay-bench
```

The tests run the replay against a stand-in Router and outcome API, without a
Router, a model or network access.
