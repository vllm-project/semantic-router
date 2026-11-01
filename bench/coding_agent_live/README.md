# Codex CLI lane

This lane runs Codex CLI itself on Harbor tasks, such as Terminal-Bench, with
every model call going through the Router's Responses ingress. It answers what
a contract test cannot: whether a real Codex session finishes through the
Router, which API errors Codex printed, and what the task's verifier says.

Each trial reports the verifier reward, Harbor's exception (a timeout keeps the
reward the verifier gave), the API errors Codex printed, the Codex version, and
the Router's replay records for that run: calls and tokens per selected model,
response statuses, and protection actions. Records are grouped by the run.
Codex's root also sends `thread-id`, so a record that carries an agent id is
not reported as a subagent. Splitting those records by thread waits until
replay stores the agent id.

The lane needs a model backend, Docker, and network access so Harbor can install
Codex in each task container. It runs locally and never in CI. The tests in
this directory check the command, the Codex settings, and the report offline.

Chat, Responses, and Messages outcomes are the `api_format` of the selected
backend. Codex always calls the Responses ingress. Point `--model` at an
entrypoint whose route uses the backend format you want to record.

## Set up

Use a Python environment with Harbor installed. `vllm-sr bench` sets one up at
the pinned Harbor revision for its Terminal-Bench harness, and the task
checkout it prepares can serve as the dataset.

The Router needs:

- Responses enabled (`global.services.response_api`), so Codex's
  `POST /v1/responses` is served;
- `global.services.router_replay.enabled: true` and the management API, so the
  report can read the run's records;
- protection identity that reads Codex's session header, with session scope so
  a missing conversation header does not drop the join:

```yaml
global:
  router:
    learning:
      protection:
        scope: session
        identity:
          headers:
            session: session-id
```

- an entrypoint for every model name the lane sends;
- a listener the task containers can reach, usually on the Docker bridge. If
  that deployment drops client headers unless the peer is trusted, trust the
  bridge.

## Run

```bash
python bench/coding_agent_live/codex_run.py \
  --dataset /path/to/terminal-bench-2-1/tasks \
  --task fix-git \
  --router-url http://172.17.0.1:8899 \
  --model vllm-sr/parent \
  --jobs-dir jobs --job-name lane-1

python bench/coding_agent_live/codex_report.py jobs/lane-1 \
  --replay-url http://127.0.0.1:8080
```

`--model` is the Router entrypoint. The lane writes it into Codex's `model`
setting and into every other model setting already present in the config Harbor
merges. `--router-url` is the inference address the task containers use; the
command appends `/v1` when it is missing, because Codex posts to
`{OPENAI_BASE_URL}/responses`. `--codex-version` pins the release (default
`0.163.0`). `--dry-run` prints the config and the Harbor command without
starting Harbor.

The config the lane gives Harbor sets the protocol-guide limits:

```toml
web_search = "disabled"

[features]
multi_agent = false
```

Hosted web search and Codex `multi_agent` stay off.

The report prints one JSON row per trial and a final summary with the pass
count, the trials that ended on an API error, Harbor's exceptions, and the
Codex versions in the job.
