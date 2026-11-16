---
name: vllm-sr-agent-operations
description: Install, configure, verify, maintain, troubleshoot and optimize vLLM Semantic Router v0.5 on a Docker host (CPU, AMD or NVIDIA GPU) or a Kubernetes cluster (Helm or the operator), through its CLI and Router API. Use for a first install, configuration and routing changes, checks of routed requests, upgrades and the v0.5 config migration, secrets, logs, health, backup and restore, removal, failure diagnosis, and tuning of the decision model, recipes, caches, latency and cost with sr-bench.
---

# vLLM Semantic Router: operate it as an agent

The Router serves an OpenAI-compatible API and picks a model for each request.
Every routing judgment (a domain, a jailbreak, the complexity of a request, a
question you write) is a question to the **decision model**, System One, which
the Router's built-in model runtime serves; it starts a model only when a rule
or feature asks it. The models that answer users are the user's own endpoints,
such as vLLM, Ollama or a hosted API. `vllm-sr serve` runs the stack in Docker:
the Router (inference on port 8899, management API on 8080) with its model
runtime, the Dashboard (8700), sr-bench (8090) and supporting services.

This file routes you to a task. Each task names its preconditions, the
reference that holds its steps, what success looks like, what to do when a
step fails, and when to stop and ask the user. Every step is safe to run again.

## Ground rules

- Nothing waits for a person.
  Never run `vllm-sr serve` without a complete `--config`: without one it
  starts the Dashboard's setup mode and waits indefinitely. Pass every answer
  as a flag; no command here prompts.
- Keep secrets in environment variables, never in YAML, command arguments,
  logs or your report. Name the variable (`api_key_env`, `--token-env`).
- Don't stop, restart or replace what you didn't start (containers, a running
  `vllm-sr` stack, services on its ports) without asking.
- Ask before system-wide changes: installing packages, adding the user to a
  group, publishing a port beyond loopback, deleting volumes or state.
- Success is what the output contains: an exit code, a JSON field, a header
  value. An HTTP 200 alone is not success.
- Change one thing at a time, then rerun the step that failed. Keep the
  evidence of a failed attempt; don't rewrite a check to make it pass.
- When an error names the command that fixes it (an image pull, a rebuild,
  `vllm-sr config migrate`), run that command if these rules allow it, then
  rerun the step. Don't work around the message.

## Conventions

- A stack is named by `VLLM_SR_STACK_NAME` (default `vllm-sr`, which names its
  containers) and moved by `VLLM_SR_PORT_OFFSET` (default 0, added to every
  host port). Export both for every command of a second stack.
- Run a stack's `vllm-sr` commands from its config directory; the stack keeps
  its state in `.vllm-sr/` beside `config.yaml`.
- With offset 0: inference `http://127.0.0.1:8899/v1`, management
  `ROUTER_ORIGIN=http://127.0.0.1:8080`, Dashboard `http://127.0.0.1:8700`.
  Add the offset to each port otherwise.
- JSON checks use `python3 -c`; header names are case-insensitive, so use
  `grep -i`.

## Preflight

These commands change nothing. Run them and keep the results.

```bash
uname -sm
docker info --format '{{.ServerVersion}} {{.MemTotal}}'
python3 -c 'import sys; print(sys.version.split()[0])'
python3 -m ensurepip --version
df -h "$HOME" | tail -1
free -g 2>/dev/null | head -2
ls /dev/kfd /dev/dri/render* 2>/dev/null
nvidia-smi -L 2>/dev/null
vllm-sr --version 2>/dev/null && vllm-sr status
```

| Check | Pass | Otherwise |
| --- | --- | --- |
| Docker | prints a version | "permission denied": ask the user to add their account to the `docker` group. Not installed: ask before installing it. Podman works too: add `--container-runtime podman` to `vllm-sr` commands. |
| Python | 3.10 or newer, and `ensurepip` prints `pip …` | Ubuntu and Debian: with the user's OK, `sudo apt-get install -y python3-venv`. Without it the installer stops at "ensurepip is not available". |
| Disk | 15 GB free; 40 GB for a GPU platform | Ask the user. The images take about 4 GB (about 22 GB for ROCm) and the default decision model about 6 GB. |
| Memory | `MemTotal` of 12 GB or more, and 8 GB available | The default decision model holds about 7 GB on the CPU. Ask: leave the decision-model route out of the first configuration, or run the model on a GPU or a remote System One API ([Optimization](https://vllm-sr.ai/install/agent/vllm-sr/references/optimization.md#choose-the-decision-model)). |
| `vllm-sr` | not installed, or `status` says `Not running` | A stack is already running. It is the user's: find its directory (below), verify it with [Verification](https://vllm-sr.ai/install/agent/vllm-sr/references/route-verification.md) using its own decision and model names, and ask before changing it. |

Then check the host ports a stack publishes:

```bash
bash -c 'for p in 8899 8080 8700 8090 9190 50051 6379 5432 3000 9090 4317 16686; do
  (exec 3<>/dev/tcp/127.0.0.1/$p) 2>/dev/null && echo "port $p is in use"; done'
```

Nothing printed: pass. Otherwise ask the user, or run this stack beside the
other one with a [stack name and port offset](https://vllm-sr.ai/install/agent/vllm-sr/references/deployment-loop.md#stack-identity).

Each running stack has one Router container: `vllm-sr-router-container` for
the default stack, `NAME-vllm-sr-router-container` for a stack named `NAME`.
Its directory is the parent of the `.vllm-sr` directory that Router mounts:

```bash
docker ps --format '{{.Names}}' | grep -- '-router-container$'
docker inspect vllm-sr-router-container \
  --format '{{range .Mounts}}{{println .Source}}{{end}}' | grep -m1 '/\.vllm-sr$'
```

## Tasks

| Task | When | Reference |
| --- | --- | --- |
| Deploy | A first install on Docker (CPU, AMD, NVIDIA) or Kubernetes (Helm, the operator); GPUs, Envoy, engine mode, a second stack, a pinned image, a new backend | [Deployment](https://vllm-sr.ai/install/agent/vllm-sr/references/deployment-loop.md) |
| Configure | Change models, signals, decisions, plugins, recipes or entrypoints of a stack | [Configuration](https://vllm-sr.ai/install/agent/vllm-sr/references/configuration-loop.md) |
| Verify | Prove that a request takes the intended route and gets a complete answer | [Verification](https://vllm-sr.ai/install/agent/vllm-sr/references/route-verification.md) |
| Maintain | Upgrade, migrate a configuration to v0.5, roll back, back up and restore, rotate secrets, read logs and health, remove | [Maintenance](https://vllm-sr.ai/install/agent/vllm-sr/references/maintenance.md) |
| Troubleshoot | A step failed, or the stack misbehaves | [Troubleshooting](https://vllm-sr.ai/install/agent/vllm-sr/references/troubleshooting.md) |
| Optimize | Improve quality, latency or cost: the decision model and its hardware, recipes, caches, judges, sr-bench comparisons | [Optimization](https://vllm-sr.ai/install/agent/vllm-sr/references/optimization.md) |

### Deploy

- **Preconditions:** preflight passed; a model endpoint the user named, or one
  you found and they confirmed; a target (Docker on this host, or a kubectl
  context).
- **Run:** [Deployment](https://vllm-sr.ai/install/agent/vllm-sr/references/deployment-loop.md), steps 1 to 5: choose the path, install
  the CLI, write and validate the configuration, start the stack, check it.
- **Success:** `vllm-sr config schema` reports `"config_version": "v0.5"`;
  `serve` exits 0 after `✓ vLLM Semantic Router is running (standalone gateway)`;
  `vllm-sr route probe … --expect-decision code-route` exits 0; a preview of a
  computer-science question returns `"routing_decision": "cs-route"`, and
  `GET $ROUTER_ORIGIN/api/v1/inventory/model-runtime` lists `primary` with
  `"ready": true`.
- **On failure:** the Install and Start tables of
  [Troubleshooting](https://vllm-sr.ai/install/agent/vllm-sr/references/troubleshooting.md).
- **Stop and ask:** Docker is missing or denied; a stack already runs; a port
  is in use; memory is short for the decision model; the API must be reachable
  beyond loopback; Kubernetes has no context or storage class.

### Configure

- **Preconditions:** a running stack and its `ROUTER_ORIGIN`, or a config
  directory for a new one; one stated intent per change.
- **Run:** [Configuration](https://vllm-sr.ai/install/agent/vllm-sr/references/configuration-loop.md): read the active document
  with `vllm-sr config get`, edit a copy, then `config validate`, `config plan`
  and `config apply`.
- **Success:** `config apply` exits 0 and prints `"applied": true`;
  `vllm-sr config versions` shows the new version; a preview of each affected
  route returns the intended `routing_decision`.
- **On failure:** a validation error names the field in one line; a removed
  form says ``run `vllm-sr config migrate` ``; `Restart required` needs a
  `serve`. See the Configuration table of Troubleshooting.
- **Stop and ask:** a change that needs a restart of a stack you didn't start;
  publishing a listener; adding credentials, or delegating each user's own
  credential to an upstream (it needs an authenticator in front); removing
  decisions, recipes or entrypoints the user didn't name.

### Verify

- **Preconditions:** the stack runs; the entrypoint is listed by
  `GET /v1/models` (`vllm-sr/auto` by default).
- **Run:** [Verification](https://vllm-sr.ai/install/agent/vllm-sr/references/route-verification.md): preview, then probe, then
  read the headers, the decision model's readiness and, when allowed, the
  trace.
- **Success:** preview JSON `routing_decision` and `selected_model` are the
  intended ones; `route probe` exits 0 with `"passed": true` and
  `response.status` 200; a traced preview has no `signal_errors`.
- **On failure:** the Verification and Decision model tables of
  Troubleshooting.
- **Stop and ask:** never change policy, thresholds or Learning to make a
  check pass; report the failure instead.

### Maintain

- **Preconditions:** the stack's directory and its current `vllm-sr --version`;
  a backup of `config.yaml` and `.vllm-sr/` before any upgrade or migration.
- **Run:** [Maintenance](https://vllm-sr.ai/install/agent/vllm-sr/references/maintenance.md): upgrade, `vllm-sr config migrate`,
  `config versions` and `config rollback`, backups, `vllm-sr storage rotate`,
  logs and health, removal. An upgrade is: install the new CLI, migrate,
  read every note and restore what was dropped, validate, then `serve`,
  which keeps the old stack running until the new Router accepts the
  configuration. A configuration that reads the caller's identity ran
  behind Envoy, and takes `--gateway extproc` in all three commands.
- **Success:** migrate exits 0 after `✓ Configuration migrated` and its output
  validates with `Checked by the Router in …`; after an upgrade the stack
  runs, the new version prints and the probe passes again.
- **On failure:** the Configuration and migration table of Troubleshooting;
  a migrate that can't find its Router image prints the command that fetches
  it. Restore the backup when the new version can't serve.
- **Stop and ask:** downtime of a stack the user runs, deleting volumes or
  `.vllm-sr/`, rotating credentials, removing the CLI or images.

### Troubleshoot

- **Preconditions:** the failing step and its exact output.
- **Run:** find the symptom in [Troubleshooting](https://vllm-sr.ai/install/agent/vllm-sr/references/troubleshooting.md), run its
  check, apply its fix, rerun the failed step.
- **Success:** the failed step now meets its own success criterion.
- **On failure:** the next matching row; then `vllm-sr logs router` and report
  what you saw.
- **Stop and ask:** a fix that needs system-wide changes, the user's data or a
  stack you didn't start.

### Optimize

- **Preconditions:** a verified stack; an objective (quality, latency or cost)
  with its acceptable trade-offs; a frozen baseline.
- **Run:** [Optimization](https://vllm-sr.ai/install/agent/vllm-sr/references/optimization.md), with
  [recipe tuning](https://vllm-sr.ai/install/agent/vllm-sr/references/recipe-tuning.md) and [sr-bench](https://vllm-sr.ai/install/agent/vllm-sr/references/sr-bench.md) for the
  details.
- **Success:** an sr-bench run returns `"status": "completed"` with every
  target `"complete": true` and `"failed": 0`; a change is kept only when it
  beats the baseline on the same cases and the stated objective.
- **On failure:** the sr-bench rows of Troubleshooting.
- **Stop and ask:** new models, GPUs or paid API calls; a Standard or holdout
  run; any change to a production stack.

## Hand off

Report to the user:

- `vllm-sr --version`, the channel, platform and gateway you chose;
- the config path, and that `vllm-sr stop` stops the stack while keeping the
  config, `.vllm-sr/`, downloaded models, volumes and images;
- the inference API, `http://127.0.0.1:8899/v1` with model `vllm-sr/auto`, and
  the decision model and its state;
- the Dashboard, `http://127.0.0.1:8700`; from another machine, through
  `ssh -L 8700:127.0.0.1:8700 HOST`. Its first visitor creates the
  administrator account, so the user should open it now (or set
  `DASHBOARD_ADMIN_EMAIL` and `DASHBOARD_ADMIN_PASSWORD` before `serve`);
- what each check returned, and anything you skipped, changed or left failing.

For the full contract, use `vllm-sr --help`, command help, `vllm-sr config
schema`, and the running Router's `GET /api/v1`. The
[Quickstart](https://vllm-sr.ai/docs/installation),
[System One](https://vllm-sr.ai/docs/system-one/overview) and the
[v0.5 migration guide](https://vllm-sr.ai/docs/installation/migrate-to-v0-5)
describe the same stack for people.
