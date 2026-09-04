---
name: vllm-sr-install
category: support
description: Harness-neutral public installation skill for vLLM Semantic Router. Use when an agent must discover the user environment, choose a maintained installation path, show a bounded plan, install through the supported installer, validate the result, and stop before unsafe or out-of-scope mutation.
---

# vLLM Semantic Router Agent Install

This skill is published at
<https://vllm-sr.ai/install/agent/vllm-sr/SKILL.md>
and is usable by any compatible coding agent or agent harness that can
consume the public skill document — no repository clone required.

## Trigger

- A user asks the agent to install vLLM Semantic Router.
- A user asks the agent to check whether vLLM Semantic Router is installed.
- A user pastes the website Agent install prompt.

## Scope

- Discover the user environment.
- Choose a supported installation path from the public installer contract.
- Show a bounded plan before changing anything.
- Install vLLM Semantic Router through the maintained installer.
- Validate the installation.
- Explain the next supported step.
- Stop before configuration generation, evaluation, tuning, activation, or
  rollback — those are separate journeys tracked by other issues.

Out of scope:

- Generating or applying Router configuration or Recipes.
- Running evaluation or tuning.
- Activating production traffic or performing rollback.
- Modifying an existing deployment without explicit user approval.
- Creating a second installer or new deployment semantics.

## Safety Boundary

- **Plan first.** Show the user what will happen before any mutation.
- **Ask before overwriting.** If an existing installation, configuration, or
  deployment is detected, stop and report; do not reinstall, rewrite, migrate,
  or redeploy without explicit approval.
- **No credential exposure.** Never print secret values, write credentials into
  skill output, commit secrets to a repository, or include private endpoints in
  public artifacts.
- **No implicit production activation.** Agent-driven initial install uses
  CLI-only, no-launch mode. Starting or modifying a live deployment requires a
  separate explicit decision.
- **Unsupported environments.** If the environment does not match a supported
  path, report it and stop. Do not invent an installation path.

## Environment Discovery

Before choosing an installation path, detect and report:

| Check | How |
| --- | --- |
| Operating system | `uname -s` |
| Architecture | `uname -m` |
| Python ≥ 3.10 | `python3 --version` or `python --version` |
| Existing `vllm-sr` | `command -v vllm-sr` and `vllm-sr --version` |
| Docker | `command -v docker` and `docker info` |
| Podman | `command -v podman` and `podman info` |
| Existing config | `ls config.yaml` in the working directory |
| Existing local runtime state | `ls ~/.local/share/vllm-sr/runtime.env` |

Docker / Podman discovery is **informational only**. It may be useful for
later deployment decisions, but it is not used to silently select or activate a
runtime during the initial agent-safe installation. The installation step in
this skill always uses `--runtime skip`, so neither Docker nor Podman is a
selection condition for the CLI installation path.

If the working directory contains a `tools/agent/repo-manifest.yaml` file, the
agent MAY read it for repository-native supported paths. Otherwise, the agent
must rely solely on the public installer contract described below.

## Supported Installation Paths

### Primary path — one-line installer

```bash
curl -fsSL https://vllm-sr.ai/install.sh | bash -s -- --mode cli --runtime skip --no-launch
```

This installs the CLI into an isolated virtual environment under
`~/.local/share/vllm-sr`, links a launcher into `~/.local/bin`, and does **not**
start the serving stack or launch the Dashboard.

The installer accepts these relevant flags:

| Flag | Values | Default | Notes |
| --- | --- | --- | --- |
| `--mode` | `cli` \| `serve` | `serve` | `cli` installs the CLI only. |
| `--runtime` | `auto` \| `docker` \| `skip` | `auto` | `skip` disables container runtime detection. |
| `--no-launch` | flag | off | Skip the automatic first `vllm-sr serve`. |
| `--install-root` | path | `~/.local/share/vllm-sr` | Installation root. |
| `--bin-dir` | path | `~/.local/bin` | Launcher directory. |

For agent-driven initial install, prefer `--mode cli --runtime skip --no-launch`
to avoid implicit container or serve behaviour. The user can later run
`vllm-sr serve` to start the full local stack.

The installer also supports `--channel stable|dev`. The public skill does not
override the installer's channel default — the installer is responsible for
its own release-channel semantics. Do not pass `--channel` unless the user
explicitly requests a specific channel.

## Plan Before Mutation

After environment discovery and before running any installation command, present
a plan to the user. Example:

```text
Detected environment:
- Linux x86_64
- Python 3.12
- Docker available
- No existing vllm-sr installation

Plan:
1. Install the vLLM-SR CLI via the one-line installer (CLI-only, no launch).
2. Verify with `vllm-sr --version`.
3. Stop before configuration or serving — the user decides the next step.
```

Do not proceed until the user confirms, unless the user already gave explicit
instructions to install.

## Workflow

1. **Discover** the environment using the checks above.
2. **Detect existing installation.** If `vllm-sr --version` succeeds, report the
   version and stop. Do not reinstall unless the user explicitly asks.
3. **Present the plan** and wait for confirmation if the user has not already
   approved.
4. **Install** using the one-line installer in agent-safe mode.
5. **Validate** (see Validation below).
6. **Report** the result and the next supported step.

## Validation

After installation, verify:

```bash
vllm-sr --version
```

A successful install prints a version string. If the command is not found, the
launcher directory (`~/.local/bin`) may not be on `PATH` — report this to the
user and suggest adding it.

Do not claim success without this verification.

## Existing Installation, Runtime State, Configuration, and Deployment

### Existing CLI installation

If `vllm-sr --version` succeeds:

- Report the installed version.
- Do not reinstall.
- If the user asks to upgrade, point them to the installation docs at
  <https://vllm-sr.ai/docs/installation/installation>.

### Existing local runtime state

If `~/.local/share/vllm-sr/runtime.env` exists, report it as **runtime state**,
not as proof of an active deployment:

```text
Existing vLLM-SR runtime state was detected at ~/.local/share/vllm-sr/runtime.env.
This does not by itself prove that a deployment is currently active.
No runtime state was modified.
```

Continue to the next checks (existing configuration, active deployment signals)
rather than stopping the journey solely because `runtime.env` exists.

### Existing configuration

If a `config.yaml` exists in the working directory:

- Do not rewrite, replace, or migrate it.
- Report its presence.
- The user may run `vllm-sr validate --config config.yaml` separately.

### Active deployment

Only treat the environment as having an **active deployment** when strong
signals are present — for example running vLLM-SR containers, a clearly active
listener or process associated with vLLM-SR, or other unambiguous deployment
indicators. The presence of `runtime.env` alone is **not** sufficient.

When an active deployment is detected:

```text
An active deployment was detected.
No changes were made.
Changing an active deployment requires explicit user approval and is outside
the scope of this installation skill.
```

Do not redeploy, upgrade, activate, or roll back.

## Unsupported States

- **Windows (native):** The CLI can install and validate configs, but the local
  Docker serving workflow requires WSL2 or another Linux environment. Report
  this and stop.
- **Python < 3.10:** Report the detected version and stop.
- **No Python:** Report that Python 3.10+ is required and stop.
- **Unknown OS or architecture:** Report and stop.

## Recovery

If installation fails:

1. Report the exact error output from the installer.
2. Do not retry blindly — diagnose the cause.
3. Common causes: network failure, missing `curl`, insufficient permissions for
   `~/.local/bin`, or Python version mismatch.
4. Direct the user to
  [Troubleshooting](https://vllm-sr.ai/docs/troubleshooting/common-errors).

## Next Supported Step

After a successful install:

- **Start the local stack:** `vllm-sr serve`
- **Choose a deployment:** <https://vllm-sr.ai/docs/installation/deployment-options>
- **Configuration guide:** <https://vllm-sr.ai/docs/installation/configuration>
- **Quickstart:** <https://vllm-sr.ai/docs/installation/installation>

Do not proceed into configuration generation, evaluation, or activation without
explicit user direction — those are separate workflows.

## Gotchas

- The one-line installer's default mode (`serve`) starts the full stack. For
  agent-driven install, always pass `--mode cli --runtime skip --no-launch`.
- The installer persists the selected container runtime to
  `~/.local/share/vllm-sr/runtime.env`. Passing `--runtime skip` prevents
  unintended container detection.
- `runtime.env` records runtime state, not an active deployment. Do not infer
  deployment status from its presence alone.
- `vllm-sr serve` starts the routing stack only; provider backends must already
  be reachable.
- Do not confuse installing the CLI with starting or configuring a deployment.

## Must Read

- [Quickstart](https://vllm-sr.ai/docs/installation/installation)
- [Deployment Options](https://vllm-sr.ai/docs/installation/deployment-options)
- [Troubleshooting](https://vllm-sr.ai/docs/troubleshooting/common-errors)

## Standard Commands

- `curl -fsSL https://vllm-sr.ai/install.sh | bash -s -- --mode cli --runtime skip --no-launch`
- `vllm-sr --version`

## Acceptance

- The skill selects only supported installation paths from the public installer
  contract.
- A bounded plan is shown before any mutation.
- The installation is validated with `vllm-sr --version`.
- Existing installations, configurations, and deployments are not overwritten
  without explicit approval.
- `runtime.env` is reported as runtime state, not conflated with an active
  deployment.
- No credentials, private endpoints, or secret values appear in skill output.
- The skill stops before configuration generation, evaluation, tuning,
  activation, or rollback.
