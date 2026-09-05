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
| Existing `vllm-sr` | `command -v vllm-sr`, `test -e ~/.local/bin/vllm-sr \|\| test -L ~/.local/bin/vllm-sr`, or `test -d ~/.local/share/vllm-sr` |
| Existing `vllm-sr` version | `vllm-sr --version` or `~/.local/bin/vllm-sr --version` (diagnostic only) |
| Installer path overrides | `echo "${VLLM_SR_INSTALL_ROOT:-}"` and `echo "${VLLM_SR_BIN_DIR:-}"` |
| Installer package override | `test -n "${VLLM_SR_PIP_SPEC:-}"` — report **set/unset only**; never print the value |
| Docker | `command -v docker` and `docker info` |
| Podman | `command -v podman` and `podman info` |
| Existing config | `test -f config.yaml` in the working directory |
| Existing local runtime state | `test -f ~/.local/share/vllm-sr/runtime.env` |

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
start the serving stack or launch the Dashboard. When invoking the installer,
unset `VLLM_SR_PIP_SPEC` first (see Workflow) so an inherited package override
cannot change what gets installed.

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

The installer also sets up shell completions by default, which may edit the
user's shell rc files (for example `~/.bashrc`, `~/.zshrc`, or equivalent).
Disclose this side effect in the plan so the user can approve it explicitly.
If the user wants to avoid shell rc changes, they can run the installer
interactively and skip the completion step when prompted, or set up
completions manually later via `vllm-sr completion install`.

## Workflow

1. **Discover** the environment using the checks above.
2. **Check for an existing installation.** Non-interactive agent shells
   often omit `~/.local/bin` from `PATH`, and a stale non-executable file or
   a dangling symlink at the launcher path is still an occupied install
   location. Check **all** applicable signals:

   **Default paths:**
   - `command -v vllm-sr` (launcher on PATH)
   - `test -e ~/.local/bin/vllm-sr || test -L ~/.local/bin/vllm-sr` (default
     absolute launcher; `test -e` catches regular files, `test -L` catches
     dangling symlinks even when their target is gone)
   - `test -d ~/.local/share/vllm-sr` (default install root)

   **Override paths (only when the corresponding env var is set):**
   - If `VLLM_SR_BIN_DIR` is set: `test -e "$VLLM_SR_BIN_DIR/vllm-sr" ||
     test -L "$VLLM_SR_BIN_DIR/vllm-sr"`
   - If `VLLM_SR_INSTALL_ROOT` is set: `test -d "$VLLM_SR_INSTALL_ROOT"`

   **Any signal found** — an existing installation is present. Attempt
   `vllm-sr --version` or `~/.local/bin/vllm-sr --version` for diagnostics.
   Whether or not version succeeds, **stop and report**; do not reinstall,
   overwrite, or repair without explicit user approval.
   **No signal found** — continue to step 3.

   **Package override** — if `test -n "${VLLM_SR_PIP_SPEC:-}"` shows that
   `VLLM_SR_PIP_SPEC` is set, **stop and report its presence without
   printing the value.** The installer uses any non-empty value as the pip
   package spec and would install an alternate package instead of the
   official vLLM-SR CLI, which is outside the supported installation path.
   The value may embed credentials such as private index tokens, so it
   must never be echoed, logged, or included in the report.
3. **Present the plan** and wait for confirmation if the user has not already
   approved.
4. **Install** using the one-line installer in agent-safe mode, explicitly
   unsetting the package override so it cannot be inherited by the
   installer's shell:

   ```bash
   unset VLLM_SR_PIP_SPEC
   curl -fsSL https://vllm-sr.ai/install.sh | bash -s -- --mode cli --runtime skip --no-launch
   ```
5. **Validate** (see Validation below).
6. **Report** the result and the next supported step.

## Validation

After installation, verify:

```bash
vllm-sr --version
```

If `~/.local/bin` is not on `PATH` (common in non-interactive agent shells),
validate via the launcher's absolute path instead:

```bash
~/.local/bin/vllm-sr --version
```

A successful install prints a version string. Do not claim success without
this verification. If neither command works, report the failure and suggest
adding `~/.local/bin` to `PATH`.

## Existing Installation, Runtime State, Configuration, and Deployment

### Existing CLI installation

An existing installation is detected by **any** of these signals:

**Default paths:**
- `command -v vllm-sr` finds the launcher on `PATH`.
- `test -e ~/.local/bin/vllm-sr || test -L ~/.local/bin/vllm-sr` finds the
  default absolute launcher (common when `~/.local/bin` is absent from the
  non-interactive shell `PATH`). `test -e` catches any regular file even
  without the executable bit; `test -L` catches dangling symlinks whose
  target is gone but still occupy the launcher location.
- `test -d ~/.local/share/vllm-sr` finds the default install root.

**Override paths (only when the corresponding env var is set):**
- If `VLLM_SR_BIN_DIR` is set: `test -e "$VLLM_SR_BIN_DIR/vllm-sr" ||
  test -L "$VLLM_SR_BIN_DIR/vllm-sr"`.
- If `VLLM_SR_INSTALL_ROOT` is set: `test -d "$VLLM_SR_INSTALL_ROOT"`.

If any signal is found:

- Report which signal(s) were detected and the launcher path if available.
- Attempt `vllm-sr --version` or `~/.local/bin/vllm-sr --version` for diagnostics.
  - If version succeeds, report the version.
  - If version fails, report that the installation exists but its version could
    not be verified.
- In **all** cases, stop and do not reinstall, overwrite, or repair without
  explicit user approval.
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
- The installer reads `VLLM_SR_INSTALL_ROOT`, `VLLM_SR_BIN_DIR`, and
  `VLLM_SR_PIP_SPEC` from the environment. If a path override is set,
  discovery must check the override path too, not only the defaults.
- `VLLM_SR_PIP_SPEC` changes **which package** the installer installs. If
  it is set, stop and report its presence; never print the value, because
  package specs may contain credentials such as private index tokens.
- The installer unconditionally sets up shell completions, which may edit
  `~/.bashrc`, `~/.zshrc`, or equivalent shell rc files. Disclose this in
  the plan so the user can approve the change.

## Must Read

- [Quickstart](https://vllm-sr.ai/docs/installation/installation)
- [Deployment Options](https://vllm-sr.ai/docs/installation/deployment-options)
- [Troubleshooting](https://vllm-sr.ai/docs/troubleshooting/common-errors)

## Standard Commands

- `unset VLLM_SR_PIP_SPEC && curl -fsSL https://vllm-sr.ai/install.sh | bash -s -- --mode cli --runtime skip --no-launch`
- `vllm-sr --version`
- If `~/.local/bin` is not on PATH, use `~/.local/bin/vllm-sr --version`.

## Acceptance

- The skill selects only supported installation paths from the public installer
  contract.
- A bounded plan is shown before any mutation.
- The installation is validated with `vllm-sr --version`.
- Existing installations, configurations, and deployments are not overwritten
  without explicit approval.
- A set `VLLM_SR_PIP_SPEC` stops the flow, its value is never printed, and the
  installer is run with the variable explicitly unset.
- `runtime.env` is reported as runtime state, not conflated with an active
  deployment.
- No credentials, private endpoints, or secret values appear in skill output.
- The skill stops before configuration generation, evaluation, tuning,
  activation, or rollback.
