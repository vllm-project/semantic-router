---
name: claude-code-install
category: support
description: Guides coding agents through a session-isolated install of the vllm-sr CLI. The install is scoped to a temporary directory so it does not modify the user's global PATH or system packages. Use when an agent or user wants to try vllm-sr without a permanent install.
---

# Claude Code Session-Isolated Install

## Trigger

- User or agent wants to install and try `vllm-sr` CLI without permanent system changes
- A coding agent needs a quick, disposable install for validation or exploration

## Required Surfaces

- `cli_install`

## Stop Conditions

- The host lacks `bash`, `curl`, or `python3` (install.sh prerequisites)
- The user explicitly wants a permanent/global install (use `install.sh` directly instead)

## Workflow

1. Run the session-isolated wrapper script:

   ```bash
   bash tools/agent/scripts/cc-install.sh
   ```

   This creates a temp directory, installs the CLI there, and prints the `export PATH` line.

2. Export the PATH as printed:

   ```bash
   export PATH="/tmp/vllm-sr-session-<tag>/bin:$PATH"
   ```

3. Validate the install:

   ```bash
   vllm-sr --version
   vllm-sr validate          # if a config.yaml exists
   ```

4. When done, the session directory is cleaned up automatically on reboot, or remove it manually:

   ```bash
   rm -rf /tmp/vllm-sr-session-<tag>
   ```

## Passing Options

Any extra flags are forwarded to `install.sh`. For example:

```bash
bash tools/agent/scripts/cc-install.sh --runtime docker
```

The wrapper only sets defaults for `--mode`, `--runtime`, `--install-root`, `--bin-dir`, and `--no-launch` when they are not already present in the caller's arguments.

## Gotchas

- The wrapper only sets defaults when flags are absent. If you pass `--mode` or `--runtime` explicitly, the wrapper does not override them.
- PID-based session tags (`$$`) can collide if the OS reuses PIDs quickly. Set `VLLM_SR_SESSION_TAG` to a unique value for concurrent installs.
- If `install.sh` fails mid-install, the trap automatically cleans up the session directory. Do not rely on its partial contents.

## Must Read

- [install.sh](../../../../install.sh)
- [tools/agent/scripts/cc-install.sh](../../scripts/cc-install.sh)

## Standard Commands

- `bash tools/agent/scripts/cc-install.sh`
- `vllm-sr --version`
- `vllm-sr validate`

## Acceptance

- `vllm-sr --version` succeeds after running the wrapper and exporting PATH
- No files are written outside the session-scoped temp directory
- The user's global PATH and system packages are unchanged
