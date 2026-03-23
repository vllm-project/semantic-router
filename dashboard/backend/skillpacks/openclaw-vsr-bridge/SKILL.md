---
name: openclaw-vsr-bridge
description: Install vLLM Semantic Router in agent-safe mode, import supported OpenClaw model providers into canonical VSR config, and rewrite OpenClaw to target VSR.
user-invocable: true
---

# OpenClaw VSR Bridge

## When To Use

- Use this skill when an OpenClaw agent should install VSR without auto-launching the dashboard.
- Use this skill when an existing `openclaw.json` should be moved under VSR management.
- Use this skill when you want the OpenClaw model base URL rewritten only after canonical `config.yaml` is written successfully.

## Requirements

- `curl`
- `bash`
- write access to the OpenClaw config path and the target VSR config path

## Workflow

1. Check whether VSR is already installed with `command -v vllm-sr`.
2. If VSR is missing, run the supported installer in agent-safe mode:

   ```bash
   curl -fsSL https://vllm-semantic-router.com/install.sh | bash -s -- --mode cli --runtime skip --no-launch
   ```

3. Locate the OpenClaw config in this order unless the user already gave you a path:
   - `$OPENCLAW_CONFIG_PATH`
   - `./openclaw.json`
   - `~/.openclaw/openclaw.json`
4. Import the OpenClaw model providers into canonical VSR config:

   ```bash
   vllm-sr config import --from openclaw --source <openclaw.json> --target config.yaml
   ```

5. Do not pass `--force` unless the user explicitly approves overwriting existing backup files.
6. After a successful import, report:
   - the imported logical model names
   - the canonical target config path
   - the rewritten OpenClaw config path
   - the rewritten OpenClaw base URL
   - the backup file paths, if any
7. Suggest the next validation commands:

   ```bash
   vllm-sr validate config.yaml
   vllm-sr config router --config config.yaml
   ```

8. If the user wants runtime verification after validation, suggest:

   ```bash
   vllm-sr serve --config config.yaml
   curl http://127.0.0.1:8899/v1/models
   ```

## Boundaries

- Do not invent a second installer path. Use the supported installer above.
- Do not rewrite OpenClaw before the VSR target config has been written successfully.
- Do not import OpenClaw rooms, teams, browser settings, command settings, or memory history into VSR.
