---
name: vllm-semantic-router-harness
description: Use for any task inside the vLLM Semantic Router repository. It bridges native skill discovery into the repo harness entrypoint, canonical agent-report routing flow, repo-local skill registry, and validation commands without duplicating the canonical rules.
---

# vLLM Semantic Router Harness

This is a native-discovery bridge. The canonical source of truth remains:

- [`AGENTS.md`](../../../AGENTS.md)
- [`docs/agent/README.md`](../../../docs/agent/README.md)
- [`tools/agent/skill-registry.yaml`](../../../tools/agent/skill-registry.yaml)
- [`tools/make/agent.mk`](../../../tools/make/agent.mk)

## When To Use

Use this skill whenever the current workspace is this repository or the task touches files under `src/`, `tools/agent/`, `docs/agent/`, `dashboard/`, or `.github/`.

## Required Flow

1. Read [`AGENTS.md`](../../../AGENTS.md) first.
2. Read [`docs/agent/README.md`](../../../docs/agent/README.md) and the nearest local `AGENTS.md` for hotspot directories.
3. Run `make agent-report ENV=cpu|amd CHANGED_FILES="..."` before non-trivial edits.
4. Start from the resolved primary skill under [`tools/agent/skills/`](../../../tools/agent/skills/).
5. Validate with the canonical harness commands from [`tools/make/agent.mk`](../../../tools/make/agent.mk).

## Boundaries

- Do not treat this wrapper as canonical policy text.
- Do not copy or fork skill instructions out of `tools/agent/**`.
- Use `make agent-report` to expose the full routed context pack; this wrapper only improves native startup discovery.
