---
name: vllm-semantic-router-harness
description: Bridges native skill discovery into the vLLM Semantic Router repository harness, routing tasks through the canonical agent-report flow, repo-local skill registry, and validation commands. Use when starting any task inside the vLLM Semantic Router repository to resolve the correct primary skill, read canonical docs, and run harness validation.
---

# vLLM Semantic Router Harness

This is a native-discovery bridge. The canonical source of truth remains:

- [`AGENTS.md`](../../../AGENTS.md)
- [`docs/agent/README.md`](../../../docs/agent/README.md)
- [`tools/agent/skill-registry.yaml`](../../../tools/agent/skill-registry.yaml)
- [`tools/agent/maintainer-policy.yaml`](../../../tools/agent/maintainer-policy.yaml)
- [`tools/make/agent.mk`](../../../tools/make/agent.mk)

## When To Use

Use this skill whenever the current workspace is this repository or the task touches files under `src/`, `tools/agent/`, `docs/agent/`, `dashboard/`, or `.github/`.

## Required Flow

1. Read [`AGENTS.md`](../../../AGENTS.md) first.
2. Read [`docs/agent/README.md`](../../../docs/agent/README.md) and the nearest local `AGENTS.md` for hotspot directories.
3. Run `make agent-report ENV=cpu|amd CHANGED_FILES="..."` before non-trivial coding-agent edits.
4. Start from the resolved primary skill under [`tools/agent/skills/`](../../../tools/agent/skills/).
5. For maintainer issue, PR, or release work, use [`docs/agent/maintainer-ops.md`](../../../docs/agent/maintainer-ops.md) and generated local state under `.agent-harness/maintainer/`.
6. Validate with the canonical harness commands from [`tools/make/agent.mk`](../../../tools/make/agent.mk).

## Boundaries

- Do not treat this wrapper as canonical policy text.
- Do not copy or fork skill instructions out of `tools/agent/**`.
- Use `make agent-report` to expose the full routed context pack; this wrapper only improves native startup discovery.
