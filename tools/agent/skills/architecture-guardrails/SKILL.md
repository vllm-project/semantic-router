---
name: architecture-guardrails
category: support
description: Runs an explicit structural review for boundary-sensitive edits that need dependency, module, and structure-rule scrutiny without forcing that material into every default context pack. Use when a task crosses hotspot boundaries, adds new seams, or needs a deliberate architecture pass.
---

# Architecture Guardrails

## Trigger

- Use when a task crosses hotspot boundaries, adds a new seam, or needs an explicit structure review
- Use when default primary or fragment guidance is not enough to judge dependency direction or interface placement

## Workflow

1. Read architecture guardrails, module boundaries, and structure rules for applicable constraints
2. Make code changes while respecting dependency boundaries and interface placement
3. Run `make agent-lint CHANGED_FILES="..."` to validate structure rules
4. Verify changed code passes all structure rules and stays modular

## Gotchas

- Do not load this by reflex. Most tasks already carry the relevant subsystem docs through their primary skill and surface refs.
- Use it when the change genuinely needs a structural review, not as a substitute for understanding the owning subsystem.

## Must Read

- [docs/agent/architecture-guardrails.md](../../../../docs/agent/architecture-guardrails.md)
- [tools/agent/structure-rules.yaml](../../structure-rules.yaml)

## Standard Commands

- `make agent-lint CHANGED_FILES="..."`

## Acceptance

- Changed code passes structure rules and stays modular
