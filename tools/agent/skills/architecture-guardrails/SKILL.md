---
name: architecture-guardrails
category: fragment
description: Enforces structural rules, dependency boundaries, interface placement, and composition-oriented design patterns across the codebase. Use when making non-trivial code changes to verify module boundaries, check dependency direction, or validate that new code follows the project's architecture conventions.
---

# Architecture Guardrails

## Trigger

- Load this fragment for any non-trivial code change

## Workflow

1. Read architecture guardrails, module boundaries, and structure rules for applicable constraints
2. Make code changes while respecting dependency boundaries and interface placement
3. Run `make agent-lint CHANGED_FILES="..."` to validate structure rules
4. Verify changed code passes all structure rules and stays modular

## Must Read

- [docs/agent/architecture-guardrails.md](../../../../docs/agent/architecture-guardrails.md)
- [docs/agent/module-boundaries.md](../../../../docs/agent/module-boundaries.md)
- [tools/agent/structure-rules.yaml](../../structure-rules.yaml)
- [tools/agent/scripts/structure_check.py](../../scripts/structure_check.py)

## Standard Commands

- `make agent-lint CHANGED_FILES="..."`

## Acceptance

- Changed code passes structure rules and stays modular
