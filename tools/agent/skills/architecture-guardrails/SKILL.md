---
name: architecture-guardrails
category: fragment
description: Structural limits, dependency boundaries, interface placement, and preferred composition-oriented design patterns.
---

# Architecture Guardrails

## Trigger

- Load this fragment for any non-trivial code change

## Must Read

- [docs/agent/architecture-guardrails.md](../../../../docs/agent/architecture-guardrails.md)
- [docs/agent/module-boundaries.md](../../../../docs/agent/module-boundaries.md)
- [tools/agent/structure-rules.yaml](../../structure-rules.yaml)
- [tools/agent/scripts/structure_check.py](../../scripts/structure_check.py)

## Standard Commands

- `make agent-lint CHANGED_FILES="..."`

## Acceptance

- Changed code passes structure rules and stays modular
