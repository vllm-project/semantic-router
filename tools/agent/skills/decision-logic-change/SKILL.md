---
name: decision-logic-change
category: primary
description: Use when changing boolean control logic that combines signals and route conditions into decisions.
---

# Decision Logic Change

## Trigger

- Change decision predicates, thresholds, gates, or priority-driven routing branches
- Change how signals are combined into boolean control logic

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/module-boundaries.md](../../../../docs/agent/module-boundaries.md)
- [docs/agent/playbooks/go-router.md](../../../../docs/agent/playbooks/go-router.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu CHANGED_FILES="..."`

## Acceptance

- Boolean decision behavior is covered by targeted tests and affected E2E profiles
- Signal extraction and decision ownership stay separated by layer
