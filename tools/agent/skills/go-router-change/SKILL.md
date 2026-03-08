---
name: go-router-change
category: legacy_reference
description: Legacy router-centric reference skill retained while project-level primary skills take over the main entry point.
---

# Legacy Go Router Change

## Trigger

- Use this only as a reference after a project-level primary skill is selected and the remaining work is centered on Go router implementation details

## Must Read

- Read [docs/agent/playbooks/go-router.md](../../../../docs/agent/playbooks/go-router.md)
- Prefer the `router-runtime` fragment for current runtime guidance

## Standard Commands

- Run `make agent-feature-gate ENV=cpu CHANGED_FILES="..."` for behavior changes

## Acceptance

- The active primary skill remains authoritative and this file only supplements Go-router-specific implementation guidance
