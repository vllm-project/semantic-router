---
name: e2e-profile-selection
category: legacy_reference
description: Legacy E2E mapping reference skill retained while `e2e-selection` becomes the active fragment skill.
---

# Legacy E2E Profile Selection

## Trigger

- Use this only as a reference after a project-level primary skill is selected and you need the old E2E mapping entrypoint

## Must Read

- Prefer [tools/agent/skills/e2e-selection/SKILL.md](../e2e-selection/SKILL.md)

## Standard Commands

- Run `make agent-e2e-affected CHANGED_FILES="..."`

## Acceptance

- The active primary or fragment skill remains authoritative and this file only redirects to the current E2E selection path
