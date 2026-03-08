---
name: cross-stack-bugfix
category: primary
description: Use when fixing a multi-surface issue that does not map cleanly to a narrower project-level skill.
---

# Cross Stack Bugfix

## Trigger

- A bug spans multiple layers and no narrower primary skill clearly applies
- The fix needs coordinated changes across runtime, CLI, UI, platform, or test surfaces

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/feature-complete-checklist.md](../../../../docs/agent/feature-complete-checklist.md)
- [docs/agent/tech-debt-register.md](../../../../docs/agent/tech-debt-register.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu CHANGED_FILES="..."`

## Acceptance

- The final report explicitly names impacted surfaces and intentionally skipped conditional surfaces
- Any real code or spec mismatch left behind is promoted into the debt register
