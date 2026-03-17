---
name: cross-stack-bugfix
category: primary
description: Diagnoses and fixes bugs that span multiple layers (runtime, CLI, UI, platform, tests) requiring coordinated changes across surfaces. Use when a bug does not map cleanly to a narrower skill, the fix touches more than one surface, or changes need cross-cutting validation.
---

# Cross Stack Bugfix

## Trigger

- A bug spans multiple layers and no narrower primary skill clearly applies
- The fix needs coordinated changes across runtime, CLI, UI, platform, or test surfaces

## Workflow

1. Read change surfaces and feature-complete checklist to identify all impacted layers
2. Diagnose the bug across affected surfaces (runtime, CLI, UI, platform, tests)
3. Implement coordinated fixes across all impacted surfaces
4. Run `make agent-report ENV=cpu CHANGED_FILES="..."` to verify all surfaces are accounted for
5. Promote any remaining mismatches into indexed debt entries

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/feature-complete-checklist.md](../../../../docs/agent/feature-complete-checklist.md)
- [docs/agent/tech-debt-register.md](../../../../docs/agent/tech-debt-register.md)
- [docs/agent/tech-debt/README.md](../../../../docs/agent/tech-debt/README.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu CHANGED_FILES="..."`

## Acceptance

- The final report explicitly names impacted surfaces and intentionally skipped conditional surfaces
- Any real code or spec mismatch left behind is promoted into the matching indexed debt entry
