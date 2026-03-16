---
name: feature-complete-checklist
category: support
description: Runs the repository-standard completion checklist before closing a task, verifying all surfaces are validated, E2E profiles pass, and any remaining gaps are documented as tech debt. Use when a primary skill is nearly done and the close-out report needs to be generated.
---

# Feature Complete Checklist

## Trigger

- A primary skill is nearly done and you need the close-out checklist

## Required Surfaces

- `local_e2e`
- `ci_e2e`

## Conditional Surfaces

- `local_smoke`

## Stop Conditions

- Validation gaps are still unresolved or intentionally skipped without explanation

## Workflow

1. Read the feature-complete checklist to understand required close-out steps
2. Run `make agent-ci-gate CHANGED_FILES="..."` to validate CI constraints
3. Run `make agent-feature-gate ENV=cpu|amd CHANGED_FILES="..."` to verify feature readiness
4. Generate the final report with primary skill, impacted surfaces, validation results, and any debt entries

## Must Read

- [docs/agent/feature-complete-checklist.md](../../../../docs/agent/feature-complete-checklist.md)

## Standard Commands

- `make agent-ci-gate CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu|amd CHANGED_FILES="..."`

## Acceptance

- Final report includes primary skill, impacted surfaces, validation results, any skipped conditional surfaces, and any durable debt item created or updated because the work left a known architecture gap behind
