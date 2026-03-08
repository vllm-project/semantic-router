---
name: feature-complete-checklist
category: support
description: Repository-standard completion checklist and reporting shape before closing a task.
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

## Must Read

- [docs/agent/feature-complete-checklist.md](../../../../docs/agent/feature-complete-checklist.md)

## Standard Commands

- `make agent-ci-gate CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu|amd CHANGED_FILES="..."`

## Acceptance

- Final report includes primary skill, impacted surfaces, validation results, any skipped conditional surfaces, and any durable debt item created or updated because the work left a known architecture gap behind
